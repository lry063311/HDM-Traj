#!/usr/bin/env python
"""Diagnostic Table III/IV geometry checks for route-memory reranker outputs.

This script is intentionally diagnostic. The route-memory reranker currently
exports trajectories, not a full semantic/counterfactual decoder interface.
Table III is computed from generated trajectories and instruction labels.
Table IV uses a no-semantic-change diagnostic: for a counterfactual record with
the same source OD, the route-memory output is treated as unchanged unless a
separate counterfactual output is provided by a future model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langdirecttraj.counterfactual_policy import independent_collateral_field_names  # noqa: E402
from langdirecttraj.data.compact_store import (  # noqa: E402
    expand_compact_record,
    is_compact_meta_path,
    load_compact_manifest,
    open_compact_arrays,
)
from langdirecttraj.data.instruction_dataset import InstructionTrajectoryDataset  # noqa: E402
from langdirecttraj.eval.export_model_predictions import reference_to_local, resample_np  # noqa: E402
from langdirecttraj.eval.instruction_geometry import (  # noqa: E402
    GeometryTokenCalibrator,
    add_pace_metrics_to_rows,
    apply_pace_progress_labels,
    contract_labels_from_tensors,
    geometry_values,
    macro_f1,
    tensor_dict_to_rows,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--split-input", required=True, type=Path)
    parser.add_argument("--generated-npz", required=True, type=Path)
    parser.add_argument("--real-npz", required=True, type=Path)
    parser.add_argument("--counterfactual-input", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--counterfactual-scan-limit", type=int, default=200000)
    parser.add_argument("--target-len", type=int, default=128)
    parser.add_argument("--include-empty-labels", action="store_true")
    return parser.parse_args()


def summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def load_generated(path: Path, real_path: Path, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    if "generated_trajectories" not in payload:
        raise ValueError(f"{path} must contain generated_trajectories")
    generated = np.asarray(payload["generated_trajectories"], dtype=np.float32)
    real_payload = np.load(real_path)
    if "sample_id" not in real_payload:
        raise ValueError(f"{real_path} must contain sample_id")
    sample_ids = np.asarray(real_payload["sample_id"], dtype=np.int64)
    n = generated.shape[0]
    if sample_ids.shape[0] < n:
        raise ValueError(f"real sample_id shorter than generated: {sample_ids.shape[0]} < {n}")
    n = min(n, int(max_samples)) if int(max_samples) > 0 else n
    return generated[:n], sample_ids[:n]


def record_map(path: Path, sample_ids: np.ndarray, target_key: str = "trajectory") -> Dict[int, Dict]:
    wanted = {int(x) for x in sample_ids.tolist()}
    out: Dict[int, Dict] = {}
    dataset = InstructionTrajectoryDataset(path, max_samples=0, target_key=target_key)
    for idx in range(len(dataset)):
        record = dataset.get_record(idx)
        sid = int(record.get("sample_id", -1))
        if sid in wanted:
            out[sid] = record
            if len(out) >= len(wanted):
                break
    missing = sorted(wanted.difference(out))
    if missing:
        raise ValueError(f"Missing {len(missing)} sample ids in {path}; first missing={missing[:5]}")
    return out


def localize(points_lonlat: np.ndarray, reference_lonlat: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray, float]:
    ref_norm, origin, scale, route_scale = reference_to_local(reference_lonlat)
    points = resample_np(np.asarray(points_lonlat, dtype=np.float32), target_len)
    points_m = (points - origin[None, :]) * scale[None, :]
    points_norm = points_m / max(float(route_scale), 1e-6)
    return points_norm.astype(np.float32), resample_np(ref_norm, target_len), float(route_scale)


def geometry_row(points_lonlat: np.ndarray, record: Mapping, target_len: int) -> Dict[str, float]:
    reference = np.asarray(record["reference"], dtype=np.float32)
    points_norm, reference_norm, route_scale = localize(points_lonlat, reference, target_len)
    points_t = torch.from_numpy(points_norm[None]).float()
    reference_t = torch.from_numpy(reference_norm[None]).float()
    scale_t = torch.tensor([route_scale], dtype=torch.float32)
    row = tensor_dict_to_rows(geometry_values(points_t, reference_t, scale_t))[0]
    return add_pace_metrics_to_rows([row], [record.get("trip_attributes", {})])[0]


def build_calibrator(records: List[Mapping], target_len: int) -> GeometryTokenCalibrator:
    rows: List[tuple[Dict[str, str], Dict[str, float]]] = []
    for record in records:
        rows.append(
            (
                dict(record.get("labels", {})),
                geometry_row(np.asarray(record["trajectory"], dtype=np.float32), record, target_len),
            )
        )
    return GeometryTokenCalibrator.from_target_batches(rows)


def evaluate_table3(args: argparse.Namespace) -> Dict:
    generated, sample_ids = load_generated(args.generated_npz, args.real_npz, args.max_samples)
    records_by_id = record_map(args.split_input, sample_ids)
    records = [records_by_id[int(sid)] for sid in sample_ids.tolist()]
    calibrator = build_calibrator(records, int(args.target_len))

    true_by_field: Dict[str, List[str]] = {field: [] for field in DEFAULT_MOBILITY_VOCAB.fields}
    pred_by_field: Dict[str, List[str]] = {field: [] for field in DEFAULT_MOBILITY_VOCAB.fields}
    for traj, record in zip(generated, records):
        reference = np.asarray(record["reference"], dtype=np.float32)
        pred_norm, ref_norm, route_scale = localize(traj, reference, int(args.target_len))
        pred_t = torch.from_numpy(pred_norm[None]).float()
        ref_t = torch.from_numpy(ref_norm[None]).float()
        scale_t = torch.tensor([route_scale], dtype=torch.float32)
        geom = tensor_dict_to_rows(geometry_values(pred_t, ref_t, scale_t))[0]
        geom = add_pace_metrics_to_rows([geom], [record.get("trip_attributes", {})])[0]
        pred_labels = contract_labels_from_tensors(pred_t, ref_t, scale_t, dataset=args.dataset)[0]
        pred_labels = apply_pace_progress_labels(
            [pred_labels],
            [geom],
            [record.get("trip_attributes", {})],
        )[0]
        labels = record.get("labels", {})
        for field in DEFAULT_MOBILITY_VOCAB.fields:
            true_by_field[field].append(str(labels.get(field, "unknown")))
            pred_by_field[field].append(str(pred_labels[field]))

    field_metrics = {
        field: macro_f1(
            true_by_field[field],
            pred_by_field[field],
            DEFAULT_MOBILITY_VOCAB.token_to_id[field].keys(),
            include_empty=bool(args.include_empty_labels),
        )
        for field in DEFAULT_MOBILITY_VOCAB.fields
    }
    table3 = {
        "distance_f1": float(field_metrics["distance"]["macro_f1"]),
        "direction_f1": float(field_metrics["direction"]["macro_f1"]),
        "curvature_f1": float(field_metrics["curvature"]["macro_f1"]),
        "progress_f1": float(field_metrics["progress"]["macro_f1"]),
        "reference_adherence_f1": float(field_metrics["reference_adherence"]["macro_f1"]),
    }
    table3["mean_instruction_following_f1"] = float(np.mean(list(table3.values())))
    return {
        "num_samples": int(len(records)),
        "sample_id_min": int(np.min(sample_ids)) if len(sample_ids) else None,
        "sample_id_max": int(np.max(sample_ids)) if len(sample_ids) else None,
        "table3_instruction_following": table3,
        "field_metrics": field_metrics,
        "diagnostic_note": "Computed from exported route-memory generated trajectories; no model semantic decoder was invoked.",
    }


def iter_expanded_records(path: Path):
    manifest = load_compact_manifest(path) if is_compact_meta_path(path) else None
    arrays = open_compact_arrays(manifest) if manifest else {}
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line.decode("utf-8"))
            if manifest is not None:
                record = expand_compact_record(record, arrays)
            yield record


def group_pairs_streaming(
    path: Path,
    available_source_ids: set[int],
    max_pairs: int,
    scan_limit: int,
) -> tuple[List[tuple[Dict, Dict]], Dict[str, int]]:
    originals: Dict[int, Dict] = {}
    counterfactuals: Dict[int, List[Dict]] = defaultdict(list)
    scanned = 0
    for record in iter_expanded_records(path):
        scanned += 1
        cf = record.get("counterfactual", {})
        if bool(cf.get("enabled", False)):
            source_id = int(cf.get("source_sample_id", record.get("sample_id", -1)))
            if source_id in available_source_ids:
                counterfactuals[source_id].append(record)
        else:
            sid = int(record.get("sample_id", -1))
            if sid in available_source_ids:
                originals[sid] = record
        if scanned >= int(scan_limit) > 0:
            break
        if max_pairs > 0:
            rough_pairs = sum(len(items) for sid, items in counterfactuals.items() if sid in originals)
            if rough_pairs >= max_pairs * 4:
                break

    fields = list(DEFAULT_MOBILITY_VOCAB.fields)
    buckets: Dict[str, List[tuple[Dict, Dict]]] = {field: [] for field in fields}
    for sid, cf_records in counterfactuals.items():
        original = originals.get(sid)
        if original is None:
            continue
        for cf_record in cf_records:
            field = str(cf_record.get("counterfactual", {}).get("field", ""))
            if field in buckets:
                buckets[field].append((original, cf_record))
    if max_pairs <= 0:
        pairs = [pair for field in fields for pair in buckets[field]]
        return pairs, {"scanned_records": int(scanned), "matched_pairs": int(len(pairs))}
    selected: List[tuple[Dict, Dict]] = []
    cursor = {field: 0 for field in fields}
    while len(selected) < max_pairs:
        added = False
        for field in fields:
            idx = cursor[field]
            if idx < len(buckets[field]):
                selected.append(buckets[field][idx])
                cursor[field] = idx + 1
                added = True
                if len(selected) >= max_pairs:
                    break
        if not added:
            break
    return selected, {"scanned_records": int(scanned), "matched_pairs": int(len(selected))}


def evaluate_table4(args: argparse.Namespace) -> Dict | None:
    if args.counterfactual_input is None:
        return None
    generated, sample_ids = load_generated(args.generated_npz, args.real_npz, args.max_samples)
    generated_by_id = {int(sid): generated[i] for i, sid in enumerate(sample_ids.tolist())}
    pairs, pair_stats = group_pairs_streaming(
        args.counterfactual_input,
        set(generated_by_id),
        int(args.max_pairs),
        int(args.counterfactual_scan_limit),
    )
    if not pairs:
        return {
            "num_pairs": 0,
            "pair_scan_stats": pair_stats,
            "diagnostic_note": "No counterfactual pairs matched the generated route-memory sample ids.",
        }
    calibrator = build_calibrator([item for pair in pairs for item in pair], int(args.target_len))

    target_effects: List[float] = []
    success: List[bool] = []
    raw_collateral_values: List[float] = []
    independent_values: List[float] = []
    by_field: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for original, counterfactual in pairs:
        cf_meta = counterfactual.get("counterfactual", {})
        field = str(cf_meta.get("field", ""))
        source_token = str(cf_meta.get("from", original.get("labels", {}).get(field, "unknown")))
        target_token = str(cf_meta.get("to", counterfactual.get("labels", {}).get(field, "unknown")))
        source_id = int(cf_meta.get("source_sample_id", original.get("sample_id", -1)))
        base_traj = generated_by_id[source_id]
        # Diagnostic assumption for the current route-memory reranker: semantic
        # counterfactual input is not consumed, so the counterfactual output is unchanged.
        cf_traj = base_traj
        base_geom = geometry_row(base_traj, original, int(args.target_len))
        cf_geom = geometry_row(cf_traj, counterfactual, int(args.target_len))
        _metric, _base_to_target, _observed, _signed, signed_z = calibrator.transition_metric_delta(
            field,
            source_token,
            target_token,
            base_geom,
            cf_geom,
        )
        target_effects.append(float(signed_z))
        success.append(float(signed_z) > 0.0)
        drift_by_field = {}
        for other in DEFAULT_MOBILITY_VOCAB.fields:
            if other == field:
                continue
            token = str(original.get("labels", {}).get(other, "unknown"))
            if token not in calibrator.stats[other]:
                continue
            drift_by_field[other] = calibrator.normalized_metric_delta(other, token, base_geom, cf_geom)
        raw = float(np.mean(list(drift_by_field.values()))) if drift_by_field else 0.0
        independent_fields = independent_collateral_field_names(field, fields=drift_by_field.keys())
        independent = [drift_by_field[name] for name in independent_fields if name in drift_by_field]
        independent_value = float(np.mean(independent)) if independent else 0.0
        raw_collateral_values.append(raw)
        independent_values.append(independent_value)
        by_field[field]["target_effect"].append(float(signed_z))
        by_field[field]["success"].append(1.0 if float(signed_z) > 0.0 else 0.0)
        by_field[field]["raw_collateral"].append(raw)
        by_field[field]["independent_collateral"].append(independent_value)

    by_field_summary = {
        field: {
            "counterfactual_target_effect_z": summarize(vals["target_effect"]),
            "counterfactual_success_rate": float(np.mean(vals["success"])) if vals["success"] else float("nan"),
            "raw_collateral_drift_z": summarize(vals["raw_collateral"]),
            "independent_collateral_drift_z": summarize(vals["independent_collateral"]),
        }
        for field, vals in sorted(by_field.items())
    }
    return {
        "num_pairs": int(len(pairs)),
        "pair_scan_stats": pair_stats,
        "table4_counterfactual_controllability": {
            "counterfactual_target_effect": float(np.mean(target_effects)),
            "counterfactual_success_rate": float(np.mean(success)),
            "raw_collateral_drift": float(np.mean(raw_collateral_values)),
            "independent_collateral_drift": float(np.mean(independent_values)),
        },
        "counterfactual_target_effect_z": summarize(target_effects),
        "raw_collateral_drift_z": summarize(raw_collateral_values),
        "independent_collateral_drift_z": summarize(independent_values),
        "by_field": by_field_summary,
        "diagnostic_note": (
            "No-semantic-change diagnostic for the current route-memory reranker. "
            "Counterfactual output is treated as identical to the base output because "
            "the exported model does not consume counterfactual semantic tokens."
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table3 = evaluate_table3(args)
    table4 = evaluate_table4(args)
    (args.output_dir / "table3_metrics.json").write_text(
        json.dumps(table3, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if table4 is not None:
        (args.output_dir / "table4_metrics.json").write_text(
            json.dumps(table4, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    summary = {
        "table3": table3.get("table3_instruction_following"),
        "table4": table4.get("table4_counterfactual_controllability") if table4 else None,
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
