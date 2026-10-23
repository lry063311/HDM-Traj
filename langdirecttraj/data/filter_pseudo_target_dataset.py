from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None

from langdirecttraj.data.diagnose_pseudo_targets import parse_field_float_map, record_metrics
from langdirecttraj.data.instruction_dataset import collate_instruction_batch, validate_official_record_protocol
from langdirecttraj.counterfactual_policy import independent_collateral_field_names
from langdirecttraj.eval.eval_anchor_decoder import prepare_eval_batch
from langdirecttraj.eval.instruction_geometry import (
    GeometryTokenCalibrator,
    add_pace_metrics_to_rows,
    geometry_values,
    tensor_dict_to_rows,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.data.compact_store import CompactCounterfactualWriter


REPO_ROOT = Path(__file__).resolve().parents[2]


def progress_iter(iterable, desc: str, total: int | None = None):
    if tqdm is not None:
        yield from tqdm(iterable, total=total, desc=desc, unit="record", dynamic_ncols=True)
        return
    started = time.time()
    for idx, item in enumerate(iterable, start=1):
        if idx % 50000 == 0:
            elapsed = max(time.time() - started, 1e-6)
            rate = idx / elapsed
            if total:
                eta = max(total - idx, 0) / max(rate, 1e-6)
                print(
                    f"[filter] {desc}: {idx:,}/{total:,} ({idx / max(total, 1):.1%}), "
                    f"{rate:.1f} record/s, eta={eta / 60.0:.1f} min",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(f"[filter] {desc}: {idx:,}, {rate:.1f} record/s", file=sys.stderr, flush=True)
        yield item


def read_jsonl(path: Path, max_samples: int = 0) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            if line.strip():
                yield json.loads(line)


def summarize_records(records: List[Dict], metrics_rows: List[Dict], dropped_rows: List[Dict]) -> Dict:
    label_counts = defaultdict(Counter)
    cf_counts = Counter()
    fail_reasons = Counter()
    invalid_counterfactual_counts = Counter()
    for record in records:
        for field, value in record.get("labels", {}).items():
            label_counts[field][value] += 1
        cf = record.get("counterfactual", {})
        if cf.get("enabled"):
            cf_counts[f"{cf.get('field')}:{cf.get('to')}"] += 1
    for row in dropped_rows:
        for reason in row.get("fail_reasons", []):
            fail_reasons[reason] += 1
        invalid_reason = row.get("invalid_counterfactual_reason")
        if invalid_reason:
            invalid_counterfactual_counts[invalid_reason] += 1
    return {
        "num_kept": len(records),
        "num_dropped": len(dropped_rows),
        "keep_rate": float(len(records) / max(1, len(records) + len(dropped_rows))),
        "quality_pass_rate_before_filter": float(np.mean([row["quality_pass"] for row in metrics_rows])) if metrics_rows else 0.0,
        "drop_fail_reasons": dict(sorted(fail_reasons.items())),
        "invalid_counterfactual_counts": dict(sorted(invalid_counterfactual_counts.items())),
        "label_counts": {
            field: dict(sorted(counter.items()))
            for field, counter in sorted(label_counts.items())
        },
        "counterfactual_counts": dict(sorted(cf_counts.items())),
    }


def invalid_counterfactual_reason(record: Dict) -> str | None:
    cf = record.get("counterfactual", {})
    if not cf.get("enabled"):
        return None
    field = str(cf.get("field", ""))
    source_value = cf.get("from")
    target_value = cf.get("to")
    label_value = record.get("labels", {}).get(field)
    if source_value == target_value:
        return "counterfactual_noop"
    if target_value != label_value:
        return "counterfactual_label_mismatch"
    return None


def collateral_drift_z(
    calibrator: GeometryTokenCalibrator,
    target_field: str,
    original_labels: Dict[str, str],
    base_geom: Dict[str, float],
    cf_geom: Dict[str, float],
    independent_only: bool = True,
) -> float:
    values: List[float] = []
    if independent_only:
        collateral_fields = set(independent_collateral_field_names(target_field, fields=DEFAULT_MOBILITY_VOCAB.fields))
    else:
        collateral_fields = {field for field in DEFAULT_MOBILITY_VOCAB.fields if field != target_field}
    for field_name in DEFAULT_MOBILITY_VOCAB.fields:
        if field_name == target_field:
            continue
        if field_name not in collateral_fields:
            continue
        token = str(original_labels.get(field_name, "unknown"))
        if token not in calibrator.stats[field_name]:
            continue
        values.append(calibrator.normalized_metric_delta(field_name, token, base_geom, cf_geom))
    return float(np.mean(values)) if values else float("nan")


def grouped_by_sample_id(records: List[Dict]) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for record in records:
        grouped[int(record.get("sample_id", record.get("counterfactual_id", -1)))].append(record)
    return grouped


def tensor_record(record: Dict, target_key: str) -> Dict:
    labels = dict(record.get("labels", {}))
    trajectory = np.asarray(record.get(target_key, record.get("trajectory", [])), dtype=np.float32)
    reference = np.asarray(record.get("reference", []), dtype=np.float32)
    validate_official_record_protocol(record, row_index=int(record.get("sample_id", 0)))
    if reference.ndim != 2 or reference.shape[-1] != 2 or reference.shape[0] == 0:
        raise ValueError("Official OD-chord pseudo-target filtering requires an explicit reference field.")
    if trajectory.ndim != 2 or trajectory.shape[-1] != 2 or trajectory.shape[0] == 0:
        raise ValueError(f"Invalid trajectory shape for target_key={target_key!r}: {trajectory.shape}")
    return {
        "sample_id": int(record.get("sample_id", record.get("counterfactual_id", 0))),
        "dataset": record.get("dataset", ""),
        "traj": torch.from_numpy(trajectory).float(),
        "reference": torch.from_numpy(reference).float(),
        "mobility_tokens": torch.tensor(DEFAULT_MOBILITY_VOCAB.encode(labels), dtype=torch.long),
        "labels": labels,
        "instruction": str(record.get("instruction", "")),
        "metrics": dict(record.get("metrics", {})),
        "counterfactual": dict(record.get("counterfactual", {})),
    }


def geometry_for_record_local(record: Dict, target_key: str, target_len: int) -> Dict[str, float]:
    batch = collate_instruction_batch([tensor_record(record, target_key=target_key)])
    target, reference, route_scales = prepare_eval_batch(batch, target_len)
    geom = geometry_values(target.float(), reference.float(), route_scales[:, 0].float())
    row = tensor_dict_to_rows(geom)[0]
    return add_pace_metrics_to_rows([row], [record.get("trip_attributes", {})])[0]


def add_calibration_quality(
    records: List[Dict],
    metrics_rows: List[Dict],
    calibrator: GeometryTokenCalibrator | None,
    target_len: int,
    min_effect_z: float,
    min_token_match_rate: float,
    max_collateral_z: float,
    field_min_effect_z: Dict[str, float],
    field_min_token_match_rate: Dict[str, float],
    field_max_collateral_z: Dict[str, float],
) -> None:
    if calibrator is None:
        return
    by_sample = grouped_by_sample_id(records)
    original_by_sample: Dict[int, Dict] = {}
    for sample_id, sample_records in by_sample.items():
        original = next(
            (
                record
                for record in sample_records
                if not bool(record.get("counterfactual", {}).get("enabled", False))
            ),
            None,
        )
        if original is not None:
            original_by_sample[sample_id] = original
    geom_cache: Dict[int, Dict[str, float]] = {}

    def original_geometry(sample_id: int, fallback_record: Dict) -> Dict[str, float] | None:
        original = original_by_sample.get(sample_id, fallback_record)
        key = id(original)
        if key not in geom_cache:
            geom_cache[key] = geometry_for_record_local(original, "pseudo_trajectory", target_len)
        return geom_cache[key]

    for record, metrics in progress_iter(
        list(zip(records, metrics_rows)),
        "apply calibration-aware quality",
        len(metrics_rows),
    ):
        cf = record.get("counterfactual", {})
        if not cf.get("enabled"):
            continue
        field = str(cf.get("field", ""))
        source_token = str(cf.get("from", record.get("labels", {}).get(field, "unknown")))
        target_token = str(cf.get("to", record.get("labels", {}).get(field, "unknown")))
        if field not in DEFAULT_MOBILITY_VOCAB.fields:
            continue
        if source_token == target_token:
            continue
        if target_token not in calibrator.stats[field] or source_token not in calibrator.stats[field]:
            continue
        sample_id = int(record.get("sample_id", record.get("counterfactual_id", -1)))
        base_geom = original_geometry(sample_id, record)
        cf_geom = geometry_for_record_local(record, "pseudo_trajectory", target_len)
        base_target_score = calibrator.score_token(field, target_token, base_geom)
        cf_target_score = calibrator.score_token(field, target_token, cf_geom)
        effect_z = float(base_target_score - cf_target_score)
        predicted_token = str(calibrator.predict_field(field, cf_geom))
        token_match = float(predicted_token == target_token)
        independent_collateral = collateral_drift_z(
            calibrator,
            field,
            record.get("labels", {}),
            base_geom,
            cf_geom,
            independent_only=True,
        )
        raw_collateral = collateral_drift_z(
            calibrator,
            field,
            record.get("labels", {}),
            base_geom,
            cf_geom,
            independent_only=False,
        )
        metrics["calibration_target_effect_z"] = effect_z
        metrics["calibration_predicted_token"] = predicted_token
        metrics["calibration_target_match"] = token_match
        metrics["calibration_collateral_drift_z"] = independent_collateral
        metrics["calibration_raw_collateral_drift_z"] = raw_collateral

        required_effect = float(field_min_effect_z.get(field, min_effect_z))
        required_match = float(field_min_token_match_rate.get(field, min_token_match_rate))
        allowed_collateral = float(field_max_collateral_z.get(field, max_collateral_z))
        reasons = list(metrics.get("fail_reasons", []))
        if effect_z < required_effect:
            reasons.append("calibration_target_effect_low")
        if token_match < required_match:
            reasons.append("calibration_target_mismatch")
        if (
            np.isfinite(independent_collateral)
            and allowed_collateral > 0.0
            and independent_collateral > allowed_collateral
        ):
            reasons.append("calibration_collateral_high")
        metrics["fail_reasons"] = reasons
        metrics["quality_pass"] = bool(metrics.get("quality_pass", True)) and not reasons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter or annotate pseudo target samples using geometry quality gates.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "outputs" / "instruction_data" / "porto_counterfactual_instruction_samples.jsonl")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "instruction_data" / "porto_counterfactual_instruction_samples.filtered.jsonl")
    parser.add_argument("--summary-output", type=Path, default=REPO_ROOT / "outputs" / "instruction_data" / "porto_counterfactual_instruction_samples.filtered_summary.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--annotate-only", action="store_true", help="Keep all rows but add pseudo_quality annotations.")
    parser.add_argument("--drop-original-failures", action="store_true", help="Also drop original rows when they fail quality gates.")
    parser.add_argument("--min-length-ratio", type=float, default=0.60)
    parser.add_argument("--max-length-ratio", type=float, default=1.80)
    parser.add_argument("--max-curvature-ratio", type=float, default=4.00)
    parser.add_argument("--max-mean-reference-distance-m", type=float, default=250.0)
    parser.add_argument(
        "--field-max-mean-reference-distance-m",
        nargs="*",
        default=[],
        help="Optional field-specific reference gates, e.g. direction=600 reference_adherence=250.",
    )
    parser.add_argument("--max-endpoint-drift-m", type=float, default=5.0)
    parser.add_argument("--max-step-m", type=float, default=300.0)
    parser.add_argument(
        "--token-calibration-path",
        type=Path,
        default=None,
        help="Optional calibration JSON used to drop pseudo targets that do not actually satisfy their changed token.",
    )
    parser.add_argument("--target-len", type=int, default=128)
    parser.add_argument("--min-calibration-effect-z", type=float, default=-float("inf"))
    parser.add_argument("--min-calibration-token-match-rate", type=float, default=0.0)
    parser.add_argument("--max-calibration-collateral-z", type=float, default=0.0)
    parser.add_argument("--field-min-calibration-effect-z", nargs="*", default=[])
    parser.add_argument("--field-min-calibration-token-match-rate", nargs="*", default=[])
    parser.add_argument("--field-max-calibration-collateral-z", nargs="*", default=[])
    parser.add_argument("--compact-output-dir", type=Path, default=None)
    parser.add_argument("--compact-dataset", default="porto")
    parser.add_argument("--compact-split", default="")
    parser.add_argument("--compact-prefix", default="filtered")
    args = parser.parse_args()
    args.field_reference_limits = parse_field_float_map(args.field_max_mean_reference_distance_m)
    args.field_min_calibration_effect_z_map = parse_field_float_map(args.field_min_calibration_effect_z)
    args.field_min_calibration_token_match_rate_map = parse_field_float_map(
        args.field_min_calibration_token_match_rate
    )
    args.field_max_calibration_collateral_z_map = parse_field_float_map(args.field_max_calibration_collateral_z)
    return args


def main() -> None:
    args = parse_args()
    kept: List[Dict] = []
    dropped: List[Dict] = []
    metrics_rows: List[Dict] = []
    input_records = list(progress_iter(read_jsonl(args.input, args.max_samples), "load pseudo-target records", args.max_samples or None))
    calibrator = GeometryTokenCalibrator.from_json(str(args.token_calibration_path)) if args.token_calibration_path else None

    for record in progress_iter(input_records, "score pseudo-target quality", len(input_records)):
        metrics = record_metrics(record, args)
        invalid_reason = invalid_counterfactual_reason(record)
        if invalid_reason is not None:
            metrics["quality_pass"] = False
            metrics["fail_reasons"] = [*metrics.get("fail_reasons", []), invalid_reason]
            metrics["invalid_counterfactual_reason"] = invalid_reason
        metrics_rows.append(metrics)

    add_calibration_quality(
        input_records,
        metrics_rows,
        calibrator,
        target_len=int(args.target_len),
        min_effect_z=float(args.min_calibration_effect_z),
        min_token_match_rate=float(args.min_calibration_token_match_rate),
        max_collateral_z=float(args.max_calibration_collateral_z),
        field_min_effect_z=args.field_min_calibration_effect_z_map,
        field_min_token_match_rate=args.field_min_calibration_token_match_rate_map,
        field_max_collateral_z=args.field_max_calibration_collateral_z_map,
    )

    for record, metrics in progress_iter(list(zip(input_records, metrics_rows)), "apply pseudo-target filter", len(metrics_rows)):
        annotated = dict(record)
        annotated["pseudo_quality"] = {
            key: metrics.get(key)
            for key in [
                "quality_pass",
                "fail_reasons",
                "length_ratio",
                "curvature_ratio",
                "mean_reference_distance_m",
                "p90_reference_distance_m",
                "endpoint_drift_max_m",
                "max_step_m",
                "jump_rate",
                "calibration_target_effect_z",
                "calibration_predicted_token",
                "calibration_target_match",
                "calibration_collateral_drift_z",
                "calibration_raw_collateral_drift_z",
            ]
        }
        is_original = not bool(record.get("counterfactual", {}).get("enabled", False))
        should_keep = bool(metrics["quality_pass"])
        if is_original and not args.drop_original_failures:
            should_keep = True
        if args.annotate_only or should_keep:
            kept.append(annotated)
        else:
            dropped.append(metrics)

    if not metrics_rows:
        raise ValueError(f"No records loaded from {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    compact_writer = None
    compact_base_arrays = None
    if args.compact_output_dir is not None:
        split = str(args.compact_split or args.output.stem.split(".")[-1])
        instruction_dir = args.compact_output_dir.parent.parent / "instruction_data"
        base_path = instruction_dir / f"{args.compact_dataset}_base_lonlat.npy"
        reference_path = instruction_dir / f"{args.compact_dataset}_reference_lonlat.npy"
        if base_path.exists() and reference_path.exists():
            compact_base_arrays = {
                "base_lonlat": os.path.relpath(base_path, args.compact_output_dir),
                "reference_lonlat": os.path.relpath(reference_path, args.compact_output_dir),
            }
        compact_writer = CompactCounterfactualWriter(
            output_dir=args.compact_output_dir,
            dataset=str(args.compact_dataset),
            split=split,
            target_len=int(args.target_len),
            prefix=str(args.compact_prefix),
        )
    with args.output.open("w", encoding="utf-8") as f:
        for i, record in enumerate(progress_iter(kept, "write filtered pseudo-target records", len(kept))):
            record = dict(record)
            record["filtered_id"] = int(i)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if compact_writer is not None:
                compact_writer.add(record)
    compact_output = compact_writer.close(base_arrays=compact_base_arrays) if compact_writer is not None else None

    summary = summarize_records(kept, metrics_rows, dropped)
    summary.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "annotate_only": bool(args.annotate_only),
            "drop_original_failures": bool(args.drop_original_failures),
            "quality_thresholds": {
                "min_length_ratio": args.min_length_ratio,
                "max_length_ratio": args.max_length_ratio,
                "max_curvature_ratio": args.max_curvature_ratio,
                "max_mean_reference_distance_m": args.max_mean_reference_distance_m,
                "field_max_mean_reference_distance_m": args.field_reference_limits,
                "max_endpoint_drift_m": args.max_endpoint_drift_m,
                "max_step_m": args.max_step_m,
                "token_calibration_path": str(args.token_calibration_path) if args.token_calibration_path else None,
                "target_len": int(args.target_len),
                "min_calibration_effect_z": args.min_calibration_effect_z,
                "min_calibration_token_match_rate": args.min_calibration_token_match_rate,
                "max_calibration_collateral_z": args.max_calibration_collateral_z,
                "field_min_calibration_effect_z": args.field_min_calibration_effect_z_map,
                "field_min_calibration_token_match_rate": args.field_min_calibration_token_match_rate_map,
                "field_max_calibration_collateral_z": args.field_max_calibration_collateral_z_map,
            },
            "compact_output": compact_output,
        }
    )
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
