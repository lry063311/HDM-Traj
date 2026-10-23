from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None

from langdirecttraj.data.compact_store import (
    expand_compact_record,
    is_compact_meta_path,
    load_compact_manifest,
    open_compact_arrays,
)
from langdirecttraj.data.instruction_dataset import (
    collate_instruction_batch,
    validate_official_record_protocol,
)
from langdirecttraj.eval.eval_anchor_decoder import prepare_eval_batch
from langdirecttraj.eval.instruction_geometry import (
    GeometryTokenCalibrator,
    add_pace_metrics_to_rows,
    geometry_values,
    tensor_dict_to_rows,
)
from langdirecttraj.counterfactual_policy import (
    STRUCTURAL_COLLATERAL_EXCLUSIONS,
    independent_collateral_field_names,
    serializable_collateral_exclusions,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB


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
                    f"[transition] {desc}: {idx:,}/{total:,} ({idx / max(total, 1):.1%}), "
                    f"{rate:.1f} record/s, eta={eta / 60.0:.1f} min",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(f"[transition] {desc}: {idx:,}, {rate:.1f} record/s", file=sys.stderr, flush=True)
        yield item


def read_records(path: Path, max_records: int = 0) -> List[Dict]:
    records: List[Dict] = []
    manifest = load_compact_manifest(path) if is_compact_meta_path(path) else None
    arrays = open_compact_arrays(manifest) if manifest else {}
    with path.open("r", encoding="utf-8") as f:
        for line in progress_iter(f, "load transition records", max_records or None):
            if not line.strip():
                continue
            record = json.loads(line)
            if manifest is not None:
                record = expand_compact_record(record, arrays)
            records.append(record)
            if max_records > 0 and len(records) >= max_records:
                break
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def iter_records(path: Path, max_records: int = 0) -> Iterator[Dict]:
    manifest = load_compact_manifest(path) if is_compact_meta_path(path) else None
    arrays = open_compact_arrays(manifest) if manifest else {}
    emitted = 0
    with path.open("r", encoding="utf-8") as f:
        for line in progress_iter(f, "load transition records", max_records or None):
            if not line.strip():
                continue
            record = json.loads(line)
            if manifest is not None:
                record = expand_compact_record(record, arrays)
            emitted += 1
            yield record
            if max_records > 0 and emitted >= max_records:
                break


class StreamingPairIterator:
    """Yield original/counterfactual pairs without retaining expanded trajectories."""

    def __init__(self, path: Path, fields: set[str], max_records: int = 0):
        self.path = Path(path)
        self.fields = fields
        self.max_records = int(max_records)
        self.num_records = 0
        self.num_pairs = 0
        self.transition_counts = Counter()
        self.field_counts = Counter()
        self.selected_field_counts = Counter()
        self.no_op_counts = Counter()
        self.label_mismatch_counts = Counter()
        self._iterated = False

    def _flush_group(self, original: Dict | None, counterfactuals: List[Dict]) -> Iterator[Tuple[Dict, Dict]]:
        if original is None:
            return
        for cf_record in counterfactuals:
            field = str(cf_record.get("counterfactual", {}).get("field", "unknown"))
            self.selected_field_counts[field] += 1
            self.num_pairs += 1
            yield original, cf_record

    def __iter__(self) -> Iterator[Tuple[Dict, Dict]]:
        if self._iterated:
            raise RuntimeError("StreamingPairIterator can only be iterated once.")
        self._iterated = True
        current_source_id = None
        current_original: Dict | None = None
        current_counterfactuals: List[Dict] = []

        for record in iter_records(self.path, self.max_records):
            self.num_records += 1
            cf = record.get("counterfactual", {})
            source_id = int(cf.get("source_sample_id", record.get("sample_id", -1)))
            if current_source_id is not None and source_id != current_source_id:
                yield from self._flush_group(current_original, current_counterfactuals)
                current_original = None
                current_counterfactuals = []
            current_source_id = source_id

            if cf.get("enabled"):
                field = str(cf.get("field", "unknown"))
                source = str(cf.get("from", "unknown"))
                target = str(cf.get("to", "unknown"))
                label_value = str(record.get("labels", {}).get(field, "unknown"))
                transition_key = f"{field}:{source}->{target}"
                self.transition_counts[transition_key] += 1
                self.field_counts[field] += 1
                if source == target:
                    self.no_op_counts[transition_key] += 1
                if target != label_value:
                    self.label_mismatch_counts[f"{transition_key}/label={label_value}"] += 1
                if not self.fields or field in self.fields:
                    current_counterfactuals.append(record)
            else:
                current_original = record

        yield from self._flush_group(current_original, current_counterfactuals)

    def metadata(self) -> Dict:
        return {
            "transition_counts": self.transition_counts,
            "field_counts": self.field_counts,
            "selected_field_counts": self.selected_field_counts,
            "no_op_counts": self.no_op_counts,
            "label_mismatch_counts": self.label_mismatch_counts,
        }


def tensor_record(record: Dict, target_key: str) -> Dict:
    labels = dict(record.get("labels", {}))
    traj = np.asarray(record.get(target_key, record.get("trajectory", [])), dtype=np.float32)
    validate_official_record_protocol(record, row_index=int(record.get("sample_id", 0)))
    if "reference" not in record:
        raise ValueError("Official OD-chord transition diagnosis requires an explicit reference field.")
    reference = np.asarray(record["reference"], dtype=np.float32)
    return {
        "sample_id": int(record.get("sample_id", record.get("counterfactual_id", 0))),
        "dataset": record.get("dataset", ""),
        "traj": torch.from_numpy(traj).float(),
        "reference": torch.from_numpy(reference).float(),
        "mobility_tokens": torch.tensor(DEFAULT_MOBILITY_VOCAB.encode(labels), dtype=torch.long),
        "labels": labels,
        "instruction": str(record.get("instruction", "")),
        "metrics": dict(record.get("metrics", {})),
    }


def group_pairs(
    records: List[Dict],
    fields: set[str],
    max_pairs: int,
    pair_sampling: str = "balanced_field",
) -> Tuple[List[Tuple[Dict, Dict]], Dict]:
    originals: Dict[int, Dict] = {}
    counterfactuals: Dict[int, List[Dict]] = defaultdict(list)
    no_op_counts = Counter()
    label_mismatch_counts = Counter()
    transition_counts = Counter()
    field_counts = Counter()

    for record in records:
        cf = record.get("counterfactual", {})
        source_id = int(cf.get("source_sample_id", record.get("sample_id", -1)))
        if cf.get("enabled"):
            field = str(cf.get("field", "unknown"))
            source = str(cf.get("from", "unknown"))
            target = str(cf.get("to", "unknown"))
            label_value = str(record.get("labels", {}).get(field, "unknown"))
            transition_key = f"{field}:{source}->{target}"
            transition_counts[transition_key] += 1
            field_counts[field] += 1
            if source == target:
                no_op_counts[transition_key] += 1
            if target != label_value:
                label_mismatch_counts[f"{transition_key}/label={label_value}"] += 1
            if not fields or field in fields:
                counterfactuals[source_id].append(record)
        else:
            originals[int(record.get("sample_id", source_id))] = record

    pairs: List[Tuple[Dict, Dict]] = []
    for source_id, cf_records in counterfactuals.items():
        original = originals.get(source_id)
        if original is None:
            continue
        for cf_record in cf_records:
            pairs.append((original, cf_record))
    if max_pairs > 0 and len(pairs) > max_pairs:
        if pair_sampling == "sequential":
            pairs = pairs[:max_pairs]
        elif pair_sampling == "balanced_field":
            field_order = [field for field in DEFAULT_MOBILITY_VOCAB.fields if not fields or field in fields]
            buckets: Dict[str, List[Tuple[Dict, Dict]]] = {field: [] for field in field_order}
            for pair in pairs:
                field = str(pair[1].get("counterfactual", {}).get("field", ""))
                if field in buckets:
                    buckets[field].append(pair)
            selected: List[Tuple[Dict, Dict]] = []
            cursor = {field: 0 for field in field_order}
            while len(selected) < max_pairs:
                added = False
                for field in field_order:
                    idx = cursor[field]
                    if idx >= len(buckets[field]):
                        continue
                    selected.append(buckets[field][idx])
                    cursor[field] = idx + 1
                    added = True
                    if len(selected) >= max_pairs:
                        break
                if not added:
                    break
            pairs = selected
        else:
            raise ValueError(f"Unsupported pair_sampling={pair_sampling}")
    selected_field_counts = Counter(
        str(pair[1].get("counterfactual", {}).get("field", "unknown")) for pair in pairs
    )
    return pairs, {
        "transition_counts": transition_counts,
        "field_counts": field_counts,
        "selected_field_counts": selected_field_counts,
        "no_op_counts": no_op_counts,
        "label_mismatch_counts": label_mismatch_counts,
    }


def geometry_for_record(record: Dict, target_key: str, target_len: int, device: torch.device) -> Dict[str, float]:
    batch = collate_instruction_batch([tensor_record(record, target_key)])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    target, reference, route_scales = prepare_eval_batch(batch, target_len)
    geom = geometry_values(target.float(), reference.float(), route_scales[:, 0].float())
    return add_pace_metrics_to_rows(
        tensor_dict_to_rows(geom),
        [record.get("trip_attributes", {})],
    )[0]


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def collateral_drift_z(
    calibrator: GeometryTokenCalibrator,
    target_field: str,
    original_labels: Dict[str, str],
    base_geom: Dict[str, float],
    cf_geom: Dict[str, float],
) -> float:
    values: List[float] = []
    for field_name in DEFAULT_MOBILITY_VOCAB.fields:
        if field_name == target_field:
            continue
        token = str(original_labels.get(field_name, "unknown"))
        if token not in calibrator.stats[field_name]:
            continue
        values.append(calibrator.normalized_metric_delta(field_name, token, base_geom, cf_geom))
    return float(np.mean(values)) if values else float("nan")


def collateral_drift_z_by_field(
    calibrator: GeometryTokenCalibrator,
    target_field: str,
    original_labels: Dict[str, str],
    base_geom: Dict[str, float],
    cf_geom: Dict[str, float],
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for field_name in DEFAULT_MOBILITY_VOCAB.fields:
        if field_name == target_field:
            continue
        token = str(original_labels.get(field_name, "unknown"))
        if token not in calibrator.stats[field_name]:
            continue
        values[field_name] = calibrator.normalized_metric_delta(field_name, token, base_geom, cf_geom)
    return values


def independent_collateral_drift_z(
    calibrator: GeometryTokenCalibrator,
    target_field: str,
    original_labels: Dict[str, str],
    base_geom: Dict[str, float],
    cf_geom: Dict[str, float],
) -> float:
    values: List[float] = []
    for field_name in independent_collateral_field_names(target_field):
        token = str(original_labels.get(field_name, "unknown"))
        if token not in calibrator.stats[field_name]:
            continue
        values.append(calibrator.normalized_metric_delta(field_name, token, base_geom, cf_geom))
    return float(np.mean(values)) if values else float("nan")


def evaluate_pairs(
    pairs: Iterable[Tuple[Dict, Dict]],
    calibrator: GeometryTokenCalibrator,
    target_key: str,
    target_len: int,
    device: torch.device,
    total_pairs: int | None = None,
) -> Dict:
    def new_bucket() -> Dict:
        return {
            "effect": [],
            "success": [],
            "signed_delta": [],
            "signed_delta_z": [],
            "center_delta": [],
            "collateral": [],
            "independent_collateral": [],
            "collateral_by_non_target": defaultdict(list),
            "independent_collateral_by_non_target": defaultdict(list),
            "predicted_target_match": [],
            "predicted_source_match": [],
            "predicted_tokens": Counter(),
        }

    by_field = defaultdict(new_bucket)
    by_transition = defaultdict(new_bucket)
    target_effects: List[float] = []
    success_values: List[float] = []
    collateral_values: List[float] = []
    independent_collateral_values: List[float] = []
    target_match_values: List[float] = []
    source_match_values: List[float] = []
    predicted_token_counts: Counter[str] = Counter()

    for original, counterfactual in progress_iter(pairs, "score transition pairs", total_pairs):
        cf = counterfactual.get("counterfactual", {})
        field = str(cf.get("field", "unknown"))
        source_token = str(cf.get("from", original.get("labels", {}).get(field, "unknown")))
        target_token = str(cf.get("to", counterfactual.get("labels", {}).get(field, "unknown")))
        if field not in DEFAULT_MOBILITY_VOCAB.fields:
            continue
        if source_token == target_token:
            continue
        if target_token not in calibrator.stats[field] or source_token not in calibrator.stats[field]:
            continue

        base_geom = geometry_for_record(original, target_key, target_len, device)
        cf_geom = geometry_for_record(counterfactual, target_key, target_len, device)
        base_target_score = calibrator.score_token(field, target_token, base_geom)
        cf_target_score = calibrator.score_token(field, target_token, cf_geom)
        target_effect = float(base_target_score - cf_target_score)
        success = float(target_effect > 0.0)
        collateral = collateral_drift_z(calibrator, field, original.get("labels", {}), base_geom, cf_geom)
        independent_collateral = independent_collateral_drift_z(
            calibrator,
            field,
            original.get("labels", {}),
            base_geom,
            cf_geom,
        )
        collateral_by_field = collateral_drift_z_by_field(
            calibrator,
            field,
            original.get("labels", {}),
            base_geom,
            cf_geom,
        )
        independent_collateral_by_field = {
            name: value
            for name, value in collateral_by_field.items()
            if name not in STRUCTURAL_COLLATERAL_EXCLUSIONS.get(field, set())
        }
        predicted_token = str(calibrator.predict_field(field, cf_geom))
        predicted_target_match = float(predicted_token == target_token)
        predicted_source_match = float(predicted_token == source_token)
        (
            metric_name,
            expected_center_delta,
            observed_delta,
            signed_delta,
            signed_delta_z,
        ) = calibrator.transition_metric_delta(field, source_token, target_token, base_geom, cf_geom)
        key = f"{field}:{source_token}->{target_token}"

        for bucket in (by_field[field], by_transition[key]):
            bucket["effect"].append(target_effect)
            bucket["success"].append(success)
            bucket["signed_delta"].append(signed_delta)
            bucket["signed_delta_z"].append(signed_delta_z)
            bucket["center_delta"].append(float(expected_center_delta))
            bucket["collateral"].append(collateral)
            bucket["independent_collateral"].append(independent_collateral)
            for collateral_field, collateral_value in collateral_by_field.items():
                bucket["collateral_by_non_target"][collateral_field].append(collateral_value)
            for collateral_field, collateral_value in independent_collateral_by_field.items():
                bucket["independent_collateral_by_non_target"][collateral_field].append(collateral_value)
            bucket["predicted_target_match"].append(predicted_target_match)
            bucket["predicted_source_match"].append(predicted_source_match)
            bucket["predicted_tokens"][predicted_token] += 1
        target_effects.append(target_effect)
        success_values.append(success)
        collateral_values.append(collateral)
        independent_collateral_values.append(independent_collateral)
        target_match_values.append(predicted_target_match)
        source_match_values.append(predicted_source_match)
        predicted_token_counts[f"{field}:{predicted_token}"] += 1

    def summarize_bucket(bucket: Dict[str, List[float]]) -> Dict:
        return {
            "num_pairs": int(len(bucket["success"])),
            "pseudo_target_effect_z": summarize(bucket["effect"]),
            "pseudo_target_success_rate": float(np.mean(bucket["success"])) if bucket["success"] else float("nan"),
            "predicted_target_match_rate": (
                float(np.mean(bucket["predicted_target_match"])) if bucket["predicted_target_match"] else float("nan")
            ),
            "predicted_source_match_rate": (
                float(np.mean(bucket["predicted_source_match"])) if bucket["predicted_source_match"] else float("nan")
            ),
            "predicted_token_counts": {
                str(token): int(count) for token, count in sorted(bucket["predicted_tokens"].items())
            },
            "signed_target_metric_delta": summarize(bucket["signed_delta"]),
            "signed_target_metric_delta_z": summarize(bucket["signed_delta_z"]),
            "expected_center_delta": summarize(bucket["center_delta"]),
            "collateral_drift_z": summarize(bucket["collateral"]),
            "independent_collateral_drift_z": summarize(bucket["independent_collateral"]),
            "collateral_by_non_target": {
                field_name: summarize(field_values)
                for field_name, field_values in sorted(bucket["collateral_by_non_target"].items())
            },
            "independent_collateral_by_non_target": {
                field_name: summarize(field_values)
                for field_name, field_values in sorted(bucket["independent_collateral_by_non_target"].items())
            },
        }

    return {
        "num_scored_pairs": int(len(success_values)),
        "pseudo_target_effect_z": summarize(target_effects),
        "pseudo_target_success_rate": float(np.mean(success_values)) if success_values else float("nan"),
        "collateral_drift_z": summarize(collateral_values),
        "independent_collateral_drift_z": summarize(independent_collateral_values),
        "independent_collateral_exclusions": serializable_collateral_exclusions(),
        "predicted_target_match_rate": (
            float(np.mean(target_match_values)) if target_match_values else float("nan")
        ),
        "predicted_source_match_rate": (
            float(np.mean(source_match_values)) if source_match_values else float("nan")
        ),
        "predicted_token_counts": {
            str(token): int(count) for token, count in sorted(predicted_token_counts.items())
        },
        "by_field": {key: summarize_bucket(value) for key, value in sorted(by_field.items())},
        "by_transition": {key: summarize_bucket(value) for key, value in sorted(by_transition.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose counterfactual transition metadata and pseudo-target effect.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "local_porto_od_chord_256_verify"
        / "controllability_data"
        / "porto_counterfactual_instruction_samples.filtered.jsonl",
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "counterfactual_transition_diagnostics.json")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--pair-sampling",
        choices=["sequential", "balanced_field"],
        default="balanced_field",
        help="How to choose scored pairs when --max-pairs truncates the diagnostic set.",
    )
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--target-len", type=int, default=128)
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["distance", "direction", "curvature", "progress", "reference_adherence"],
    )
    parser.add_argument(
        "--token-calibration-path",
        type=Path,
        default=REPO_ROOT / "outputs" / "calibration" / "porto_token_geometry_calibration_direction_v2_auto.json",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibrator = GeometryTokenCalibrator.from_json(str(args.token_calibration_path))
    if int(args.max_pairs) > 0:
        records = read_records(args.input, args.max_records)
        pairs, metadata = group_pairs(records, set(args.fields), args.max_pairs, args.pair_sampling)
        if not pairs:
            raise ValueError("No original/counterfactual pairs found.")
        metrics = evaluate_pairs(
            pairs,
            calibrator=calibrator,
            target_key=args.target_key,
            target_len=args.target_len,
            device=torch.device(args.device),
            total_pairs=len(pairs),
        )
        num_records = len(records)
        num_pairs = len(pairs)
    else:
        pair_stream = StreamingPairIterator(args.input, set(args.fields), args.max_records)
        metrics = evaluate_pairs(
            pair_stream,
            calibrator=calibrator,
            target_key=args.target_key,
            target_len=args.target_len,
            device=torch.device(args.device),
            total_pairs=None,
        )
        if pair_stream.num_pairs <= 0:
            raise ValueError("No original/counterfactual pairs found.")
        metadata = pair_stream.metadata()
        num_records = pair_stream.num_records
        num_pairs = pair_stream.num_pairs
    output = {
        "input": str(args.input),
        "target_key": args.target_key,
        "target_len": int(args.target_len),
        "num_records": int(num_records),
        "num_pairs": int(num_pairs),
        "fields": args.fields,
        "pair_sampling": args.pair_sampling,
        "transition_counts": {str(k): int(v) for k, v in sorted(metadata["transition_counts"].items())},
        "field_counts": {str(k): int(v) for k, v in sorted(metadata["field_counts"].items())},
        "selected_field_counts": {
            str(k): int(v) for k, v in sorted(metadata["selected_field_counts"].items())
        },
        "no_op_counts": {str(k): int(v) for k, v in sorted(metadata["no_op_counts"].items())},
        "label_mismatch_counts": {
            str(k): int(v) for k, v in sorted(metadata["label_mismatch_counts"].items())
        },
        "calibration_path": str(args.token_calibration_path),
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**output, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
