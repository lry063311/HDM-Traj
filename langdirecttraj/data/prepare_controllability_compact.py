from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from langdirecttraj.data.build_counterfactual_instruction_dataset import (
    make_counterfactual_records,
    summary_from_counters,
    update_summary_counters,
    validate_counterfactual_records,
)
from langdirecttraj.data.build_token_geometry_calibration import build_calibration
from langdirecttraj.data.compact_store import (
    CompactCounterfactualWriter,
    SPLIT_ORDER,
    progress_iter,
    read_jsonl,
)
from langdirecttraj.data.diagnose_counterfactual_transitions import (
    evaluate_pairs,
    group_pairs,
    read_records,
)
from langdirecttraj.data.diagnose_pseudo_targets import record_metrics
from langdirecttraj.data.filter_pseudo_target_dataset import (
    add_calibration_quality,
    invalid_counterfactual_reason,
    summarize_records,
)
from langdirecttraj.data.prepare_controllability_data import (
    SCALAR_EFFECT_ONLY_MIN_EFFECT_OVERRIDES,
    SCALAR_EFFECT_ONLY_TOKEN_MATCH_OVERRIDES,
    add_transition_coverage_check,
    merge_field_overrides,
    summarize_transition_diagnostics,
)
from langdirecttraj.eval.instruction_geometry import GeometryTokenCalibrator
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB


REPO_ROOT = Path(__file__).resolve().parents[2]


def split_input_path(instruction_input: Path, dataset: str, split: str) -> Path:
    path = Path(instruction_input)
    if ".train." in path.name or ".val." in path.name or ".test." in path.name:
        return path.with_name(path.name.replace(".train.", f".{split}.").replace(".val.", f".{split}.").replace(".test.", f".{split}."))
    candidate = path.with_name(f"{path.stem}.{split}{path.suffix}")
    if candidate.exists():
        return candidate
    compact_candidate = path.parent / f"{dataset}_instruction_meta.{split}.jsonl"
    if compact_candidate.exists():
        return compact_candidate
    legacy_candidate = path.parent / f"{dataset}_instruction_samples.{split}.jsonl"
    return legacy_candidate


def _record_stratum(record: Dict, fields: Sequence[str]) -> Tuple[str, ...]:
    labels = record.get("labels", {})
    return tuple(str(labels.get(field, "<missing>")) for field in fields)


def _stable_priority(seed: int, split: str, row_index: int, record: Dict) -> int:
    sample_id = record.get("sample_id", record.get("raw_trip_id", row_index))
    payload = f"{int(seed)}|{split}|{sample_id}|{row_index}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _proportional_stratum_quotas(counts: Dict[Tuple[str, ...], int], cap: int) -> Dict[Tuple[str, ...], int]:
    total = int(sum(counts.values()))
    if cap <= 0 or cap >= total:
        return dict(counts)
    strata = sorted(counts)
    quotas = {key: 0 for key in strata}
    remaining = int(cap)

    # Preserve rare semantic combinations whenever the requested cap permits.
    if cap >= len(strata):
        for key in strata:
            quotas[key] = 1
        remaining -= len(strata)

    residual_counts = {key: counts[key] - quotas[key] for key in strata}
    residual_total = int(sum(residual_counts.values()))
    if remaining <= 0 or residual_total <= 0:
        return quotas

    exact = {key: remaining * residual_counts[key] / residual_total for key in strata}
    for key in strata:
        add = min(residual_counts[key], int(exact[key]))
        quotas[key] += add
    left = cap - sum(quotas.values())
    ranked = sorted(
        strata,
        key=lambda key: (exact[key] - int(exact[key]), residual_counts[key], key),
        reverse=True,
    )
    for key in ranked:
        if left <= 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            left -= 1
    return quotas


def stratified_hash_sample_indices(
    path: Path,
    max_samples: int,
    seed: int,
    split: str,
    fields: Sequence[str],
) -> tuple[List[int], Dict]:
    rows = list(read_jsonl(path))
    total = len(rows)
    cap = total if int(max_samples) <= 0 else min(int(max_samples), total)
    counts = Counter(_record_stratum(record, fields) for record in rows)
    quotas = _proportional_stratum_quotas(dict(counts), cap)
    ranked_by_stratum: Dict[Tuple[str, ...], List[tuple[int, int]]] = defaultdict(list)
    for row_index, record in enumerate(rows):
        key = _record_stratum(record, fields)
        ranked_by_stratum[key].append((_stable_priority(seed, split, row_index, record), row_index))

    selected: List[int] = []
    selected_counts: Counter = Counter()
    for key in sorted(ranked_by_stratum):
        ranked = sorted(ranked_by_stratum[key])
        chosen = [row_index for _, row_index in ranked[: quotas.get(key, 0)]]
        selected.extend(chosen)
        selected_counts[key] = len(chosen)
    selected.sort()

    def serialise(counter: Dict[Tuple[str, ...], int]) -> Dict[str, int]:
        return {"|".join(key): int(value) for key, value in sorted(counter.items()) if int(value) > 0}

    return selected, {
        "mode": "stratified_hash",
        "seed": int(seed),
        "split": split,
        "source_records": int(total),
        "selected_records": int(len(selected)),
        "fields": list(fields),
        "source_strata": serialise(dict(counts)),
        "selected_strata": serialise(dict(selected_counts)),
    }


def split_max_samples(args: argparse.Namespace, split: str) -> int:
    value = getattr(args, f"max_samples_{split}", None)
    return int(args.max_samples if value is None else value)


def base_records(
    path: Path,
    max_samples: int = 0,
    sampling_mode: str = "head",
    sampling_seed: int = 7,
    split: str = "train",
    fields: Sequence[str] = (),
) -> tuple[Iterator[Dict], Dict]:
    # InstructionTrajectoryDataset would expand arrays but imports torch.  For
    # preprocessing, compact metadata can be read through compact_store only
    # after arrays are needed by pseudo-target transforms, so reuse the dataset
    # loader's expansion behavior here through a local import.
    from langdirecttraj.data.instruction_dataset import InstructionTrajectoryDataset

    if sampling_mode == "stratified_hash" and int(max_samples) > 0:
        indices, summary = stratified_hash_sample_indices(path, max_samples, sampling_seed, split, fields)
        ds = InstructionTrajectoryDataset(path, max_samples=None, target_key="trajectory")
        return (ds.get_record(index) for index in indices), summary

    ds = InstructionTrajectoryDataset(path, max_samples=max_samples or None, target_key="trajectory")
    summary = {
        "mode": "all" if int(max_samples) <= 0 else "head",
        "seed": int(sampling_seed),
        "split": split,
        "source_records": None,
        "selected_records": int(len(ds)),
        "fields": list(fields),
    }
    return iter(ds.records), summary


def counterfactual_batches(
    path: Path,
    fields: List[str],
    args: argparse.Namespace,
    split: str,
    sampling_summary: Dict | None = None,
) -> Iterator[List[Dict]]:
    split_seed = int(args.seed) + {"train": 0, "val": 1, "test": 2}.get(split, 0)
    rng = random.Random(split_seed)
    records, summary = base_records(
        path,
        split_max_samples(args, split),
        sampling_mode=str(args.sampling_mode),
        sampling_seed=int(args.sampling_seed),
        split=split,
        fields=fields,
    )
    if sampling_summary is not None:
        sampling_summary.update(summary)
    total = int(summary["selected_records"])
    for base_record in progress_iter(
        records,
        f"generate compact candidates {split}",
        total,
        prefix="compact-stage2",
    ):
        base = dict(base_record)
        base["mobility_tokens"] = DEFAULT_MOBILITY_VOCAB.encode(base["labels"])
        base["pseudo_trajectory"] = base["trajectory"]
        base["counterfactual"] = {
            "enabled": False,
            "source_sample_id": int(base["sample_id"]),
            "field": None,
            "from": None,
            "to": None,
        }
        batch = [base] if args.include_original else []
        batch.extend(
            make_counterfactual_records(
                base,
                fields,
                int(args.variants_per_field),
                rng,
                direction_target_policy=args.direction_target_policy,
                counterfactual_geometry_policy=args.counterfactual_geometry_policy,
            )
        )
        validate_counterfactual_records(batch)
        for record in batch:
            record["split"] = split
        yield batch


def quality_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        min_length_ratio=float(args.min_length_ratio),
        max_length_ratio=float(args.max_length_ratio),
        max_curvature_ratio=float(args.max_curvature_ratio),
        max_mean_reference_distance_m=float(args.max_mean_reference_distance_m),
        field_reference_limits=parse_field_float_map(args.field_max_mean_reference_distance_m),
        max_endpoint_drift_m=float(args.max_endpoint_drift_m),
        max_step_m=float(args.max_step_m),
    )


def parse_field_float_map(items: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items or []:
        key, sep, value = str(item).partition("=")
        if sep:
            out[key.strip()] = float(value)
    return out


def annotate_basic_quality(record: Dict, qargs: SimpleNamespace) -> tuple[Dict, Dict]:
    metrics = record_metrics(record, qargs)
    invalid_reason = invalid_counterfactual_reason(record)
    if invalid_reason is not None:
        metrics["quality_pass"] = False
        metrics["fail_reasons"] = [*metrics.get("fail_reasons", []), invalid_reason]
        metrics["invalid_counterfactual_reason"] = invalid_reason
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
        ]
    }
    return annotated, metrics


def should_keep(record: Dict, metrics: Dict, drop_original_failures: bool = False) -> bool:
    is_original = not bool(record.get("counterfactual", {}).get("enabled", False))
    if is_original and not drop_original_failures:
        return True
    return bool(metrics.get("quality_pass", False))


def build_bootstrap_compact(
    args: argparse.Namespace,
    fields: List[str],
    train_input: Path,
    cache_dir: Path,
) -> Dict:
    writer = CompactCounterfactualWriter(cache_dir, args.dataset, "train", args.target_len, prefix="prefiltered")
    qargs = quality_args(args)
    kept_count = 0
    dropped_count = 0
    sampling_summary: Dict = {}
    for batch in counterfactual_batches(train_input, fields, args, "train", sampling_summary=sampling_summary):
        for record in batch:
            annotated, metrics = annotate_basic_quality(record, qargs)
            if should_keep(annotated, metrics):
                writer.add(annotated)
                kept_count += 1
            else:
                dropped_count += 1
    info = writer.close(base_arrays=base_array_refs(cache_dir, train_input, args.dataset))
    if kept_count <= 0:
        raise ValueError("No train prefiltered records kept for compact calibration.")
    cache_summary = {
        "candidate_source": "generated_and_basic_quality_prefiltered",
        "num_kept": int(kept_count),
        "num_dropped": int(dropped_count),
        "sampling": sampling_summary,
        "compact_output": info,
    }
    (cache_dir / f"{args.dataset}_candidate_cache_summary.json").write_text(
        json.dumps(cache_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"meta_path": Path(info["meta_path"]), "summary": cache_summary}


def cached_candidate_batches(path: Path) -> Iterator[List[Dict]]:
    records, _ = base_records(path, max_samples=0)
    batch: List[Dict] = []
    current_id = None
    for record in records:
        sample_id = int(record.get("sample_id", record.get("counterfactual", {}).get("source_sample_id", -1)))
        if batch and sample_id != current_id:
            yield batch
            batch = []
        batch.append(record)
        current_id = sample_id
    if batch:
        yield batch


def cached_basic_metrics(record: Dict) -> Dict:
    cached = dict(record.get("pseudo_quality", {}))
    if "quality_pass" not in cached:
        raise ValueError("Candidate cache record is missing basic pseudo_quality metrics.")
    cached["fail_reasons"] = list(cached.get("fail_reasons", []))
    return cached


def validate_candidate_cache_summary(
    summary: Dict,
    args: argparse.Namespace,
    fields: Sequence[str],
    split: str = "train",
) -> None:
    sampling = summary.get("sampling", {})
    expected_cap = split_max_samples(args, split)
    source_records = int(sampling.get("source_records", -1))
    expected_selected = source_records if expected_cap <= 0 else min(expected_cap, source_records)
    mismatches = []
    if sampling.get("mode") != str(args.sampling_mode):
        mismatches.append(f"mode={sampling.get('mode')!r}, expected {args.sampling_mode!r}")
    if int(sampling.get("seed", -1)) != int(args.sampling_seed):
        mismatches.append(f"seed={sampling.get('seed')!r}, expected {args.sampling_seed!r}")
    if list(sampling.get("fields", [])) != list(fields):
        mismatches.append("semantic fields differ")
    if source_records < 0 or int(sampling.get("selected_records", -1)) != expected_selected:
        mismatches.append(
            f"selected_records={sampling.get('selected_records')!r}, expected {expected_selected} for cap={expected_cap}"
        )
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "Existing train candidate cache does not match the requested Stage-2 sampling configuration: "
            f"{details}. Rerun without --reuse-calibration to rebuild calibration and cache together."
        )


def base_array_refs(compact_dir: Path, instruction_input: Path, dataset: str) -> Dict[str, str] | None:
    instruction_dir = Path(instruction_input).parent
    base_path = instruction_dir / f"{dataset}_base_lonlat.npy"
    reference_path = instruction_dir / f"{dataset}_reference_lonlat.npy"
    if not base_path.exists() or not reference_path.exists():
        return None
    return {
        "base_lonlat": os.path.relpath(base_path, compact_dir),
        "reference_lonlat": os.path.relpath(reference_path, compact_dir),
    }


def calibration_args(args: argparse.Namespace, input_path: Path, output_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        input=input_path,
        output=output_path,
        target_key="pseudo_trajectory",
        target_len=int(args.target_len),
        max_samples=int(args.calibration_max_samples),
        min_count=8,
        min_reference_scale=0.005,
        min_curvature_scale=0.005,
        min_progress_scale=0.03,
        min_direction_scale=0.15,
        min_distance_scale=0.05,
        active_field_only=True,
        max_token_samples_per_field=0,
        direction_metric_mode=str(args.direction_metric_mode),
        direction_lr_contract_min_step=0.15,
        direction_lr_contract_scale=0.15,
        reference_order_mode="ordered",
        reference_order_min_step=0.006,
        reference_order_scale=0.0,
        progress_order_mode="ordered",
        progress_order_min_step=0.08,
        progress_order_scale=0.08,
        curvature_order_mode=str(args.curvature_order_mode),
        curvature_order_min_step=0.12,
        curvature_order_scale=0.12,
    )


def write_final_split(
    args: argparse.Namespace,
    fields: List[str],
    split: str,
    input_path: Path,
    output_dir: Path,
    calibrator: GeometryTokenCalibrator,
    candidate_cache_path: Path | None = None,
    cached_sampling_summary: Dict | None = None,
) -> Dict:
    writer = CompactCounterfactualWriter(output_dir / "compact", args.dataset, split, args.target_len, prefix="filtered")
    qargs = quality_args(args)
    label_counts = defaultdict(Counter)
    drop_fail_reasons = Counter()
    invalid_counterfactual_counts = Counter()
    total_metrics = 0
    quality_pass_before_filter = 0
    dropped_count = 0
    counts: Dict[str, Counter] = defaultdict(Counter)
    summary_cf_counts = Counter()
    cf_counts = Counter()
    cf_transition_counts = Counter()
    no_op_counts = Counter()
    record_count = 0

    sampling_summary: Dict = dict(cached_sampling_summary or {})
    batches = (
        cached_candidate_batches(candidate_cache_path)
        if candidate_cache_path is not None
        else counterfactual_batches(input_path, fields, args, split, sampling_summary=sampling_summary)
    )
    for batch in batches:
        annotated_batch: List[Dict] = []
        batch_metrics: List[Dict] = []
        for record in batch:
            if candidate_cache_path is not None:
                annotated, metrics = dict(record), cached_basic_metrics(record)
            else:
                annotated, metrics = annotate_basic_quality(record, qargs)
            annotated_batch.append(annotated)
            batch_metrics.append(metrics)
        add_calibration_quality(
            annotated_batch,
            batch_metrics,
            calibrator,
            target_len=int(args.target_len),
            min_effect_z=float(args.min_calibration_effect_z),
            min_token_match_rate=float(args.min_calibration_token_match_rate),
            max_collateral_z=float(args.max_calibration_collateral_z),
            field_min_effect_z=parse_field_float_map(args.field_min_calibration_effect_z),
            field_min_token_match_rate=parse_field_float_map(args.field_min_calibration_token_match_rate),
            field_max_collateral_z=parse_field_float_map(args.field_max_calibration_collateral_z),
        )
        for record, metrics in zip(annotated_batch, batch_metrics):
            total_metrics += 1
            quality_pass_before_filter += int(bool(metrics.get("quality_pass", False)))
            if should_keep(record, metrics):
                record = dict(record)
                record["filtered_id"] = record_count
                record["pseudo_quality"] = {**record.get("pseudo_quality", {}), **{
                    "calibration_target_effect_z": metrics.get("calibration_target_effect_z"),
                    "calibration_predicted_token": metrics.get("calibration_predicted_token"),
                    "calibration_target_match": metrics.get("calibration_target_match"),
                    "calibration_collateral_drift_z": metrics.get("calibration_collateral_drift_z"),
                    "calibration_raw_collateral_drift_z": metrics.get("calibration_raw_collateral_drift_z"),
                    "quality_pass": metrics.get("quality_pass"),
                    "fail_reasons": metrics.get("fail_reasons"),
                }}
                writer.add(record)
                for field, value in record.get("labels", {}).items():
                    label_counts[field][value] += 1
                cf = record.get("counterfactual", {})
                if cf.get("enabled"):
                    summary_cf_counts[f"{cf.get('field')}:{cf.get('to')}"] += 1
                update_summary_counters(record, counts, cf_counts, cf_transition_counts, no_op_counts)
                record_count += 1
            else:
                dropped_count += 1
                for reason in metrics.get("fail_reasons", []):
                    drop_fail_reasons[reason] += 1
                invalid_reason = metrics.get("invalid_counterfactual_reason")
                if invalid_reason:
                    invalid_counterfactual_counts[invalid_reason] += 1

    compact_info = writer.close(base_arrays=base_array_refs(output_dir / "compact", input_path, args.dataset))
    summary = {
        "num_kept": int(record_count),
        "num_dropped": int(dropped_count),
        "keep_rate": float(record_count / max(1, record_count + dropped_count)),
        "quality_pass_rate_before_filter": float(quality_pass_before_filter / max(1, total_metrics)),
        "drop_fail_reasons": dict(sorted(drop_fail_reasons.items())),
        "invalid_counterfactual_counts": dict(sorted(invalid_counterfactual_counts.items())),
        "label_counts": {
            field: dict(sorted(counter.items()))
            for field, counter in sorted(label_counts.items())
        },
        "counterfactual_counts": dict(sorted(summary_cf_counts.items())),
    }
    summary.update(summary_from_counters(record_count, counts, cf_counts, cf_transition_counts, no_op_counts))
    summary.update({
        "split": split,
        "candidate_source": "cached_train_prefiltered" if candidate_cache_path is not None else "generated",
        "base_sampling": sampling_summary,
        "compact_output": compact_info,
    })
    summary_path = output_dir / f"{args.dataset}_counterfactual_instruction_samples.filtered_summary.{split}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_transition_diagnostics(args: argparse.Namespace, compact_train_path: Path, calibration_json: Path, output_dir: Path, fields: List[str]) -> Dict:
    import torch

    records = read_records(compact_train_path, args.transition_max_records)
    pairs, metadata = group_pairs(records, set(fields), args.transition_max_pairs, "balanced_field")
    if not pairs:
        if int(args.transition_min_kept_per_field) > 0:
            raise ValueError("No original/counterfactual pairs found for compact transition diagnostics.")
        output = {
            "input": str(compact_train_path),
            "target_key": "pseudo_trajectory",
            "target_len": int(args.target_len),
            "num_records": int(len(records)),
            "num_pairs": 0,
            "fields": fields,
            "pair_sampling": "balanced_field",
            "transition_counts": {str(k): int(v) for k, v in sorted(metadata["transition_counts"].items())},
            "field_counts": {str(k): int(v) for k, v in sorted(metadata["field_counts"].items())},
            "selected_field_counts": {},
            "no_op_counts": {str(k): int(v) for k, v in sorted(metadata["no_op_counts"].items())},
            "label_mismatch_counts": {str(k): int(v) for k, v in sorted(metadata["label_mismatch_counts"].items())},
            "calibration_path": str(calibration_json),
            "num_scored_pairs": 0,
            "by_field": {},
            "by_transition": {},
        }
        path = output_dir / f"{args.dataset}_counterfactual_transition_diagnostics.json"
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        return output
    calibrator = GeometryTokenCalibrator.from_json(str(calibration_json))
    metrics = evaluate_pairs(
        pairs,
        calibrator=calibrator,
        target_key="pseudo_trajectory",
        target_len=int(args.target_len),
        device=torch.device("cpu"),
    )
    output = {
        "input": str(compact_train_path),
        "target_key": "pseudo_trajectory",
        "target_len": int(args.target_len),
        "num_records": int(len(records)),
        "num_pairs": int(len(pairs)),
        "fields": fields,
        "pair_sampling": "balanced_field",
        "transition_counts": {str(k): int(v) for k, v in sorted(metadata["transition_counts"].items())},
        "field_counts": {str(k): int(v) for k, v in sorted(metadata["field_counts"].items())},
        "selected_field_counts": {str(k): int(v) for k, v in sorted(metadata["selected_field_counts"].items())},
        "no_op_counts": {str(k): int(v) for k, v in sorted(metadata["no_op_counts"].items())},
        "label_mismatch_counts": {str(k): int(v) for k, v in sorted(metadata["label_mismatch_counts"].items())},
        "calibration_path": str(calibration_json),
        **metrics,
    }
    path = output_dir / f"{args.dataset}_counterfactual_transition_diagnostics.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def clean_final_outputs(output_dir: Path, dataset: str) -> None:
    """Remove interrupted final split outputs while preserving calibration."""

    compact_dir = output_dir / "compact"
    if compact_dir.exists():
        shutil.rmtree(compact_dir)
    for path in output_dir.glob(f"{dataset}_counterfactual_instruction_samples.filtered_summary.*.json"):
        path.unlink()
    for name in [
        f"{dataset}_counterfactual_transition_diagnostics.json",
        f"{dataset}_counterfactual_transition_training_summary.json",
        f"{dataset}_counterfactual_transition_training_summary.csv",
        f"{dataset}_controllability_data_manifest.json",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()
    for tmp_dir in output_dir.glob("compact_calibration_*"):
        if tmp_dir.is_dir():
            shutil.rmtree(tmp_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact-only controllability preprocessing for LangDirectTraj.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--instruction-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fields", nargs="+", default=["distance", "direction", "curvature", "progress", "reference_adherence"])
    parser.add_argument("--variants-per-field", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-samples-train", type=int, default=None, help="Train base-record cap; defaults to --max-samples.")
    parser.add_argument("--max-samples-val", type=int, default=None, help="Validation base-record cap; defaults to --max-samples.")
    parser.add_argument("--max-samples-test", type=int, default=None, help="Test base-record cap; defaults to --max-samples.")
    parser.add_argument(
        "--sampling-mode",
        choices=["head", "stratified_hash"],
        default="head",
        help="Bounded split sampling policy. stratified_hash samples across the full split deterministically.",
    )
    parser.add_argument("--sampling-seed", type=int, default=7)
    parser.add_argument("--target-len", type=int, default=128)
    parser.add_argument("--include-original", action="store_true", default=True)
    parser.add_argument("--direction-target-policy", default="transition_aware")
    parser.add_argument("--counterfactual-geometry-policy", default="field_isolated")
    parser.add_argument("--direction-metric-mode", default="signed_lr")
    parser.add_argument("--curvature-order-mode", default="ordered")
    parser.add_argument("--calibration-max-samples", type=int, default=0)
    parser.add_argument("--transition-max-records", type=int, default=0)
    parser.add_argument("--transition-max-pairs", type=int, default=0)
    parser.add_argument("--transition-min-success-rate", type=float, default=0.6)
    parser.add_argument("--transition-min-effect-z", type=float, default=0.05)
    parser.add_argument("--transition-min-token-match-rate", type=float, default=0.0)
    parser.add_argument("--transition-min-pairs", type=int, default=2)
    parser.add_argument("--transition-min-kept-per-field", type=int, default=2)
    parser.add_argument("--min-length-ratio", type=float, default=0.60)
    parser.add_argument("--max-length-ratio", type=float, default=1.80)
    parser.add_argument("--max-curvature-ratio", type=float, default=4.00)
    parser.add_argument("--max-mean-reference-distance-m", type=float, default=250.0)
    parser.add_argument("--field-max-mean-reference-distance-m", nargs="+", default=["direction=600"])
    parser.add_argument("--max-endpoint-drift-m", type=float, default=5.0)
    parser.add_argument("--max-step-m", type=float, default=300.0)
    parser.add_argument("--min-calibration-effect-z", type=float, default=0.0)
    parser.add_argument("--min-calibration-token-match-rate", type=float, default=1.0)
    parser.add_argument("--max-calibration-collateral-z", type=float, default=0.8)
    parser.add_argument("--field-min-calibration-effect-z", nargs="*", default=[])
    parser.add_argument("--field-min-calibration-token-match-rate", nargs="*", default=[])
    parser.add_argument("--field-max-calibration-collateral-z", nargs="*", default=[])
    parser.add_argument(
        "--reuse-calibration",
        action="store_true",
        help=(
            "Resume after an interrupted final compact split write by reusing "
            "<dataset>_token_geometry_calibration.json and rerunning only final splits plus diagnostics."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp-calibration", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.field_min_calibration_effect_z = merge_field_overrides(
        args.field_min_calibration_effect_z,
        SCALAR_EFFECT_ONLY_MIN_EFFECT_OVERRIDES,
    )
    args.field_min_calibration_token_match_rate = merge_field_overrides(
        args.field_min_calibration_token_match_rate,
        SCALAR_EFFECT_ONLY_TOKEN_MATCH_OVERRIDES,
    )
    output_dir = Path(args.output_dir)
    calibration_json = output_dir / f"{args.dataset}_token_geometry_calibration.json"
    if output_dir.exists() and args.overwrite and not args.reuse_calibration:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_calibration:
        if not calibration_json.exists():
            raise FileNotFoundError(f"Cannot reuse missing calibration file: {calibration_json}")
        clean_final_outputs(output_dir, args.dataset)
    fields = [field for field in args.fields if field in DEFAULT_MOBILITY_VOCAB.fields]
    split_inputs = {split: split_input_path(args.instruction_input, args.dataset, split) for split in SPLIT_ORDER}
    missing = [str(path) for path in split_inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split instruction inputs: {missing}")

    candidate_cache_dir = output_dir / "candidate_cache"
    try:
        cached_train_meta = candidate_cache_dir / f"{args.dataset}_counterfactual_instruction_samples.prefiltered.train.compact.jsonl"
        cached_train_summary_path = candidate_cache_dir / f"{args.dataset}_candidate_cache_summary.json"
        bootstrap_info = None
        if args.reuse_calibration:
            print(f"\n[compact resume] reuse token-geometry calibration: {calibration_json}", flush=True)
        else:
            candidate_cache_dir.mkdir(parents=True, exist_ok=True)
            print("\n[compact 1/4] generate and cache train candidates for calibration", flush=True)
            bootstrap_info = build_bootstrap_compact(args, fields, split_inputs["train"], candidate_cache_dir)
            bootstrap_meta = bootstrap_info["meta_path"]

            print("\n[compact 2/4] build token-geometry calibration", flush=True)
            calibration = build_calibration(calibration_args(args, bootstrap_meta, calibration_json))
            calibration_json.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
        calibrator = GeometryTokenCalibrator.from_json(str(calibration_json))

        print("\n[compact 3/4] write final compact filtered splits", flush=True)
        split_summaries = {}
        for split in SPLIT_ORDER:
            use_train_cache = split == "train" and cached_train_meta.exists()
            cached_sampling = None
            if use_train_cache and cached_train_summary_path.exists():
                cache_summary = json.loads(cached_train_summary_path.read_text(encoding="utf-8"))
                validate_candidate_cache_summary(cache_summary, args, fields, split="train")
                cached_sampling = cache_summary.get("sampling", {})
            split_summaries[split] = write_final_split(
                args,
                fields,
                split,
                split_inputs[split],
                output_dir,
                calibrator,
                candidate_cache_path=cached_train_meta if use_train_cache else None,
                cached_sampling_summary=cached_sampling,
            )

        print("\n[compact 4/4] compact transition diagnostics", flush=True)
        train_compact = output_dir / "compact" / f"{args.dataset}_counterfactual_instruction_samples.filtered.train.compact.jsonl"
        transition_payload = run_transition_diagnostics(args, train_compact, calibration_json, output_dir, fields)
        transition_json = output_dir / f"{args.dataset}_counterfactual_transition_diagnostics.json"
        transition_summary_json = output_dir / f"{args.dataset}_counterfactual_transition_training_summary.json"
        transition_summary_csv = output_dir / f"{args.dataset}_counterfactual_transition_training_summary.csv"
        transition_summary = summarize_transition_diagnostics(
            transition_json,
            transition_summary_json,
            transition_summary_csv,
            min_success_rate=args.transition_min_success_rate,
            min_effect_z=args.transition_min_effect_z,
            min_token_match_rate=args.transition_min_token_match_rate,
            min_pairs=args.transition_min_pairs,
        )
        coverage_check = add_transition_coverage_check(
            transition_summary,
            fields=fields,
            min_kept_per_field=args.transition_min_kept_per_field,
        )
        transition_summary_json.write_text(json.dumps(transition_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.transition_min_kept_per_field > 0 and not coverage_check["passed"]:
            failed = ", ".join(f"{field}={count}" for field, count in sorted(coverage_check["failed_fields"].items()))
            raise RuntimeError(f"Compact transition coverage is too weak: {failed}")

        manifest = {
            "dataset": args.dataset,
            "storage_format": "compact_only",
            "instruction_inputs": {split: str(path) for split, path in split_inputs.items()},
            "output_dir": str(output_dir),
            "fields": fields,
            "variants_per_field": int(args.variants_per_field),
            "target_len": int(args.target_len),
            "base_sampling": {
                "mode": str(args.sampling_mode),
                "seed": int(args.sampling_seed),
                "max_samples": {split: split_max_samples(args, split) for split in SPLIT_ORDER},
            },
            "train_candidate_cache": str(cached_train_meta) if cached_train_meta.exists() else None,
            "calibration_json": str(calibration_json),
            "transition_diagnostics_json": str(transition_json),
            "transition_training_summary_json": str(transition_summary_json),
            "split": {
                "enabled": True,
                "split_unit": "base_sample_id",
                "calibration_source_split": "train",
                "transition_diagnostics_source_split": "train",
            },
            "split_compact_jsonl": {
                split: str(output_dir / "compact" / f"{args.dataset}_counterfactual_instruction_samples.filtered.{split}.compact.jsonl")
                for split in SPLIT_ORDER
            },
            "split_summaries": split_summaries,
            "transition_num_pairs": transition_payload.get("num_pairs"),
        }
        manifest_path = output_dir / f"{args.dataset}_controllability_data_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({**manifest, "manifest_json": str(manifest_path)}, ensure_ascii=False, indent=2))
    finally:
        gc.collect()


if __name__ == "__main__":
    main()
