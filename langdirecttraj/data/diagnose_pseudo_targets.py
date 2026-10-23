from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from langdirecttraj.data.instruction_dataset import validate_official_record_protocol


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path, max_samples: int = 0) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            if line.strip():
                yield json.loads(line)


def to_local_m(points: np.ndarray, origin: np.ndarray | None = None, lat_ref: float | None = None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if origin is None:
        origin = points[0]
    if lat_ref is None:
        lat_ref = float(np.nanmean(points[:, 1]))
    scale_x = 111_320.0 * math.cos(math.radians(lat_ref))
    scale_y = 110_540.0
    local = np.empty_like(points, dtype=np.float64)
    local[:, 0] = (points[:, 0] - origin[0]) * scale_x
    local[:, 1] = (points[:, 1] - origin[1]) * scale_y
    return local


def resample_polyline(points: np.ndarray, target_len: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == target_len:
        return points
    if len(points) == 0:
        return np.zeros((target_len, 2), dtype=np.float64)
    if len(points) == 1:
        return np.repeat(points[:1], target_len, axis=0)
    steps = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(np.clip(steps, 1e-8, None))])
    if cumulative[-1] <= 1e-8:
        return np.repeat(points[:1], target_len, axis=0)
    targets = np.linspace(0.0, cumulative[-1], target_len)
    return np.stack(
        [
            np.interp(targets, cumulative, points[:, 0]),
            np.interp(targets, cumulative, points[:, 1]),
        ],
        axis=1,
    )


def path_length_m(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    steps = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return float(np.sum(steps[np.isfinite(steps)]))


def endpoint_drift_m(candidate: np.ndarray, source: np.ndarray) -> Dict[str, float]:
    if len(candidate) == 0 or len(source) == 0:
        return {"start": float("nan"), "end": float("nan"), "max": float("nan")}
    start = float(np.linalg.norm(candidate[0] - source[0]))
    end = float(np.linalg.norm(candidate[-1] - source[-1]))
    return {"start": start, "end": end, "max": max(start, end)}


def heading_angles(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.zeros(0, dtype=np.float64)
    diffs = points[1:] - points[:-1]
    valid = np.linalg.norm(diffs, axis=1) > 1e-6
    diffs = diffs[valid]
    if len(diffs) < 2:
        return np.zeros(0, dtype=np.float64)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    return np.diff(np.unwrap(angles))


def curvature_stats(points: np.ndarray) -> Dict[str, float]:
    delta = heading_angles(points)
    if len(delta) == 0:
        return {"mean_abs": 0.0, "total_abs": 0.0, "max_abs": 0.0}
    abs_delta = np.abs(delta)
    return {
        "mean_abs": float(np.mean(abs_delta)),
        "total_abs": float(np.sum(abs_delta)),
        "max_abs": float(np.max(abs_delta)),
    }


def reference_distance_m(candidate: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    if len(candidate) == 0 or len(reference) == 0:
        return {"mean": float("nan"), "p90": float("nan"), "max": float("nan")}
    ref = resample_polyline(reference, len(candidate))
    dist = np.linalg.norm(candidate - ref, axis=1)
    return {
        "mean": float(np.mean(dist)),
        "p90": float(np.percentile(dist, 90)),
        "max": float(np.max(dist)),
    }


def jump_stats(points: np.ndarray, jump_threshold_m: float) -> Dict[str, float]:
    if len(points) < 2:
        return {"max_step": 0.0, "jump_rate": 0.0, "jump_count": 0.0}
    steps = np.linalg.norm(points[1:] - points[:-1], axis=1)
    jumps = steps > float(jump_threshold_m)
    return {
        "max_step": float(np.max(steps)),
        "jump_rate": float(np.mean(jumps)),
        "jump_count": float(np.sum(jumps)),
    }


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def parse_field_float_map(items: Iterable[str]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for item in items:
        for part in str(item).replace(",", " ").split():
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Expected FIELD=VALUE assignment, got: {part}")
            key, value = part.split("=", 1)
            output[key.strip()] = float(value)
    return output


def check_quality(metrics: Dict[str, float], args: argparse.Namespace) -> List[str]:
    reasons = []
    field = str(metrics.get("field", "original"))
    field_reference_limits = getattr(args, "field_reference_limits", {}) or {}
    max_reference_distance = float(field_reference_limits.get(field, args.max_mean_reference_distance_m))
    if metrics["length_ratio"] < args.min_length_ratio:
        reasons.append("length_too_short")
    if metrics["length_ratio"] > args.max_length_ratio:
        reasons.append("length_too_long")
    if metrics["curvature_ratio"] > args.max_curvature_ratio:
        reasons.append("curvature_too_high")
    if metrics["mean_reference_distance_m"] > max_reference_distance:
        reasons.append("reference_too_far")
    if metrics["endpoint_drift_max_m"] > args.max_endpoint_drift_m:
        reasons.append("endpoint_drift")
    if metrics["max_step_m"] > args.max_step_m:
        reasons.append("jump_step")
    return reasons


def record_metrics(record: Dict, args: argparse.Namespace) -> Dict:
    validate_official_record_protocol(record, row_index=int(record.get("sample_id", 0)))
    if "reference" not in record:
        raise ValueError("Official OD-chord pseudo-target diagnosis requires an explicit reference field.")
    trajectory = np.asarray(record.get("trajectory", []), dtype=np.float64)
    pseudo = np.asarray(record.get("pseudo_trajectory", trajectory), dtype=np.float64)
    reference = np.asarray(record["reference"], dtype=np.float64)
    origin = trajectory[0] if len(trajectory) else None
    lat_ref = float(np.nanmean(trajectory[:, 1])) if len(trajectory) else None

    traj_m = to_local_m(trajectory, origin=origin, lat_ref=lat_ref)
    pseudo_m = to_local_m(pseudo, origin=origin, lat_ref=lat_ref)
    ref_m = to_local_m(reference, origin=origin, lat_ref=lat_ref)

    traj_len = path_length_m(traj_m)
    pseudo_len = path_length_m(pseudo_m)
    traj_curv = curvature_stats(traj_m)
    pseudo_curv = curvature_stats(pseudo_m)
    ref_dist = reference_distance_m(pseudo_m, ref_m)
    drift = endpoint_drift_m(pseudo_m, traj_m)
    jumps = jump_stats(pseudo_m, args.max_step_m)

    metrics = {
        "sample_id": int(record.get("sample_id", record.get("counterfactual_id", -1))),
        "counterfactual_id": int(record.get("counterfactual_id", -1)),
        "counterfactual_enabled": bool(record.get("counterfactual", {}).get("enabled", False)),
        "field": str(record.get("counterfactual", {}).get("field", "original")),
        "to": str(record.get("counterfactual", {}).get("to", "original")),
        "trajectory_length_m": traj_len,
        "pseudo_length_m": pseudo_len,
        "length_ratio": pseudo_len / max(traj_len, 1e-6),
        "trajectory_curvature_total_abs": traj_curv["total_abs"],
        "pseudo_curvature_total_abs": pseudo_curv["total_abs"],
        "curvature_ratio": pseudo_curv["total_abs"] / max(traj_curv["total_abs"], 1e-6),
        "mean_reference_distance_m": ref_dist["mean"],
        "p90_reference_distance_m": ref_dist["p90"],
        "max_reference_distance_m": ref_dist["max"],
        "endpoint_drift_start_m": drift["start"],
        "endpoint_drift_end_m": drift["end"],
        "endpoint_drift_max_m": drift["max"],
        "max_step_m": jumps["max_step"],
        "jump_rate": jumps["jump_rate"],
        "jump_count": jumps["jump_count"],
    }
    reasons = check_quality(metrics, args)
    metrics["quality_pass"] = not reasons
    metrics["fail_reasons"] = reasons
    return metrics


def grouped_summary(rows: List[Dict], key_fields: List[str]) -> Dict[str, Dict]:
    groups = defaultdict(list)
    for row in rows:
        key = ":".join(str(row.get(field, "")) for field in key_fields)
        groups[key].append(row)

    output = {}
    metric_names = [
        "length_ratio",
        "curvature_ratio",
        "mean_reference_distance_m",
        "p90_reference_distance_m",
        "endpoint_drift_max_m",
        "max_step_m",
        "jump_rate",
    ]
    for key, items in sorted(groups.items()):
        fail_reasons = Counter(reason for item in items for reason in item["fail_reasons"])
        output[key] = {
            "count": len(items),
            "pass_rate": float(np.mean([item["quality_pass"] for item in items])),
            "fail_reasons": dict(sorted(fail_reasons.items())),
            "metrics": {name: summarize([item[name] for item in items]) for name in metric_names},
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose pseudo target geometry quality for counterfactual instruction data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "local_porto_od_chord_256_verify"
        / "controllability_data"
        / "porto_counterfactual_instruction_samples.filtered.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "pseudo_target_quality")
    parser.add_argument("--max-samples", type=int, default=0)
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
    parser.add_argument("--top-k-worst", type=int, default=10)
    args = parser.parse_args()
    args.field_reference_limits = parse_field_float_map(args.field_max_mean_reference_distance_m)

    rows = [record_metrics(record, args) for record in read_jsonl(args.input, args.max_samples)]
    if not rows:
        raise ValueError(f"No records loaded from {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = [
        "length_ratio",
        "curvature_ratio",
        "mean_reference_distance_m",
        "p90_reference_distance_m",
        "endpoint_drift_max_m",
        "max_step_m",
        "jump_rate",
    ]
    fail_reasons = Counter(reason for row in rows for reason in row["fail_reasons"])
    worst = sorted(
        rows,
        key=lambda row: (
            row["endpoint_drift_max_m"],
            row["max_step_m"],
            row["mean_reference_distance_m"],
            row["curvature_ratio"],
        ),
        reverse=True,
    )[: max(0, args.top_k_worst)]
    summary = {
        "input": str(args.input),
        "num_records": len(rows),
        "num_counterfactual": int(sum(row["counterfactual_enabled"] for row in rows)),
        "quality_thresholds": {
            "min_length_ratio": args.min_length_ratio,
            "max_length_ratio": args.max_length_ratio,
            "max_curvature_ratio": args.max_curvature_ratio,
            "max_mean_reference_distance_m": args.max_mean_reference_distance_m,
            "field_max_mean_reference_distance_m": args.field_reference_limits,
            "max_endpoint_drift_m": args.max_endpoint_drift_m,
            "max_step_m": args.max_step_m,
        },
        "pass_rate": float(np.mean([row["quality_pass"] for row in rows])),
        "fail_reasons": dict(sorted(fail_reasons.items())),
        "overall": {name: summarize([row[name] for row in rows]) for name in metric_names},
        "by_field": grouped_summary(rows, ["field"]),
        "by_field_to": grouped_summary(rows, ["field", "to"]),
        "worst_examples": worst,
    }

    summary_path = args.output_dir / "pseudo_target_quality_summary.json"
    rows_path = args.output_dir / "pseudo_target_quality_rows.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
