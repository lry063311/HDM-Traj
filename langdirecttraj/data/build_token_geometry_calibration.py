from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None

from langdirecttraj.data.instruction_dataset import validate_official_record_protocol
from langdirecttraj.data.compact_store import (
    expand_compact_record,
    is_compact_meta_path,
    load_compact_manifest,
    open_compact_arrays,
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
                    f"[calibration] {desc}: {idx:,}/{total:,} ({idx / max(total, 1):.1%}), "
                    f"{rate:.1f} record/s, eta={eta / 60.0:.1f} min",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(f"[calibration] {desc}: {idx:,}, {rate:.1f} record/s", file=sys.stderr, flush=True)
        yield item


def read_jsonl(path: Path, max_samples: int = 0) -> Iterable[Dict]:
    manifest = load_compact_manifest(path) if is_compact_meta_path(path) else None
    arrays = open_compact_arrays(manifest) if manifest else {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            if line.strip():
                record = json.loads(line)
                if manifest is not None:
                    record = expand_compact_record(record, arrays)
                yield record


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


def local_normalized_pair(traj_lonlat: np.ndarray, ref_lonlat: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    traj_lonlat = np.asarray(traj_lonlat, dtype=np.float64)
    ref_lonlat = np.asarray(ref_lonlat, dtype=np.float64)
    if len(ref_lonlat) < 2:
        ref_lonlat = traj_lonlat
    origin = traj_lonlat[:1]
    all_points = np.concatenate([traj_lonlat, ref_lonlat], axis=0)
    lat_mean = float(np.nanmean(all_points[:, 1]))
    meter_scale = np.array(
        [
            111_320.0 * math.cos(math.radians(lat_mean)),
            110_540.0,
        ],
        dtype=np.float64,
    )
    traj_m = (traj_lonlat - origin) * meter_scale
    ref_m = (ref_lonlat - origin) * meter_scale
    route_scale = float(np.linalg.norm(ref_m[1:] - ref_m[:-1], axis=1).sum()) if len(ref_m) > 1 else 0.0
    route_scale = max(route_scale, 100.0)
    return traj_m / route_scale, ref_m / route_scale, route_scale


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())


def curvature_energy(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    diffs = points[1:] - points[:-1]
    v1 = diffs[:-1]
    v2 = diffs[1:]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = np.sum(v1 * v2, axis=1)
    return float(np.abs(np.arctan2(cross, dot)).mean())


def contract_curvature_score(points: np.ndarray, route_scale_m: float) -> float:
    """Length-normalized strong-turn curvature aligned with geometry_contract.

    The instruction labels use strong route turns normalized by physical route
    length.  A per-step mean angle can make long smooth trips and winding trips
    overlap, so calibration should expose this contract-facing descriptor.
    """

    if len(points) < 3:
        return 0.0
    diffs = points[1:] - points[:-1]
    valid = np.linalg.norm(diffs, axis=1) > 1e-8
    diffs = diffs[valid]
    if len(diffs) < 2:
        return 0.0
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    deltas = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    abs_deltas = np.abs(deltas)
    strong_total = float(abs_deltas[abs_deltas > np.deg2rad(35.0)].sum())
    length_m = path_length(points) * max(float(route_scale_m), 1e-6)
    return strong_total / max(length_m / 100.0, 1.0)


def turn_density_score(points: np.ndarray, route_scale_m: float) -> float:
    """Continuous turn-density descriptor for trainable curvature control.

    The hard contract label uses simplified strong turns, which is useful for
    discrete labels but too brittle for calibration and loss gradients.  This
    keeps the same physical normalization, while using all local turn angles.
    """

    if len(points) < 3:
        return 0.0
    diffs = points[1:] - points[:-1]
    valid = np.linalg.norm(diffs, axis=1) > 1e-8
    diffs = diffs[valid]
    if len(diffs) < 2:
        return 0.0
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    deltas = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    length_m = path_length(points) * max(float(route_scale_m), 1e-6)
    return float(np.abs(deltas).sum()) / max(length_m / 100.0, 1.0)


def heading_stats(points: np.ndarray) -> Dict[str, float]:
    if len(points) < 3:
        return {
            "signed_net_heading_rad": 0.0,
            "total_abs_heading_rad": 0.0,
            "signed_total_turn_rad": 0.0,
            "left_turn_ratio": 0.0,
            "right_turn_ratio": 0.0,
            "max_abs_turn_rad": 0.0,
            "turn_count": 0.0,
        }
    diffs = points[1:] - points[:-1]
    valid = np.linalg.norm(diffs, axis=1) > 1e-8
    diffs = diffs[valid]
    if len(diffs) < 2:
        return {
            "signed_net_heading_rad": 0.0,
            "total_abs_heading_rad": 0.0,
            "signed_total_turn_rad": 0.0,
            "left_turn_ratio": 0.0,
            "right_turn_ratio": 0.0,
            "max_abs_turn_rad": 0.0,
            "turn_count": 0.0,
        }
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    deltas = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    net = math.atan2(math.sin(float(headings[-1] - headings[0])), math.cos(float(headings[-1] - headings[0])))
    abs_deltas = np.abs(deltas)
    turn_threshold = np.deg2rad(15.0)
    strong = abs_deltas > turn_threshold
    left = deltas > turn_threshold
    right = deltas < -turn_threshold
    strong_count = max(float(strong.sum()), 1.0)
    return {
        "signed_net_heading_rad": float(net),
        "total_abs_heading_rad": float(abs_deltas.sum()),
        "signed_total_turn_rad": float(deltas.sum()),
        "left_turn_ratio": float(left.sum() / strong_count),
        "right_turn_ratio": float(right.sum() / strong_count),
        "max_abs_turn_rad": float(abs_deltas.max()) if abs_deltas.size else 0.0,
        "turn_count": float(strong.sum()),
    }


def local_direction_stats(points: np.ndarray) -> Dict[str, float]:
    """iMotion-style direction descriptors in the trajectory's initial frame."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return {
            "local_dx": 0.0,
            "local_dy": 0.0,
            "local_heading_diff_rad": 0.0,
            "local_abs_heading_diff_rad": 0.0,
            "local_signed_total_turn_rad": 0.0,
            "local_total_abs_turn_rad": 0.0,
            "local_left_turn_ratio": 0.0,
            "local_right_turn_ratio": 0.0,
            "local_max_abs_turn_rad": 0.0,
            "local_turn_count": 0.0,
        }
    diffs = points[1:] - points[:-1]
    valid = np.linalg.norm(diffs, axis=1) > 1e-8
    diffs = diffs[valid]
    if len(diffs) < 2:
        return {
            "local_dx": 0.0,
            "local_dy": 0.0,
            "local_heading_diff_rad": 0.0,
            "local_abs_heading_diff_rad": 0.0,
            "local_signed_total_turn_rad": 0.0,
            "local_total_abs_turn_rad": 0.0,
            "local_left_turn_ratio": 0.0,
            "local_right_turn_ratio": 0.0,
            "local_max_abs_turn_rad": 0.0,
            "local_turn_count": 0.0,
        }

    start_heading = math.atan2(float(diffs[0, 1]), float(diffs[0, 0]))
    cos_h = math.cos(-start_heading)
    sin_h = math.sin(-start_heading)
    rot = np.asarray([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
    local = (points - points[:1]) @ rot.T
    local_diffs = local[1:] - local[:-1]
    valid = np.linalg.norm(local_diffs, axis=1) > 1e-8
    local_diffs = local_diffs[valid]
    if len(local_diffs) < 2:
        return {
            "local_dx": float(local[-1, 0] - local[0, 0]),
            "local_dy": float(local[-1, 1] - local[0, 1]),
            "local_heading_diff_rad": 0.0,
            "local_abs_heading_diff_rad": 0.0,
            "local_signed_total_turn_rad": 0.0,
            "local_total_abs_turn_rad": 0.0,
            "local_left_turn_ratio": 0.0,
            "local_right_turn_ratio": 0.0,
            "local_max_abs_turn_rad": 0.0,
            "local_turn_count": 0.0,
        }

    headings = np.arctan2(local_diffs[:, 1], local_diffs[:, 0])
    deltas = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    heading_diff = math.atan2(math.sin(float(headings[-1] - headings[0])), math.cos(float(headings[-1] - headings[0])))
    abs_deltas = np.abs(deltas)
    turn_threshold = np.deg2rad(15.0)
    strong = abs_deltas > turn_threshold
    strong_count = max(float(strong.sum()), 1.0)
    return {
        "local_dx": float(local[-1, 0] - local[0, 0]),
        "local_dy": float(local[-1, 1] - local[0, 1]),
        "local_heading_diff_rad": float(heading_diff),
        "local_abs_heading_diff_rad": float(abs(heading_diff)),
        "local_signed_total_turn_rad": float(deltas.sum()),
        "local_total_abs_turn_rad": float(abs_deltas.sum()),
        "local_left_turn_ratio": float((deltas > turn_threshold).sum() / strong_count),
        "local_right_turn_ratio": float((deltas < -turn_threshold).sum() / strong_count),
        "local_max_abs_turn_rad": float(abs_deltas.max()) if abs_deltas.size else 0.0,
        "local_turn_count": float(strong.sum()),
    }


def reference_lateral_stats(points: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    if len(points) < 2 or len(reference) < 2:
        return {
            "mean_signed_lateral_offset": 0.0,
            "mean_abs_lateral_offset": 0.0,
            "left_side_ratio": 0.0,
            "right_side_ratio": 0.0,
        }
    if len(reference) != len(points):
        reference = resample_polyline(reference, len(points))
    diffs = np.zeros_like(reference, dtype=np.float64)
    diffs[1:-1] = reference[2:] - reference[:-2]
    diffs[0] = reference[1] - reference[0]
    diffs[-1] = reference[-1] - reference[-2]
    denom = np.linalg.norm(diffs, axis=1, keepdims=True).clip(min=1e-8)
    tangent = diffs / denom
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    lateral = np.sum((points - reference) * normal, axis=1)
    threshold = max(float(np.percentile(np.abs(lateral), 50)) * 0.25, 1e-4)
    return {
        "mean_signed_lateral_offset": float(lateral.mean()),
        "mean_abs_lateral_offset": float(np.abs(lateral).mean()),
        "left_side_ratio": float(np.mean(lateral > threshold)),
        "right_side_ratio": float(np.mean(lateral < -threshold)),
    }


def geometry_metrics(record: Dict, target_key: str, target_len: int) -> Dict[str, float] | None:
    target = np.asarray(record.get(target_key, record.get("trajectory", [])), dtype=np.float64)
    validate_official_record_protocol(record, row_index=int(record.get("sample_id", 0)))
    if "reference" not in record:
        raise ValueError("Official OD-chord calibration requires an explicit reference field.")
    reference = np.asarray(record["reference"], dtype=np.float64)
    if target.ndim != 2 or target.shape[-1] != 2 or len(target) < 2:
        return None
    if reference.ndim != 2 or reference.shape[-1] != 2 or len(reference) < 2:
        raise ValueError("Official OD-chord calibration received an invalid reference polyline.")
    traj_norm, ref_norm, route_scale_m = local_normalized_pair(target, reference)
    traj_norm = resample_polyline(traj_norm, target_len)
    ref_norm = resample_polyline(ref_norm, target_len)

    target_len_norm = path_length(traj_norm)
    ref_len_norm = path_length(ref_norm)
    target_len_m = target_len_norm * float(route_scale_m)
    displacement = float(np.linalg.norm(traj_norm[-1] - traj_norm[0]))
    record_metrics = record.get("metrics", {}) if isinstance(record.get("metrics", {}), dict) else {}
    trip_attrs = record.get("trip_attributes", {}) if isinstance(record.get("trip_attributes", {}), dict) else {}
    duration_min = float(
        record_metrics.get(
            "trip_duration_min",
            trip_attrs.get("trip_duration_min", 0.0),
        )
        or 0.0
    )
    if duration_min > 1e-6:
        avg_speed_kmh = float(np.clip((target_len_m / 1000.0) / (duration_min / 60.0), 0.0, 120.0))
    else:
        avg_speed_kmh = float(record_metrics.get("avg_speed_kmh", trip_attrs.get("avg_speed_kmh", 0.0)) or 0.0)
    heading = heading_stats(traj_norm)
    local_direction = local_direction_stats(traj_norm)
    lateral = reference_lateral_stats(traj_norm, ref_norm)
    return {
        "mean_reference_distance": float(np.linalg.norm(traj_norm - ref_norm, axis=1).mean()),
        "curvature_energy": curvature_energy(traj_norm),
        "contract_curvature_score": contract_curvature_score(traj_norm, route_scale_m),
        "turn_density_score": turn_density_score(traj_norm, route_scale_m),
        "progress_ratio": displacement / max(target_len_norm, 1e-8),
        "avg_speed_kmh": avg_speed_kmh,
        "time_per_km_min": float(duration_min / max(target_len_m / 1000.0, 1e-6)) if duration_min > 1e-6 else 0.0,
        "length_m": target_len_m,
        "length_ratio": target_len_norm / max(ref_len_norm, 1e-8),
        "signed_net_heading_rad": heading["signed_net_heading_rad"],
        "total_abs_heading_rad": heading["total_abs_heading_rad"],
        "signed_total_turn_rad": heading["signed_total_turn_rad"],
        "left_turn_ratio": heading["left_turn_ratio"],
        "right_turn_ratio": heading["right_turn_ratio"],
        "max_abs_turn_rad": heading["max_abs_turn_rad"],
        "turn_count": heading["turn_count"],
        **local_direction,
        **lateral,
    }


def summarize(values: List[float], min_scale: float) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "iqr": float(min_scale),
            "scale": float(min_scale),
        }
    p25, p50, p75 = np.percentile(arr, [25, 50, 75])
    iqr = float(p75 - p25)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "iqr": iqr,
        "scale": float(max(iqr, min_scale)),
    }


DIRECTION_CANDIDATE_METRICS = [
    "signed_net_heading_rad",
    "total_abs_heading_rad",
    "signed_total_turn_rad",
    "left_turn_ratio",
    "right_turn_ratio",
    "max_abs_turn_rad",
    "turn_count",
    "mean_signed_lateral_offset",
    "mean_abs_lateral_offset",
    "left_side_ratio",
    "right_side_ratio",
    "local_dx",
    "local_dy",
    "local_heading_diff_rad",
    "local_abs_heading_diff_rad",
    "local_signed_total_turn_rad",
    "local_total_abs_turn_rad",
    "local_left_turn_ratio",
    "local_right_turn_ratio",
    "local_max_abs_turn_rad",
    "local_turn_count",
]


TOKEN_DIRECTION_CANDIDATE_METRICS = {
    "straight": [
        "local_abs_heading_diff_rad",
        "local_total_abs_turn_rad",
        "mean_abs_lateral_offset",
    ],
    "left_turn": [
        "local_signed_total_turn_rad",
        "local_dy",
        "local_left_turn_ratio",
    ],
    "right_turn": [
        "local_signed_total_turn_rad",
        "local_dy",
        "local_right_turn_ratio",
    ],
    "u_turn": [
        "local_abs_heading_diff_rad",
        "local_dx",
        "local_max_abs_turn_rad",
    ],
    "multi_turn": [
        "local_total_abs_turn_rad",
        "local_turn_count",
        "local_max_abs_turn_rad",
    ],
}


def direction_metric_for_token(token: str) -> str:
    if token == "left_turn":
        return "local_signed_total_turn_rad"
    if token == "right_turn":
        return "local_signed_total_turn_rad"
    if token == "u_turn":
        return "local_abs_heading_diff_rad"
    if token == "multi_turn":
        return "local_total_abs_turn_rad"
    return "local_abs_heading_diff_rad"


def choose_direction_metric(
    token: str,
    grouped_metrics: Dict[str, Dict[str, List[float]]],
    min_scale: float,
) -> tuple[str, float]:
    best_metric = direction_metric_for_token(token)
    best_score = -1.0
    candidate_metrics = TOKEN_DIRECTION_CANDIDATE_METRICS.get(token, DIRECTION_CANDIDATE_METRICS)
    for metric in candidate_metrics:
        positive = np.asarray(grouped_metrics.get(token, {}).get(metric, []), dtype=np.float64)
        negative_values: List[float] = []
        for other_token, metric_values in grouped_metrics.items():
            if other_token == token:
                continue
            negative_values.extend(metric_values.get(metric, []))
        negative = np.asarray(negative_values, dtype=np.float64)
        positive = positive[np.isfinite(positive)]
        negative = negative[np.isfinite(negative)]
        if positive.size < 2 or negative.size < 2:
            continue
        pos_q25, pos_p50, pos_q75 = np.percentile(positive, [25, 50, 75])
        neg_q25, neg_p50, neg_q75 = np.percentile(negative, [25, 50, 75])
        scale = max(float(pos_q75 - pos_q25), float(neg_q75 - neg_q25), float(min_scale), 1e-6)
        score = abs(float(pos_p50 - neg_p50)) / scale
        if score > best_score:
            best_score = score
            best_metric = metric
    return best_metric, float(max(best_score, 0.0))


def build_calibration(args: argparse.Namespace) -> Dict:
    grouped = defaultdict(lambda: defaultdict(list))
    direction_grouped = defaultdict(lambda: defaultdict(list))
    global_values = defaultdict(list)
    counts = defaultdict(lambda: defaultdict(int))

    for record in progress_iter(
        read_jsonl(args.input, args.max_samples),
        "build token-geometry calibration",
        args.max_samples or None,
    ):
        metrics = geometry_metrics(record, args.target_key, args.target_len)
        labels = record.get("labels", {})
        if metrics is None or not isinstance(labels, dict):
            continue
        cf = record.get("counterfactual", {})
        active_field = str(cf.get("field", "")) if bool(cf.get("enabled", False)) else ""
        for name, value in metrics.items():
            global_values[name].append(float(value))
        for field in DEFAULT_MOBILITY_VOCAB.fields:
            if bool(args.active_field_only) and active_field and active_field != field:
                continue
            token = str(labels.get(field, "unknown"))
            if token not in DEFAULT_MOBILITY_VOCAB.token_to_id[field]:
                continue
            if int(args.max_token_samples_per_field) > 0 and len(grouped[field][token]) >= int(args.max_token_samples_per_field):
                continue
            counts[field][token] += 1
            if field == "reference_adherence":
                grouped[field][token].append(metrics["mean_reference_distance"])
            elif field == "curvature":
                grouped[field][token].append(metrics["turn_density_score"])
            elif field == "progress":
                grouped[field][token].append(metrics["avg_speed_kmh"])
            elif field == "direction":
                grouped[field][token].append(metrics[direction_metric_for_token(token)])
                for metric_name in DIRECTION_CANDIDATE_METRICS:
                    direction_grouped[token][metric_name].append(metrics[metric_name])
            elif field == "distance":
                grouped[field][token].append(metrics["length_m"])

    metric_min_scales = {
        "reference_adherence": float(args.min_reference_scale),
        "curvature": float(args.min_curvature_scale),
        "progress": float(args.min_progress_scale),
        "direction": float(args.min_direction_scale),
        "distance": float(args.min_distance_scale),
    }
    metric_names = {
        "reference_adherence": "mean_reference_distance",
        "curvature": "turn_density_score",
        "progress": "avg_speed_kmh",
        "direction": "direction_token_metric",
        "distance": "length_m",
    }

    fields = {}
    for field in DEFAULT_MOBILITY_VOCAB.fields:
        fields[field] = {
            "metric": metric_names[field],
            "tokens": {},
        }
        for token in DEFAULT_MOBILITY_VOCAB.token_to_id[field]:
            direction_metric = direction_metric_for_token(token)
            direction_metric_score = None
            values = grouped[field][token]
            if field == "direction" and args.direction_metric_mode == "signed_lr" and token in {"left_turn", "right_turn"}:
                direction_metric = "local_signed_total_turn_rad"
                values = direction_grouped[token][direction_metric]
                direction_metric_score = None
            elif field == "direction" and args.direction_metric_mode in {"auto", "signed_lr"}:
                direction_metric, direction_metric_score = choose_direction_metric(
                    token,
                    direction_grouped,
                    metric_min_scales[field],
                )
                values = direction_grouped[token][direction_metric]
            stats = summarize(values, metric_min_scales[field])
            if stats["count"] < int(args.min_count):
                fallback_metric = (
                    direction_metric if field == "direction" else metric_names[field]
                )
                fallback_key = {
                    "mean_reference_distance": "mean_reference_distance",
                    "curvature_energy": "curvature_energy",
                    "contract_curvature_score": "contract_curvature_score",
                    "turn_density_score": "turn_density_score",
                    "progress_ratio": "progress_ratio",
                    "avg_speed_kmh": "avg_speed_kmh",
                    "time_per_km_min": "time_per_km_min",
                    "length_m": "length_m",
                    "length_ratio": "length_ratio",
                    "signed_net_heading_rad": "signed_net_heading_rad",
                    "total_abs_heading_rad": "total_abs_heading_rad",
                    "signed_total_turn_rad": "signed_total_turn_rad",
                    "left_turn_ratio": "left_turn_ratio",
                    "right_turn_ratio": "right_turn_ratio",
                    "max_abs_turn_rad": "max_abs_turn_rad",
                    "turn_count": "turn_count",
                    "mean_signed_lateral_offset": "mean_signed_lateral_offset",
                    "mean_abs_lateral_offset": "mean_abs_lateral_offset",
                    "left_side_ratio": "left_side_ratio",
                    "right_side_ratio": "right_side_ratio",
                    "local_dx": "local_dx",
                    "local_dy": "local_dy",
                    "local_heading_diff_rad": "local_heading_diff_rad",
                    "local_abs_heading_diff_rad": "local_abs_heading_diff_rad",
                    "local_signed_total_turn_rad": "local_signed_total_turn_rad",
                    "local_total_abs_turn_rad": "local_total_abs_turn_rad",
                    "local_left_turn_ratio": "local_left_turn_ratio",
                    "local_right_turn_ratio": "local_right_turn_ratio",
                    "local_max_abs_turn_rad": "local_max_abs_turn_rad",
                    "local_turn_count": "local_turn_count",
                }[fallback_metric]
                fallback = summarize(global_values[fallback_key], metric_min_scales[field])
                stats = {
                    **stats,
                    "p50": fallback["p50"],
                    "scale": max(fallback["scale"], metric_min_scales[field]),
                    "fallback": True,
                }
            else:
                stats["fallback"] = False
            if field == "direction":
                stats["direction_metric"] = direction_metric
                if direction_metric_score is not None:
                    stats["direction_metric_score"] = direction_metric_score
            fields[field]["tokens"][token] = stats

    if str(args.reference_order_mode) != "none":
        _enforce_reference_order(
            fields["reference_adherence"]["tokens"],
            min_step=float(args.reference_order_min_step),
            scale=float(args.reference_order_scale),
        )
    if str(args.progress_order_mode) != "none":
        _enforce_progress_order(
            fields["progress"]["tokens"],
            min_step=float(args.progress_order_min_step),
            scale=float(args.progress_order_scale),
        )
    if str(args.curvature_order_mode) != "none":
        _enforce_curvature_order(
            fields["curvature"]["tokens"],
            min_step=float(args.curvature_order_min_step),
            scale=float(args.curvature_order_scale),
        )
    _enforce_direction_left_right_contract(
        fields["direction"]["tokens"],
        min_step=float(args.direction_lr_contract_min_step),
        scale=float(args.direction_lr_contract_scale),
    )

    return {
        "schema_version": 1,
        "input": str(args.input),
        "target_key": str(args.target_key),
        "target_len": int(args.target_len),
        "feature_units": "local_normalized_decoder_coordinates",
        "min_count": int(args.min_count),
        "direction_metric_mode": str(args.direction_metric_mode),
        "reference_order_mode": str(args.reference_order_mode),
        "progress_order_mode": str(args.progress_order_mode),
        "curvature_order_mode": str(args.curvature_order_mode),
        "active_field_only": bool(args.active_field_only),
        "max_token_samples_per_field": int(args.max_token_samples_per_field),
        "fields": fields,
        "label_counts": {field: dict(sorted(values.items())) for field, values in counts.items()},
        "global_metrics": {name: summarize(values, 1e-6) for name, values in sorted(global_values.items())},
    }


def _enforce_reference_order(
    token_stats: Dict[str, Dict[str, float]],
    min_step: float,
    scale: float,
) -> None:
    """Make reference-adherence an ordered calibration variable.

    `mean_reference_distance` is monotonic by definition: tight should be
    closest to the reference, loose should be farthest. Counterfactual pseudo
    targets can make the raw per-token medians cross because direction and
    curvature transforms also move the trajectory. This post-process keeps the
    calibration data-driven but restores the ordinal contract used by Table 3/4.
    """
    order = ["tight", "moderate", "unknown", "loose"]
    available = [
        float(token_stats[token].get("p50", token_stats[token].get("mean", 0.0)))
        for token in order
        if token in token_stats
    ]
    if not available:
        return
    values = sorted(value for value in available if math.isfinite(value))
    if not values:
        return
    while len(values) < len(order):
        values.append(values[-1] + float(min_step))
    ordered_values = []
    last = None
    for value in values[: len(order)]:
        if last is None:
            adjusted = float(value)
        else:
            adjusted = max(float(value), last + float(min_step))
        ordered_values.append(adjusted)
        last = adjusted
    for token, center in zip(order, ordered_values):
        if token not in token_stats:
            continue
        raw_scale = float(token_stats[token].get("scale", token_stats[token].get("iqr", 1.0)))
        token_stats[token]["p50_raw"] = token_stats[token].get("p50", token_stats[token].get("mean", 0.0))
        token_stats[token]["scale_raw"] = raw_scale
        token_stats[token]["p50"] = float(center)
        token_stats[token]["scale"] = float(max(raw_scale, scale, min_step, 1e-6))
        token_stats[token]["ordered_reference"] = True


def _enforce_progress_order(
    token_stats: Dict[str, Dict[str, float]],
    min_step: float,
    scale: float,
) -> None:
    """Restore the ordinal contract for progress centers.

    Progress is path efficiency, so the intended geometry order is
    slow < normal < fast.  Counterfactual pseudo targets can pollute these
    centers because changing progress often changes curvature and route shape.
    Keeping the center order explicit prevents token losses and Table-4 target
    effects from learning a reversed or nearly flat progress signal.
    """
    order = ["slow", "normal", "fast"]
    available = [
        float(token_stats[token].get("p50", token_stats[token].get("mean", 0.0)))
        for token in order
        if token in token_stats
    ]
    if not available:
        return
    values = sorted(value for value in available if math.isfinite(value))
    if not values:
        return
    while len(values) < len(order):
        values.append(values[-1] + float(min_step))
    ordered_values = []
    last = None
    for value in values[: len(order)]:
        if last is None:
            adjusted = float(value)
        else:
            adjusted = max(float(value), last + float(min_step))
        ordered_values.append(adjusted)
        last = adjusted
    for token, center in zip(order, ordered_values):
        if token not in token_stats:
            continue
        raw_scale = float(token_stats[token].get("scale", token_stats[token].get("iqr", 1.0)))
        token_stats[token]["p50_raw"] = token_stats[token].get("p50", token_stats[token].get("mean", 0.0))
        token_stats[token]["scale_raw"] = raw_scale
        token_stats[token]["p50"] = float(center)
        token_stats[token]["scale"] = float(max(raw_scale, scale, min_step, 1e-6))
        token_stats[token]["ordered_progress"] = True


def _enforce_curvature_order(
    token_stats: Dict[str, Dict[str, float]],
    min_step: float,
    scale: float,
) -> None:
    """Restore the ordinal contract for curvature centers.

    Curvature labels have an explicit geometry order:
    smooth < moderate < winding.  Counterfactual pseudo targets can make the
    empirical medians cross after quality filtering, especially when winding
    samples with very high collateral are dropped.  Training and Table 4 both
    need a monotone target surface, so we keep the observed center range but
    impose the ordinal contract.
    """

    order = ["smooth", "moderate", "winding"]
    available = [
        float(token_stats[token].get("p50", token_stats[token].get("mean", 0.0)))
        for token in order
        if token in token_stats
    ]
    if not available:
        return
    values = sorted(value for value in available if math.isfinite(value))
    if not values:
        return
    while len(values) < len(order):
        values.append(values[-1] + float(min_step))
    ordered_values = []
    last = None
    for value in values[: len(order)]:
        if last is None:
            adjusted = float(value)
        else:
            adjusted = max(float(value), last + float(min_step))
        ordered_values.append(adjusted)
        last = adjusted
    for token, center in zip(order, ordered_values):
        if token not in token_stats:
            continue
        raw_scale = float(token_stats[token].get("scale", token_stats[token].get("iqr", 1.0)))
        token_stats[token]["p50_raw"] = token_stats[token].get("p50", token_stats[token].get("mean", 0.0))
        token_stats[token]["scale_raw"] = raw_scale
        token_stats[token]["p50"] = float(center)
        token_stats[token]["scale"] = float(max(raw_scale, scale, min_step, 1e-6))
        token_stats[token]["ordered_curvature"] = True


def _enforce_direction_left_right_contract(
    token_stats: Dict[str, Dict[str, float]],
    min_step: float,
    scale: float,
) -> None:
    """Align left/right signed-turn centers with the official evaluator.

    The OD-chord evaluator and direction-control tests define a left turn as a
    positive local signed turn and a right turn as a negative local signed turn.
    Counterfactual pseudo targets can invert the empirical medians, especially
    when endpoint-preserving sine templates are mixed with raw Porto geometry.
    Keep the empirical magnitudes and scales, but enforce the semantic sign so
    calibration losses and decoder control optimize the same contract.
    """

    for token, expected_sign in (("left_turn", 1.0), ("right_turn", -1.0)):
        if token not in token_stats:
            continue
        stats = token_stats[token]
        if str(stats.get("direction_metric", "")) != "local_signed_total_turn_rad":
            continue
        raw_center = float(stats.get("p50", stats.get("mean", 0.0)))
        raw_scale = float(stats.get("scale", stats.get("iqr", stats.get("std", 1.0))))
        magnitude = max(abs(raw_center), float(min_step), 1e-6)
        stats["p50_raw"] = raw_center
        stats["scale_raw"] = raw_scale
        stats["p50"] = float(expected_sign * magnitude)
        stats["scale"] = float(max(raw_scale, scale, min_step, 1e-6))
        stats["direction_lr_contract"] = "left_positive_right_negative"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data-driven token-to-geometry calibration for reranker features.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "local_porto_od_chord_256_verify"
        / "controllability_data"
        / "porto_counterfactual_instruction_samples.filtered.jsonl",
    )
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "calibration" / "porto_token_geometry_calibration.json")
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--target-len", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument("--min-reference-scale", type=float, default=0.005)
    parser.add_argument("--min-curvature-scale", type=float, default=0.005)
    parser.add_argument("--min-progress-scale", type=float, default=0.03)
    parser.add_argument("--min-direction-scale", type=float, default=0.15)
    parser.add_argument("--min-distance-scale", type=float, default=0.05)
    parser.add_argument(
        "--active-field-only",
        action="store_true",
        help=(
            "For counterfactual datasets, estimate each field from original records plus "
            "counterfactuals that actually changed that field. This avoids cross-field "
            "duplicates polluting token geometry centers."
        ),
    )
    parser.add_argument(
        "--max-token-samples-per-field",
        type=int,
        default=0,
        help="Optional cap per field/token when estimating calibration centers. 0 disables capping.",
    )
    parser.add_argument(
        "--direction-metric-mode",
        choices=["fixed", "auto", "signed_lr"],
        default="fixed",
        help=(
            "Use fixed direction token metrics, select the most separable metric per token, "
            "or force left/right onto local_signed_total_turn_rad while auto-selecting the other classes."
        ),
    )
    parser.add_argument(
        "--direction-lr-contract-min-step",
        type=float,
        default=0.15,
        help="Minimum absolute left/right signed-turn center after enforcing the official direction contract.",
    )
    parser.add_argument(
        "--direction-lr-contract-scale",
        type=float,
        default=0.15,
        help="Minimum left/right scale after enforcing the official direction contract.",
    )
    parser.add_argument(
        "--reference-order-mode",
        choices=["none", "ordered"],
        default="none",
        help="Optionally enforce ordered centers for reference_adherence.",
    )
    parser.add_argument("--reference-order-min-step", type=float, default=0.006)
    parser.add_argument(
        "--reference-order-scale",
        type=float,
        default=0.0,
        help=(
            "Minimum scale used after enforcing ordered reference-adherence centers. "
            "The final scale keeps the token's data-driven IQR, so Table 4 collateral "
            "is not inflated by an overly narrow fixed reference scale."
        ),
    )
    parser.add_argument(
        "--progress-order-mode",
        choices=["none", "ordered"],
        default="none",
        help="Optionally enforce ordered centers for progress: slow < normal < fast.",
    )
    parser.add_argument("--progress-order-min-step", type=float, default=0.08)
    parser.add_argument(
        "--progress-order-scale",
        type=float,
        default=0.08,
        help="Minimum scale used after enforcing ordered progress centers.",
    )
    parser.add_argument(
        "--curvature-order-mode",
        choices=["none", "ordered"],
        default="none",
        help="Optionally enforce ordered centers for curvature: smooth < moderate < winding.",
    )
    parser.add_argument("--curvature-order-min-step", type=float, default=0.12)
    parser.add_argument(
        "--curvature-order-scale",
        type=float,
        default=0.12,
        help="Minimum scale used after enforcing ordered curvature centers.",
    )
    args = parser.parse_args()

    calibration = build_calibration(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    preview = {
        "output": str(args.output),
        "input": str(args.input),
        "target_key": args.target_key,
        "target_len": args.target_len,
        "label_counts": calibration["label_counts"],
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
