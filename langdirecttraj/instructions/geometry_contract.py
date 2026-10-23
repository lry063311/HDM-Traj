from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GeometryContractResult:
    labels: Dict[str, str]
    metrics: Dict[str, float]


def valid_trajectory(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[-1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    points = points[:, :2]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) <= 2:
        return points.astype(np.float32)
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    moving = diffs > 1e-9
    valid_len = len(points)
    for i in range(len(points) - 2, 0, -1):
        if moving[i]:
            valid_len = i + 2
            break
    return points[:valid_len].astype(np.float32)


def drop_duplicate_points(points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= 1:
        return points.astype(np.float32)
    keep = [0]
    for i in range(1, len(points)):
        if float(np.linalg.norm(points[i] - points[keep[-1]])) > eps:
            keep.append(i)
    return points[keep].astype(np.float32)


def meter_scale(points_lonlat: np.ndarray) -> Tuple[float, float]:
    if len(points_lonlat) == 0:
        return 111_320.0, 110_540.0
    lat = float(np.nanmean(points_lonlat[:, 1]))
    return 111_320.0 * float(np.cos(np.deg2rad(lat))), 110_540.0


def lonlat_to_local_meters(points_lonlat: np.ndarray, origin: Optional[np.ndarray] = None) -> np.ndarray:
    points = np.asarray(points_lonlat, dtype=np.float32)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    origin = points[0] if origin is None else np.asarray(origin, dtype=np.float32)
    sx, sy = meter_scale(points)
    local = np.empty_like(points, dtype=np.float32)
    local[:, 0] = (points[:, 0] - origin[0]) * sx
    local[:, 1] = (points[:, 1] - origin[1]) * sy
    return local


def path_length_m(local: np.ndarray) -> float:
    local = np.asarray(local, dtype=np.float32)
    if len(local) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(local, axis=0), axis=1).sum())


def heading_change(local: np.ndarray) -> Tuple[float, float]:
    local = np.asarray(local, dtype=np.float32)
    if len(local) < 3:
        return 0.0, 0.0
    diffs = np.diff(local, axis=0)
    valid = np.linalg.norm(diffs, axis=1) > 0.25
    diffs = diffs[valid]
    if len(diffs) < 2:
        return 0.0, 0.0
    headings = np.unwrap(np.arctan2(diffs[:, 1], diffs[:, 0]))
    deltas = np.diff(headings)
    return float(headings[-1] - headings[0]), float(np.abs(deltas).sum())


def perpendicular_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    seg = end - start
    denom = float(np.dot(seg, seg))
    if denom <= 1e-6:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, seg) / denom)
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(point - (start + t * seg)))


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= 2:
        return points
    start, end = points[0], points[-1]
    dists = np.array([perpendicular_distance(p, start, end) for p in points[1:-1]], dtype=np.float32)
    if len(dists) == 0:
        return points[[0, -1]]
    max_idx = int(np.argmax(dists)) + 1
    if float(dists[max_idx - 1]) <= epsilon:
        return points[[0, -1]]
    left = rdp(points[: max_idx + 1], epsilon)
    right = rdp(points[max_idx:], epsilon)
    return np.concatenate([left[:-1], right], axis=0)


def route_shape_stats(local: np.ndarray) -> Tuple[int, float, float]:
    local = np.asarray(local, dtype=np.float32)
    if len(local) < 3:
        return 0, 0.0, 0.0
    simplified = rdp(local, epsilon=80.0)
    if len(simplified) < 3:
        simplified = local[:: max(1, len(local) // 8)]
        if len(simplified) and not np.array_equal(simplified[-1], local[-1]):
            simplified = np.concatenate([simplified, local[-1:]], axis=0)
    diffs = np.diff(simplified, axis=0)
    valid = np.linalg.norm(diffs, axis=1) > 25.0
    diffs = diffs[valid]
    if len(diffs) < 2:
        return 0, 0.0, 0.0
    headings = np.unwrap(np.arctan2(diffs[:, 1], diffs[:, 0]))
    deltas = np.diff(headings)
    strong = np.abs(deltas) > np.deg2rad(35.0)
    return int(np.sum(strong)), float(headings[-1] - headings[0]), float(np.abs(deltas[strong]).sum())


def point_segment_distance(points: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> np.ndarray:
    points_exp = points[:, None, :]
    start = seg_start[None, :, :]
    end = seg_end[None, :, :]
    seg = end - start
    denom = np.sum(seg * seg, axis=-1).clip(min=1e-6)
    t = np.sum((points_exp - start) * seg, axis=-1) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = start + t[..., None] * seg
    return np.linalg.norm(points_exp - proj, axis=-1).min(axis=1)


def reference_metrics_from_local(traj_m: np.ndarray, ref_m: np.ndarray) -> Tuple[str, float, float]:
    if len(traj_m) == 0 or len(ref_m) < 2:
        return "unknown", float("nan"), float("nan")
    dists = point_segment_distance(traj_m, ref_m[:-1], ref_m[1:])
    mean_dist = float(np.mean(dists))
    violation_rate = float(np.mean(dists > 30.0))
    if mean_dist <= 20.0 and violation_rate <= 0.20:
        return "tight", mean_dist, violation_rate
    if mean_dist <= 60.0 and violation_rate <= 0.50:
        return "moderate", mean_dist, violation_rate
    return "loose", mean_dist, violation_rate


def reference_lateral_stats(traj_m: np.ndarray, ref_m: np.ndarray) -> Dict[str, float]:
    if len(traj_m) < 2 or len(ref_m) < 2:
        return {
            "mean_signed_lateral_offset_m": 0.0,
            "mean_abs_lateral_offset_m": 0.0,
            "left_side_ratio": 0.0,
            "right_side_ratio": 0.0,
        }
    if len(ref_m) != len(traj_m):
        t_ref = np.linspace(0.0, 1.0, len(ref_m), dtype=np.float32)
        t_traj = np.linspace(0.0, 1.0, len(traj_m), dtype=np.float32)
        ref_m = np.stack(
            [
                np.interp(t_traj, t_ref, ref_m[:, 0]),
                np.interp(t_traj, t_ref, ref_m[:, 1]),
            ],
            axis=1,
        ).astype(np.float32)
    diffs = np.zeros_like(ref_m, dtype=np.float32)
    diffs[1:-1] = ref_m[2:] - ref_m[:-2]
    diffs[0] = ref_m[1] - ref_m[0]
    diffs[-1] = ref_m[-1] - ref_m[-2]
    tangent = diffs / np.linalg.norm(diffs, axis=1, keepdims=True).clip(min=1e-6)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    lateral = np.sum((traj_m - ref_m) * normal, axis=1)
    threshold = max(float(np.percentile(np.abs(lateral), 50)) * 0.25, 5.0)
    return {
        "mean_signed_lateral_offset_m": float(lateral.mean()),
        "mean_abs_lateral_offset_m": float(np.abs(lateral).mean()),
        "left_side_ratio": float(np.mean(lateral > threshold)),
        "right_side_ratio": float(np.mean(lateral < -threshold)),
    }


def distance_label(length_m: float, dataset: str) -> str:
    dataset_key = dataset.lower()
    if dataset_key == "porto":
        bins = (1200.0, 3000.0, 7000.0)
    elif dataset_key in {"sf", "sanfrancisco", "cabspotting", "rome", "roma"}:
        bins = (5000.0, 15000.0, 30000.0)
    elif dataset_key in {"beijing", "tdrive", "t-drive"}:
        bins = (3000.0, 10000.0, 20000.0)
    else:
        bins = (800.0, 2000.0, 5000.0)
    if length_m < bins[0]:
        return "very_short"
    if length_m < bins[1]:
        return "short"
    if length_m < bins[2]:
        return "medium"
    return "long"


def direction_label(
    net_heading: float,
    total_abs_heading: float,
    turn_count: int,
    displacement_ratio: float,
    mean_signed_lateral_offset_m: float = 0.0,
    left_side_ratio: float = 0.0,
    right_side_ratio: float = 0.0,
) -> str:
    abs_net = abs(net_heading)
    if displacement_ratio < 0.25 and total_abs_heading > np.deg2rad(120.0):
        return "u_turn"
    if abs_net > np.deg2rad(135.0) and displacement_ratio < 0.55:
        return "u_turn"
    if net_heading > np.deg2rad(35.0):
        return "left_turn"
    if net_heading < -np.deg2rad(35.0):
        return "right_turn"
    # Lateral offset is a fallback only. Endpoint-preserving turn arcs can sit
    # on the opposite side of the OD chord while still ending with the correct
    # left/right heading, so lateral should not override the signed turn.
    if mean_signed_lateral_offset_m > 30.0 and left_side_ratio >= 0.45:
        return "left_turn"
    if mean_signed_lateral_offset_m < -30.0 and right_side_ratio >= 0.45:
        return "right_turn"
    # Multi-turn should mean no dominant left/right direction, not merely many
    # small OD-chord wiggles along a clear signed turn.
    if turn_count >= 3:
        return "multi_turn"
    return "straight"


def curvature_label(total_abs_heading: float, length_m: float) -> str:
    if length_m < 1.0:
        return "smooth"
    curvature = total_abs_heading / max(length_m / 100.0, 1.0)
    if curvature < 0.08:
        return "smooth"
    if curvature < 0.22:
        return "moderate"
    return "winding"


def progress_label(length_m: float, displacement_m: float) -> str:
    ratio = displacement_m / max(length_m, 1.0)
    # Progress is intentionally independent from distance: short trips are
    # already represented by the distance token, so labeling them as "slow"
    # would entangle two instruction fields and corrupt counterfactual control.
    if ratio < 0.45:
        return "slow"
    if ratio > 0.72:
        return "fast"
    return "normal"


def contract_from_local_meters(
    traj_m: np.ndarray,
    reference_m: Optional[np.ndarray],
    dataset: str,
) -> GeometryContractResult:
    traj_m = valid_trajectory(np.asarray(traj_m, dtype=np.float32))
    ref_m = np.asarray(reference_m, dtype=np.float32) if reference_m is not None else np.empty((0, 2), dtype=np.float32)
    ref_m = drop_duplicate_points(ref_m)
    length_m = path_length_m(traj_m)
    displacement_m = float(np.linalg.norm(traj_m[-1] - traj_m[0])) if len(traj_m) >= 2 else 0.0
    raw_net_heading, raw_total_abs_heading = heading_change(traj_m)
    turn_count, net_heading, total_abs_heading = route_shape_stats(traj_m)
    if turn_count == 0 and abs(net_heading) < 1e-6:
        net_heading, total_abs_heading = raw_net_heading, min(raw_total_abs_heading, np.deg2rad(45.0))
    displacement_ratio = displacement_m / max(length_m, 1.0)
    reference_label, mean_ref_dist, ref_violation_rate = reference_metrics_from_local(traj_m, ref_m)
    lateral = reference_lateral_stats(traj_m, ref_m)
    labels = {
        "distance": distance_label(length_m, dataset),
        "direction": direction_label(
            net_heading,
            total_abs_heading,
            turn_count,
            displacement_ratio,
            lateral["mean_signed_lateral_offset_m"],
            lateral["left_side_ratio"],
            lateral["right_side_ratio"],
        ),
        "curvature": curvature_label(total_abs_heading, length_m),
        "progress": progress_label(length_m, displacement_m),
        "reference_adherence": reference_label,
    }
    metrics = {
        "length_m": length_m,
        "displacement_m": displacement_m,
        "displacement_ratio": float(displacement_ratio),
        "net_heading_deg": float(np.rad2deg(net_heading)),
        "total_abs_heading_deg": float(np.rad2deg(total_abs_heading)),
        "turn_count": float(turn_count),
        "raw_total_abs_heading_deg": float(np.rad2deg(raw_total_abs_heading)),
        "mean_reference_distance_m": mean_ref_dist,
        "reference_violation_rate_30m": ref_violation_rate,
        **lateral,
        "valid_points": float(len(traj_m)),
    }
    return GeometryContractResult(labels=labels, metrics=metrics)


def contract_from_lonlat(
    traj_lonlat: np.ndarray,
    reference_lonlat: Optional[np.ndarray],
    dataset: str,
) -> GeometryContractResult:
    traj_lonlat = valid_trajectory(traj_lonlat)
    ref_lonlat = drop_duplicate_points(reference_lonlat) if reference_lonlat is not None else np.empty((0, 2), dtype=np.float32)
    origin = traj_lonlat[0] if len(traj_lonlat) else None
    traj_m = lonlat_to_local_meters(traj_lonlat, origin=origin)
    ref_m = lonlat_to_local_meters(ref_lonlat, origin=origin) if origin is not None and len(ref_lonlat) else ref_lonlat
    return contract_from_local_meters(traj_m, ref_m, dataset)


def contract_from_normalized(
    traj_norm: np.ndarray,
    reference_norm: Optional[np.ndarray],
    route_scale_m: float,
    dataset: str,
) -> GeometryContractResult:
    scale = max(float(route_scale_m), 1e-6)
    traj_m = np.asarray(traj_norm, dtype=np.float32) * scale
    ref_m = None if reference_norm is None else np.asarray(reference_norm, dtype=np.float32) * scale
    return contract_from_local_meters(traj_m, ref_m, dataset)
