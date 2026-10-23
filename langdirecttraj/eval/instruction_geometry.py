from __future__ import annotations

import math
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.instructions.geometry_contract import contract_from_normalized
from langdirecttraj.numerics import safe_atan2
from langdirecttraj.train.control_losses import (
    contract_curvature_score,
    curvature_energy,
    displacement_ratio,
    mean_reference_distance,
    path_length,
    turn_density_score,
)


FIELD_METRIC = {
    "distance": "length_m",
    "curvature": "turn_density_score",
    "progress": "avg_speed_kmh",
    "reference_adherence": "mean_reference_distance_m",
}

DEFAULT_DIRECTION_METRIC = {
    "straight": "local_abs_heading_diff_rad",
    "left_turn": "local_signed_total_turn_rad",
    "right_turn": "local_signed_total_turn_rad",
    "u_turn": "local_abs_heading_diff_rad",
    "multi_turn": "local_total_abs_turn_rad",
}

DIRECTION_RULE_DEFAULTS = {
    "straight_heading_rad": math.pi / 6.0,
    "straight_lateral": 0.08,
    "straight_total_abs_rad": 1.20,
    "turn_margin_rad": 0.35,
    "uturn_longitudinal": -0.05,
    "uturn_heading_rad": 2.0,
    "uturn_max_abs_turn_rad": 1.0,
    "multiturn_total_abs_rad": 4.5,
    "multiturn_count": 3.0,
}

CALIBRATION_METRIC_ALIASES = {
    "progress_ratio": "displacement_ratio",
    "pace_avg_speed_kmh": "avg_speed_kmh",
    "mean_reference_distance": "mean_reference_distance_norm",
    "direction_token_metric": "signed_net_heading_rad",
}


def progress_label_from_speed(avg_speed_kmh: float, thresholds: Mapping[str, float] | None = None) -> str:
    thresholds = thresholds or {}
    slow = float(thresholds.get("slow_max_avg_speed_kmh", 12.0))
    fast = float(thresholds.get("fast_min_avg_speed_kmh", 24.0))
    speed = float(avg_speed_kmh)
    if speed <= slow:
        return "slow"
    if speed >= fast:
        return "fast"
    return "normal"


def add_pace_metrics_to_rows(
    rows: List[Dict[str, float]],
    trip_attributes: Iterable[Mapping[str, float]],
) -> List[Dict[str, float]]:
    """Attach non-leaking pace descriptors to generated-trajectory geometry rows.

    The generator outputs only a spatial trajectory.  For the official Porto
    OD-chord protocol, progress means trip pace estimated from the generated
    path length and the raw trip duration timestamp attribute, not from the GT
    middle trajectory or from reference geometry.
    """
    out: List[Dict[str, float]] = []
    for row, attrs in zip(rows, trip_attributes):
        item = dict(row)
        duration_min = max(float(attrs.get("trip_duration_min", 0.0)), 1e-6)
        length_km = max(float(row.get("length_m", 0.0)) / 1000.0, 0.0)
        avg_speed = float(np.clip(length_km / (duration_min / 60.0), 0.0, 120.0))
        item["trip_duration_min"] = duration_min
        item["generated_total_distance_km"] = length_km
        item["avg_speed_kmh"] = avg_speed
        item["time_per_km_min"] = float(duration_min / max(length_km, 1e-6))
        out.append(item)
    return out


def apply_pace_progress_labels(
    pred_labels: List[Dict[str, str]],
    geom_rows: List[Dict[str, float]],
    trip_attributes: Iterable[Mapping[str, float]],
) -> List[Dict[str, str]]:
    updated: List[Dict[str, str]] = []
    for labels, geom, attrs in zip(pred_labels, geom_rows, trip_attributes):
        thresholds = attrs.get("progress_pace_thresholds", {}) if isinstance(attrs, Mapping) else {}
        new_labels = dict(labels)
        new_labels["progress"] = progress_label_from_speed(float(geom.get("avg_speed_kmh", 0.0)), thresholds)
        updated.append(new_labels)
    return updated


def heading_metrics(points: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Return direction descriptors in radians or normalized local turn ratios."""
    diffs = points[:, 1:] - points[:, :-1]
    if diffs.shape[1] < 2:
        zeros = points.new_zeros(points.shape[0])
        return {
            "signed_net_heading_rad": zeros,
            "abs_signed_heading_rad": zeros,
            "total_abs_heading_rad": zeros,
            "signed_total_turn_rad": zeros,
            "left_turn_ratio": zeros,
            "right_turn_ratio": zeros,
            "max_abs_turn_rad": zeros,
            "turn_count": zeros,
        }
    headings = safe_atan2(diffs[..., 1], diffs[..., 0])
    delta = headings[:, 1:] - headings[:, :-1]
    delta = safe_atan2(torch.sin(delta), torch.cos(delta))
    net = headings[:, -1] - headings[:, 0]
    net = safe_atan2(torch.sin(net), torch.cos(net))
    abs_delta = delta.abs()
    threshold = math.radians(15.0)
    strong = abs_delta > threshold
    strong_count = strong.sum(dim=1).to(dtype=points.dtype).clamp_min(1.0)
    left_ratio = (delta > threshold).sum(dim=1).to(dtype=points.dtype) / strong_count
    right_ratio = (delta < -threshold).sum(dim=1).to(dtype=points.dtype) / strong_count
    return {
        "signed_net_heading_rad": net,
        "abs_signed_heading_rad": net.abs(),
        "total_abs_heading_rad": abs_delta.sum(dim=1),
        "signed_total_turn_rad": delta.sum(dim=1),
        "left_turn_ratio": left_ratio,
        "right_turn_ratio": right_ratio,
        "max_abs_turn_rad": abs_delta.amax(dim=1),
        "turn_count": strong.sum(dim=1).to(dtype=points.dtype),
    }


def reference_lateral_metrics(points: torch.Tensor, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
    if points.shape[1] < 2 or reference.shape[1] < 2:
        zeros = points.new_zeros(points.shape[0])
        return {
            "mean_signed_lateral_offset": zeros,
            "mean_abs_lateral_offset": zeros,
            "left_side_ratio": zeros,
            "right_side_ratio": zeros,
        }
    diffs = torch.zeros_like(reference)
    diffs[:, 1:-1] = reference[:, 2:] - reference[:, :-2]
    diffs[:, 0] = reference[:, 1] - reference[:, 0]
    diffs[:, -1] = reference[:, -1] - reference[:, -2]
    tangent = diffs / torch.linalg.norm(diffs, dim=-1, keepdim=True).clamp_min(1e-8)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    lateral = ((points - reference) * normal).sum(dim=-1)
    threshold = (lateral.abs().median(dim=1).values * 0.25).clamp_min(1e-4)
    return {
        "mean_signed_lateral_offset": lateral.mean(dim=1),
        "mean_abs_lateral_offset": lateral.abs().mean(dim=1),
        "left_side_ratio": (lateral > threshold[:, None]).to(dtype=points.dtype).mean(dim=1),
        "right_side_ratio": (lateral < -threshold[:, None]).to(dtype=points.dtype).mean(dim=1),
    }


def local_direction_metrics(points: torch.Tensor) -> Dict[str, torch.Tensor]:
    diffs = points[:, 1:] - points[:, :-1]
    if diffs.shape[1] < 2:
        zeros = points.new_zeros(points.shape[0])
        return {
            "local_dx": zeros,
            "local_dy": zeros,
            "local_heading_diff_rad": zeros,
            "local_abs_heading_diff_rad": zeros,
            "local_signed_total_turn_rad": zeros,
            "local_total_abs_turn_rad": zeros,
            "local_left_turn_ratio": zeros,
            "local_right_turn_ratio": zeros,
            "local_max_abs_turn_rad": zeros,
            "local_turn_count": zeros,
        }
    first = diffs[:, 0]
    first_norm = torch.linalg.norm(first, dim=-1, keepdim=True).clamp_min(1e-6)
    ex = first / first_norm
    ey = torch.stack([-ex[:, 1], ex[:, 0]], dim=-1)
    rel_end = points[:, -1] - points[:, 0]
    local_dx = (rel_end * ex).sum(dim=-1)
    local_dy = (rel_end * ey).sum(dim=-1)

    headings = safe_atan2(diffs[..., 1], diffs[..., 0])
    delta = headings[:, 1:] - headings[:, :-1]
    delta = safe_atan2(torch.sin(delta), torch.cos(delta))
    heading_diff = headings[:, -1] - headings[:, 0]
    heading_diff = safe_atan2(torch.sin(heading_diff), torch.cos(heading_diff))
    abs_delta = delta.abs()
    threshold = math.radians(15.0)
    strong = abs_delta > threshold
    strong_count = strong.sum(dim=1).to(dtype=points.dtype).clamp_min(1.0)
    return {
        "local_dx": local_dx,
        "local_dy": local_dy,
        "local_heading_diff_rad": heading_diff,
        "local_abs_heading_diff_rad": heading_diff.abs(),
        "local_signed_total_turn_rad": delta.sum(dim=1),
        "local_total_abs_turn_rad": abs_delta.sum(dim=1),
        "local_left_turn_ratio": (delta > threshold).sum(dim=1).to(dtype=points.dtype) / strong_count,
        "local_right_turn_ratio": (delta < -threshold).sum(dim=1).to(dtype=points.dtype) / strong_count,
        "local_max_abs_turn_rad": abs_delta.amax(dim=1),
        "local_turn_count": strong.sum(dim=1).to(dtype=points.dtype),
    }


def contract_direction_metrics(
    points: torch.Tensor,
    route_scales: torch.Tensor,
    coarse_points: int = 16,
) -> Dict[str, torch.Tensor]:
    """Low-frequency OD-chord direction descriptors used by the official label contract."""
    del route_scales
    coarse_points = int(max(coarse_points, 4))
    coarse = F.interpolate(
        points.transpose(1, 2),
        size=coarse_points,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)
    diffs = coarse[:, 1:] - coarse[:, :-1]
    if diffs.shape[1] < 2:
        zeros = points.new_zeros(points.shape[0])
        return {
            "contract_net_heading_rad": zeros,
            "contract_abs_net_heading_rad": zeros,
            "contract_total_abs_heading_rad": zeros,
            "contract_turn_count": zeros,
            "contract_displacement_ratio": zeros,
        }
    headings = safe_atan2(diffs[..., 1], diffs[..., 0])
    delta = headings[:, 1:] - headings[:, :-1]
    delta = safe_atan2(torch.sin(delta), torch.cos(delta))
    abs_delta = delta.abs()
    turn_threshold = math.radians(35.0)
    turn_temp = math.radians(10.0)
    soft_strong = torch.sigmoid((abs_delta - turn_threshold) / max(turn_temp, 1e-6))
    displacement = torch.linalg.norm(points[:, -1] - points[:, 0], dim=-1)
    length = path_length(points).clamp_min(1e-6)
    signed_net = delta.sum(dim=1)
    return {
        "contract_net_heading_rad": signed_net,
        "contract_abs_net_heading_rad": signed_net.abs(),
        "contract_total_abs_heading_rad": (abs_delta * soft_strong).sum(dim=1),
        "contract_raw_total_abs_heading_rad": abs_delta.sum(dim=1),
        "contract_turn_count": soft_strong.sum(dim=1).to(dtype=points.dtype),
        "contract_displacement_ratio": displacement / length,
    }


def geometry_values(
    points: torch.Tensor,
    reference: torch.Tensor,
    route_scales: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute instruction-facing geometry descriptors for normalized decoder coordinates."""
    route_scales = route_scales.view(-1).to(device=points.device, dtype=points.dtype)
    length_norm = path_length(points)
    ref_length_norm = path_length(reference).clamp_min(1e-6)
    heading = heading_metrics(points)
    local_direction = local_direction_metrics(points)
    contract_direction = contract_direction_metrics(points, route_scales)
    lateral = reference_lateral_metrics(points, reference)
    ref_dist_norm = mean_reference_distance(points, reference)
    return {
        "length_norm": length_norm,
        "length_m": length_norm * route_scales,
        "length_ratio": length_norm / ref_length_norm,
        "displacement_ratio": displacement_ratio(points),
        "curvature_energy": curvature_energy(points),
        "contract_curvature_score": contract_curvature_score(points, route_scales),
        "turn_density_score": turn_density_score(points, route_scales),
        **heading,
        **local_direction,
        **contract_direction,
        **lateral,
        "mean_reference_distance_norm": ref_dist_norm,
        "mean_reference_distance_m": ref_dist_norm * route_scales,
    }


def _positive_part(value: float) -> float:
    return max(float(value), 0.0)


def direction_rule_scores(geom: Mapping[str, float]) -> Dict[str, float]:
    """Composite local-frame direction scores; lower means more compatible.

    Direction labels are not well represented by one unrelated scalar per
    token.  These scores use the same local-frame cues that the direction
    diagnostic scripts already expose: endpoint heading, signed turn, lateral
    drift, turn count, and total absolute turn.
    """
    cfg = DIRECTION_RULE_DEFAULTS
    signed_heading = float(geom.get("local_heading_diff_rad", geom.get("signed_net_heading_rad", 0.0)))
    abs_heading = abs(float(geom.get("local_abs_heading_diff_rad", abs(signed_heading))))
    signed_turn = float(geom.get("local_signed_total_turn_rad", geom.get("signed_total_turn_rad", 0.0)))
    total_abs = float(geom.get("local_total_abs_turn_rad", geom.get("total_abs_heading_rad", 0.0)))
    max_abs_turn = float(geom.get("local_max_abs_turn_rad", geom.get("max_abs_turn_rad", 0.0)))
    turn_count = float(geom.get("local_turn_count", geom.get("turn_count", 0.0)))
    local_dx = float(geom.get("local_dx", 0.0))
    local_dy = float(geom.get("local_dy", geom.get("mean_signed_lateral_offset", 0.0)))
    left_ratio = float(geom.get("local_left_turn_ratio", geom.get("left_turn_ratio", 0.0)))
    right_ratio = float(geom.get("local_right_turn_ratio", geom.get("right_turn_ratio", 0.0)))

    straight_heading = float(cfg["straight_heading_rad"])
    straight_lateral = float(cfg["straight_lateral"])
    straight_total_abs = float(cfg["straight_total_abs_rad"])
    turn_margin = float(cfg["turn_margin_rad"])
    uturn_heading = float(cfg["uturn_heading_rad"])
    uturn_longitudinal = float(cfg["uturn_longitudinal"])
    uturn_max_abs = float(cfg["uturn_max_abs_turn_rad"])
    multiturn_abs = float(cfg["multiturn_total_abs_rad"])
    multiturn_count = float(cfg["multiturn_count"])

    left_strength = max(local_dy / straight_lateral, signed_turn / turn_margin, left_ratio)
    right_strength = max(-local_dy / straight_lateral, -signed_turn / turn_margin, right_ratio)
    uturn_strength = max(abs_heading / uturn_heading, max_abs_turn / uturn_max_abs)
    if local_dx < uturn_longitudinal:
        uturn_strength = max(uturn_strength, 1.0 + (uturn_longitudinal - local_dx) / 0.20)
    multiturn_strength = max(total_abs / multiturn_abs, turn_count / multiturn_count)
    left_clear = max(0.0, min(left_strength, 1.0))
    right_clear = max(0.0, min(right_strength, 1.0))
    directional_multiturn_penalty = _positive_part(max(left_strength, right_strength) - 1.0)

    return {
        "straight": (
            abs_heading / max(straight_heading, 1e-6)
            + abs(local_dy) / max(straight_lateral, 1e-6)
            + _positive_part(total_abs - straight_total_abs) / max(straight_total_abs, 1e-6)
            + _positive_part(turn_count - 1.0)
            + 0.03 * total_abs / max(multiturn_abs, 1e-6)
        ),
        "left_turn": (
            _positive_part(1.0 - left_strength)
            + 0.35 * _positive_part(multiturn_strength - 1.0) * (1.0 - 0.70 * left_clear)
            + 0.25 * _positive_part(uturn_strength - 1.0)
            - 0.03 * signed_turn / max(turn_margin, 1e-6)
            - 0.02 * local_dy / max(straight_lateral, 1e-6)
            - 0.02 * (left_ratio - right_ratio)
        ),
        "right_turn": (
            _positive_part(1.0 - right_strength)
            + 0.35 * _positive_part(multiturn_strength - 1.0) * (1.0 - 0.70 * right_clear)
            + 0.25 * _positive_part(uturn_strength - 1.0)
            + 0.03 * signed_turn / max(turn_margin, 1e-6)
            + 0.02 * local_dy / max(straight_lateral, 1e-6)
            - 0.02 * (right_ratio - left_ratio)
        ),
        "u_turn": (
            _positive_part(1.0 - uturn_strength)
            + 0.25 * _positive_part(multiturn_strength - 1.0)
            - 0.03 * uturn_strength
        ),
        "multi_turn": (
            _positive_part(1.0 - multiturn_strength)
            + 0.20 * _positive_part(uturn_strength - 1.0)
            + 0.45 * directional_multiturn_penalty
            - 0.03 * multiturn_strength
        ),
    }


def contract_direction_rule_scores(geom: Mapping[str, float]) -> Dict[str, float]:
    """Official OD-chord direction compatibility scores; lower is better."""
    net = float(geom.get("contract_net_heading_rad", geom.get("signed_net_heading_rad", 0.0)))
    abs_net = abs(float(geom.get("contract_abs_net_heading_rad", abs(net))))
    total_abs = float(geom.get("contract_total_abs_heading_rad", geom.get("total_abs_heading_rad", 0.0)))
    turn_count = float(geom.get("contract_turn_count", geom.get("turn_count", 0.0)))
    displacement_ratio = float(geom.get("contract_displacement_ratio", geom.get("displacement_ratio", 0.0)))
    signed_turn = float(geom.get("local_signed_total_turn_rad", geom.get("signed_total_turn_rad", net)))
    local_dy = float(geom.get("local_dy", 0.0))
    left_ratio = float(geom.get("local_left_turn_ratio", geom.get("left_turn_ratio", 0.0)))
    right_ratio = float(geom.get("local_right_turn_ratio", geom.get("right_turn_ratio", 0.0)))

    deg35 = math.radians(35.0)
    deg135 = math.radians(135.0)
    deg120 = math.radians(120.0)
    eps = 1e-6
    u_heading_strength = min(
        abs_net / max(deg135, eps),
        _positive_part(0.55 - displacement_ratio) / 0.55,
    )
    u_loop_strength = min(
        _positive_part(0.25 - displacement_ratio) / 0.25,
        total_abs / max(deg120, eps),
    )
    u_strength = max(u_heading_strength, u_loop_strength)
    multi_strength = max(
        turn_count / 3.0,
        total_abs / max(deg120, eps),
    )
    left_strength = max(
        _positive_part(net) / max(deg35, eps),
        _positive_part(signed_turn) / max(deg35, eps),
        0.35 * _positive_part(local_dy - 0.02) / 0.12,
        left_ratio,
    )
    right_strength = max(
        _positive_part(-net) / max(deg35, eps),
        _positive_part(-signed_turn) / max(deg35, eps),
        0.35 * _positive_part(-local_dy - 0.02) / 0.12,
        right_ratio,
    )
    left_clear = max(0.0, min(left_strength, 1.0))
    right_clear = max(0.0, min(right_strength, 1.0))
    directional_strength = max(left_strength, right_strength)
    return {
        "straight": (
            abs_net / deg35
            + 0.4 * total_abs / max(deg120, eps)
            + _positive_part(turn_count - 1.0)
            + 0.4 * max(left_strength, right_strength)
        ),
        "left_turn": (
            _positive_part(1.0 - left_strength)
            + 0.5 * _positive_part(multi_strength - 1.0) * (1.0 - 0.75 * left_clear)
            + 0.5 * _positive_part(u_strength - 1.0)
            - 0.08 * left_strength
        ),
        "right_turn": (
            _positive_part(1.0 - right_strength)
            + 0.5 * _positive_part(multi_strength - 1.0) * (1.0 - 0.75 * right_clear)
            + 0.5 * _positive_part(u_strength - 1.0)
            - 0.08 * right_strength
        ),
        "u_turn": (
            _positive_part(1.0 - u_strength)
            + 0.35 * _positive_part(multi_strength - 1.0)
            - 0.08 * u_strength
        ),
        "multi_turn": (
            _positive_part(1.0 - multi_strength)
            + 0.35 * _positive_part(u_strength - 1.0)
            + 0.25 * multi_strength * min(directional_strength, 1.0)
            + 0.75 * _positive_part(directional_strength - 1.0)
            - 0.08 * multi_strength
        ),
    }


def predict_direction_composite(geom: Mapping[str, float]) -> str:
    """Predict a direction token from local-frame composite geometry cues."""
    cfg = DIRECTION_RULE_DEFAULTS
    signed_heading = float(geom.get("local_heading_diff_rad", geom.get("signed_net_heading_rad", 0.0)))
    abs_heading = abs(float(geom.get("local_abs_heading_diff_rad", abs(signed_heading))))
    total_abs = float(geom.get("local_total_abs_turn_rad", geom.get("total_abs_heading_rad", 0.0)))
    max_abs_turn = float(geom.get("local_max_abs_turn_rad", geom.get("max_abs_turn_rad", 0.0)))
    turn_count = float(geom.get("local_turn_count", geom.get("turn_count", 0.0)))
    local_dx = float(geom.get("local_dx", 0.0))
    local_dy = float(geom.get("local_dy", geom.get("mean_signed_lateral_offset", 0.0)))
    scores = direction_rule_scores(geom)

    if (
        local_dx < float(cfg["uturn_longitudinal"])
        and abs_heading >= float(cfg["uturn_heading_rad"])
    ) or (
        abs_heading >= float(cfg["uturn_heading_rad"])
        and max_abs_turn >= float(cfg["uturn_max_abs_turn_rad"])
    ):
        return "u_turn"
    if (
        turn_count >= float(cfg["multiturn_count"])
        or total_abs >= float(cfg["multiturn_total_abs_rad"])
    ) and scores["multi_turn"] <= min(scores["left_turn"], scores["right_turn"], scores["straight"]):
        return "multi_turn"
    if (
        abs_heading < 0.12
        and abs(local_dy) < 0.015
        and scores["straight"] <= min(scores["left_turn"], scores["right_turn"])
    ):
        return "straight"
    return min(scores.keys(), key=lambda token: scores[token])


def _robust_center_scale(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return 0.0, 1.0
    center = float(np.percentile(arr, 50))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr = q75 - q25
    std = float(arr.std())
    scale = max(iqr, std, 1e-6)
    return center, scale


@dataclass
class TokenStat:
    metric: str
    center: float
    scale: float
    support: int = 0


@dataclass
class GeometryTokenCalibrator:
    stats: Dict[str, Dict[str, TokenStat]] = field(default_factory=dict)

    @classmethod
    def from_target_batches(cls, rows: List[tuple[Dict[str, str], Dict[str, float]]]) -> "GeometryTokenCalibrator":
        grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for labels, geom in rows:
            for field_name in DEFAULT_MOBILITY_VOCAB.fields:
                token = str(labels.get(field_name, "unknown"))
                metric_name = cls.metric_for_token(field_name, token)
                if metric_name not in geom:
                    continue
                grouped[field_name][token][metric_name].append(float(geom[metric_name]))

        stats: Dict[str, Dict[str, TokenStat]] = {}
        for field_name in DEFAULT_MOBILITY_VOCAB.fields:
            stats[field_name] = {}
            for token in DEFAULT_MOBILITY_VOCAB.token_to_id[field_name]:
                metric_name = cls.metric_for_token(field_name, token)
                center, scale = _robust_center_scale(grouped[field_name][token][metric_name])
                support = len(grouped[field_name][token][metric_name])
                stats[field_name][token] = TokenStat(
                    metric=metric_name,
                    center=center,
                    scale=scale,
                    support=support,
                )
        return cls(stats=stats)

    @classmethod
    def from_json(cls, path: str) -> "GeometryTokenCalibrator":
        payload = json.loads(open(path, "r", encoding="utf-8").read())
        raw_fields = payload.get("fields", {})
        stats: Dict[str, Dict[str, TokenStat]] = {}
        for field_name in DEFAULT_MOBILITY_VOCAB.fields:
            field_payload = raw_fields.get(field_name, {})
            default_metric = CALIBRATION_METRIC_ALIASES.get(
                str(field_payload.get("metric", cls.metric_for_token(field_name, ""))),
                str(field_payload.get("metric", cls.metric_for_token(field_name, ""))),
            )
            stats[field_name] = {}
            for token in DEFAULT_MOBILITY_VOCAB.token_to_id[field_name]:
                token_payload = field_payload.get("tokens", {}).get(token, {})
                metric_name = token_payload.get("direction_metric", default_metric)
                metric_name = CALIBRATION_METRIC_ALIASES.get(str(metric_name), str(metric_name))
                center = float(token_payload.get("p50", token_payload.get("mean", 0.0)))
                scale = float(token_payload.get("scale", token_payload.get("iqr", token_payload.get("std", 1.0))))
                if not math.isfinite(center):
                    center = 0.0
                if not math.isfinite(scale) or scale <= 0:
                    scale = 1.0
                stats[field_name][token] = TokenStat(
                    metric=metric_name,
                    center=center,
                    scale=max(scale, 1e-6),
                    support=int(token_payload.get("count", 0)),
                )
        return cls(stats=stats)

    @staticmethod
    def metric_for_token(field_name: str, token: str) -> str:
        if field_name == "direction":
            return DEFAULT_DIRECTION_METRIC.get(str(token), "signed_net_heading_rad")
        return FIELD_METRIC[field_name]

    def score_token(self, field_name: str, token: str, geom: Mapping[str, float]) -> float:
        if field_name == "direction":
            return contract_direction_rule_scores(geom).get(str(token), float("inf"))
        stat = self.stats[field_name][token]
        value = float(geom.get(stat.metric, 0.0))
        return abs(value - stat.center) / max(stat.scale, 1e-6)

    def predict_field(self, field_name: str, geom: Mapping[str, float]) -> str:
        if field_name == "direction":
            scores = contract_direction_rule_scores(geom)
            return min(scores.keys(), key=lambda token: scores[token])
        tokens = DEFAULT_MOBILITY_VOCAB.token_to_id[field_name]
        return min(tokens.keys(), key=lambda token: self.score_token(field_name, token, geom))

    def predict_labels(self, geom: Mapping[str, float]) -> Dict[str, str]:
        return {field_name: self.predict_field(field_name, geom) for field_name in DEFAULT_MOBILITY_VOCAB.fields}

    def center_delta(self, field_name: str, source_token: str, target_token: str) -> tuple[str, float]:
        source = self.stats[field_name][source_token]
        target = self.stats[field_name][target_token]
        if source.metric != target.metric:
            return target.metric, 0.0
        return target.metric, target.center - source.center

    def transition_metric_delta(
        self,
        field_name: str,
        source_token: str,
        target_token: str,
        base_geom: Mapping[str, float],
        cf_geom: Mapping[str, float],
    ) -> tuple[str, float, float, float, float]:
        """Return a Table-4-aligned signed movement for a token edit.

        For scalar fields, movement is measured in the target token's calibrated
        metric and signed toward the target center from the actual base output.
        Direction is different: its official contract is a composite local-frame
        classifier rather than one shared scalar.  For direction edits, use the
        target composite-score improvement itself as the signed movement.  This
        keeps diagnostics aligned with the iMotion-style direction rules used by
        Table 3 and avoids meaningless cross-metric deltas such as
        multi_turn(local_turn_count) -> right_turn(local_signed_total_turn_rad).
        """
        if field_name == "direction":
            base_score = self.score_token(field_name, target_token, base_geom)
            cf_score = self.score_token(field_name, target_token, cf_geom)
            signed_delta = float(base_score - cf_score)
            return "contract_direction_score", 0.0, signed_delta, signed_delta, signed_delta

        metric_name, expected_center_delta = self.center_delta(field_name, source_token, target_token)
        target_stat = self.stats[field_name][target_token]
        observed_delta = float(cf_geom.get(metric_name, 0.0) - base_geom.get(metric_name, 0.0))
        base_to_target_delta = float(target_stat.center) - float(base_geom.get(metric_name, 0.0))
        if abs(base_to_target_delta) < 1e-8:
            base_to_target_delta = float(expected_center_delta)
        signed_delta = observed_delta if base_to_target_delta >= 0 else -observed_delta
        signed_delta_z = signed_delta / max(float(target_stat.scale), 1e-6)
        return metric_name, base_to_target_delta, observed_delta, signed_delta, signed_delta_z

    def normalized_metric_delta(
        self,
        field_name: str,
        token: str,
        base_geom: Mapping[str, float],
        cf_geom: Mapping[str, float],
    ) -> float:
        if field_name == "direction":
            return abs(self.score_token(field_name, token, cf_geom) - self.score_token(field_name, token, base_geom))
        stat = self.stats[field_name][token]
        return abs(float(cf_geom.get(stat.metric, 0.0)) - float(base_geom.get(stat.metric, 0.0))) / max(stat.scale, 1e-6)

    def as_dict(self) -> Dict:
        return {
            field_name: {
                token: {
                    "metric": stat.metric,
                    "center": stat.center,
                    "scale": stat.scale,
                    "support": stat.support,
                }
                for token, stat in token_stats.items()
            }
            for field_name, token_stats in self.stats.items()
        }


def tensor_dict_to_rows(geom: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
    size = next(iter(geom.values())).shape[0]
    rows: List[Dict[str, float]] = []
    detached = {key: value.detach().cpu().float().tolist() for key, value in geom.items()}
    for idx in range(size):
        rows.append({key: float(values[idx]) for key, values in detached.items()})
    return rows


def contract_labels_from_tensors(
    points: torch.Tensor,
    reference: torch.Tensor,
    route_scales: torch.Tensor,
    dataset: str,
) -> List[Dict[str, str]]:
    points_np = points.detach().cpu().float().numpy()
    reference_np = reference.detach().cpu().float().numpy()
    scales_np = route_scales.detach().cpu().float().view(-1).numpy()
    labels: List[Dict[str, str]] = []
    for idx in range(points_np.shape[0]):
        contract = contract_from_normalized(
            points_np[idx],
            reference_np[idx],
            float(scales_np[idx]),
            dataset=dataset,
        )
        labels.append(contract.labels)
    return labels


def macro_f1(
    y_true: List[str],
    y_pred: List[str],
    labels: Iterable[str],
    include_empty: bool = False,
) -> Dict:
    per_label = {}
    f1_values = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        if include_empty or support > 0:
            f1_values.append(f1)
        per_label[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
            "predicted": int(sum(1 for p in y_pred if p == label)),
        }
    return {
        "macro_f1": float(np.mean(f1_values)) if f1_values else float("nan"),
        "accuracy": float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else float("nan"),
        "per_label": per_label,
    }
