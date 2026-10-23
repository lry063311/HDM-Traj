from __future__ import annotations

import math
import json
from pathlib import Path

from typing import Dict, Iterable, List, Literal

import torch
import torch.nn.functional as F

from langdirecttraj.counterfactual_policy import (
    STRUCTURAL_COLLATERAL_EXCLUSIONS,
    independent_collateral_field_names,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.models.anchor_residual_decoder import AnchorResidualDecoder
from langdirecttraj.numerics import safe_atan2


_FIELD_INDEX = {field: idx for idx, field in enumerate(DEFAULT_MOBILITY_VOCAB.fields)}
_INDEPENDENT_COLLATERAL_FIELD_CACHE = {
    field: independent_collateral_field_names(field) for field in DEFAULT_MOBILITY_VOCAB.fields
}


def independent_collateral_fields(changed_field: str) -> List[str]:
    """Fields used by Table-4 independent collateral for a changed token."""
    cached = _INDEPENDENT_COLLATERAL_FIELD_CACHE.get(changed_field)
    if cached is None:
        return independent_collateral_field_names(changed_field)
    return list(cached)


def collateral_fields_for_mode(changed_field: str, mode: str) -> List[str]:
    """Return non-target fields for independent or raw collateral training."""
    field_list = list(DEFAULT_MOBILITY_VOCAB.fields)
    if str(mode) == "raw":
        return [field for field in field_list if field != changed_field]
    if str(mode) != "independent":
        raise ValueError(f"Unknown collateral field mode: {mode}")
    return independent_collateral_field_names(changed_field, fields=field_list)


def path_length(points: torch.Tensor) -> torch.Tensor:
    steps = torch.linalg.norm(points[:, 1:] - points[:, :-1], dim=-1)
    return steps.sum(dim=1).clamp_min(1e-6)


def displacement_ratio(points: torch.Tensor) -> torch.Tensor:
    displacement = torch.linalg.norm(points[:, -1] - points[:, 0], dim=-1)
    return displacement / path_length(points)


def heading_change(points: torch.Tensor) -> torch.Tensor:
    diffs = points[:, 1:] - points[:, :-1]
    if diffs.shape[1] < 2:
        return points.new_zeros(points.shape[0])
    first = diffs[:, : max(1, diffs.shape[1] // 5)].mean(dim=1)
    last = diffs[:, -max(1, diffs.shape[1] // 5) :].mean(dim=1)
    cross = first[:, 0] * last[:, 1] - first[:, 1] * last[:, 0]
    dot = (first * last).sum(dim=-1)
    return safe_atan2(cross, dot.clamp_min(-1e6))


def curvature_energy(points: torch.Tensor) -> torch.Tensor:
    diffs = points[:, 1:] - points[:, :-1]
    if diffs.shape[1] < 2:
        return points.new_zeros(points.shape[0])
    v1 = diffs[:, :-1]
    v2 = diffs[:, 1:]
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = (v1 * v2).sum(dim=-1)
    angle = safe_atan2(cross, dot)
    return angle.abs().mean(dim=1)


def build_counterfactual_edit_masks(
    base_tokens: torch.Tensor,
    changed_tokens: torch.Tensor,
    changed_field: str,
    target_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    """Heuristic local edit masks for generator-side counterfactual training.

    The hard Table-4 cases are local structure-creation edits rather than
    global perturbations. This helper mirrors that assumption with a central
    edit span plus a lightweight tangent/normal slot emphasis.
    """
    batch = int(changed_tokens.shape[0])
    device = changed_tokens.device
    dtype = torch.float32
    t = torch.linspace(0.0, 1.0, target_len, device=device, dtype=dtype).view(1, target_len)
    mid = torch.sin(math.pi * t).clamp_min(0.0)
    narrow = torch.exp(-0.5 * ((t - 0.5) / 0.18).pow(2))
    broad = torch.exp(-0.5 * ((t - 0.5) / 0.28).pow(2))
    wide = torch.exp(-0.5 * ((t - 0.5) / 0.36).pow(2))
    multiturn_window = (0.65 * broad + 0.35 * torch.sin(2.0 * math.pi * t).pow(2)) * mid
    winding_window = (0.55 * broad + 0.45 * torch.sin(3.0 * math.pi * t).pow(2)) * mid
    default_window = (0.70 * narrow + 0.30 * mid).clamp(0.0, 1.0)

    field_idx = DEFAULT_MOBILITY_VOCAB.fields.index(changed_field)
    vocab = DEFAULT_MOBILITY_VOCAB.id_to_token[changed_field]
    source_ids = base_tokens[:, field_idx].to(device=device, dtype=torch.long)
    target_ids = changed_tokens[:, field_idx].to(device=device, dtype=torch.long)

    edit_mask = default_window.expand(batch, -1).clone()
    slot_mask = torch.stack(
        [0.25 * edit_mask, 0.85 * edit_mask],
        dim=-1,
    )
    proposal_families: List[str] = []
    mask_families: List[str] = []

    direction_vocab = DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]
    curvature_vocab = DEFAULT_MOBILITY_VOCAB.token_to_id["curvature"]

    for row in range(batch):
        source_name = vocab[int(source_ids[row].item())]
        target_name = vocab[int(target_ids[row].item())]
        family = f"{changed_field}:{source_name}->{target_name}"
        mask_family = "generic_local_edit"
        row_window = default_window[0]
        tangent_weight = 0.25
        normal_weight = 0.85
        if changed_field == "direction":
            target_id = int(target_ids[row].item())
            source_id = int(source_ids[row].item())
            if target_id == int(direction_vocab["multi_turn"]) and source_id != int(direction_vocab["multi_turn"]):
                row_window = multiturn_window[0]
                tangent_weight = 0.18
                normal_weight = 1.00
                family = (
                    "direction:straight->multi_turn_local"
                    if source_id == int(direction_vocab["straight"])
                    else "direction:turn->multi_turn_local"
                )
                mask_family = "multi_turn_creation_mask"
            elif {
                source_id,
                target_id,
            } == {int(direction_vocab["left_turn"]), int(direction_vocab["right_turn"])}:
                row_window = (0.80 * narrow + 0.20 * mid)[0]
                tangent_weight = 0.22
                normal_weight = 0.95
                family = "direction:reversal_local"
                mask_family = "reversal_mask"
            else:
                family = f"direction:{source_name}->{target_name}_local"
                mask_family = "direction_local_mask"
        elif changed_field == "curvature":
            target_id = int(target_ids[row].item())
            source_id = int(source_ids[row].item())
            if target_id == int(curvature_vocab["winding"]):
                row_window = winding_window[0]
                tangent_weight = 0.10
                normal_weight = 1.00
                family = (
                    "curvature:smooth->winding_local"
                    if source_id == int(curvature_vocab["smooth"])
                    else "curvature:to_winding_local"
                )
                mask_family = "winding_burst_mask"
            elif target_id == int(curvature_vocab["smooth"]):
                row_window = (0.60 * broad + 0.40 * wide)[0] * mid[0]
                tangent_weight = 0.10
                normal_weight = 0.70
                family = f"curvature:{source_name}->smooth_local"
                mask_family = "curvature_suppression_mask"
            else:
                family = f"curvature:{source_name}->{target_name}_local"
                mask_family = "curvature_local_mask"
        elif changed_field == "reference_adherence":
            row_window = (0.65 * broad + 0.35 * narrow)[0] * mid[0]
            tangent_weight = 0.08
            normal_weight = 0.90
            family = f"reference:{source_name}->{target_name}_local"
            mask_family = "reference_local_mask"
        proposal_families.append(family)
        mask_families.append(mask_family)
        edit_mask[row] = row_window.clamp(0.0, 1.0)
        slot_mask[row, :, 0] = tangent_weight * edit_mask[row]
        slot_mask[row, :, 1] = normal_weight * edit_mask[row]

    preserve_mask = (1.0 - edit_mask).clamp(0.0, 1.0)
    edge_guard = ((t < 0.08) | (t > 0.92)).expand(batch, -1)
    preserve_mask = torch.where(edge_guard, torch.ones_like(preserve_mask), preserve_mask)
    edit_mask = torch.where(edge_guard, torch.zeros_like(edit_mask), edit_mask)
    denom = slot_mask.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
    slot_mask = slot_mask / denom
    return edit_mask, preserve_mask, slot_mask, proposal_families, mask_families


def contract_curvature_score(points: torch.Tensor, route_scales: torch.Tensor | None = None) -> torch.Tensor:
    """Length-normalized strong-turn curvature used for instruction control.

    This mirrors the label contract more closely than per-step mean curvature:
    count only strong local turns, then normalize by physical route length per
    100 meters when route scales are available.
    """

    diffs = points[:, 1:] - points[:, :-1]
    if diffs.shape[1] < 2:
        return points.new_zeros(points.shape[0])
    v1 = diffs[:, :-1]
    v2 = diffs[:, 1:]
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = (v1 * v2).sum(dim=-1)
    angle = safe_atan2(cross, dot)
    abs_angle = angle.abs()
    strong = abs_angle > math.radians(35.0)
    strong_total = (abs_angle * strong.to(dtype=points.dtype)).sum(dim=1)
    length_norm = path_length(points)
    if route_scales is None:
        return strong_total / length_norm.clamp_min(1e-6)
    route_scales = route_scales.view(-1).to(device=points.device, dtype=points.dtype)
    length_m = length_norm * route_scales
    return strong_total / torch.clamp(length_m / 100.0, min=1.0)


def turn_density_score(points: torch.Tensor, route_scales: torch.Tensor | None = None) -> torch.Tensor:
    """Continuous total-turn density per 100 meters for curvature control."""

    diffs = points[:, 1:] - points[:, :-1]
    if diffs.shape[1] < 2:
        return points.new_zeros(points.shape[0])
    v1 = diffs[:, :-1]
    v2 = diffs[:, 1:]
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    dot = (v1 * v2).sum(dim=-1)
    total_abs = safe_atan2(cross, dot).abs().sum(dim=1)
    length_norm = path_length(points)
    if route_scales is None:
        return total_abs / length_norm.clamp_min(1e-6)
    route_scales = route_scales.view(-1).to(device=points.device, dtype=points.dtype)
    length_m = length_norm * route_scales
    return total_abs / torch.clamp(length_m / 100.0, min=1.0)


def point_segment_distance(points: torch.Tensor, seg_start: torch.Tensor, seg_end: torch.Tensor) -> torch.Tensor:
    points_exp = points[:, :, None, :]
    start = seg_start[:, None, :, :]
    end = seg_end[:, None, :, :]
    seg = end - start
    denom = (seg * seg).sum(dim=-1).clamp_min(1e-8)
    t = ((points_exp - start) * seg).sum(dim=-1) / denom
    t = t.clamp(0.0, 1.0)
    proj = start + t.unsqueeze(-1) * seg
    return torch.linalg.norm(points_exp - proj, dim=-1).min(dim=2).values


def mean_reference_distance(points: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return point_segment_distance(points, reference[:, :-1], reference[:, 1:]).mean(dim=1)


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(dtype=pred.dtype)
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    denom = mask_f.sum().clamp_min(1.0) * pred.shape[-1]
    return (loss * mask_f).sum() / denom


def velocity_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    pred_vel = pred[:, 1:] - pred[:, :-1]
    target_vel = target[:, 1:] - target[:, :-1]
    valid = mask[:, 1:] & mask[:, :-1]
    return masked_smooth_l1(pred_vel, target_vel, valid)


def coarse_skeleton_loss(pred: torch.Tensor, target: torch.Tensor, coarse_points: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    del mask
    coarse_points = int(coarse_points)
    if coarse_points <= 1:
        return pred.new_tensor(0.0)
    pred_coarse = _resample_like(pred, coarse_points)
    target_coarse = _resample_like(target, coarse_points)
    coarse_mask = torch.ones(pred.shape[0], coarse_points, dtype=torch.bool, device=pred.device)
    return masked_smooth_l1(pred_coarse, target_coarse, coarse_mask)


def length_ratio_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_len = path_length(pred)
    target_len = path_length(target)
    target_safe = target_len.clamp_min(1e-6)
    ratio = pred_len / target_safe
    loss_abs = F.smooth_l1_loss(pred_len, target_len)
    loss_ratio = F.smooth_l1_loss(ratio, torch.ones_like(ratio))
    loss_under = F.relu(0.90 - ratio).mean()
    return 0.5 * loss_abs + 0.5 * loss_ratio + 2.0 * loss_under


def candidate_length_coverage_loss(
    candidates: torch.Tensor,
    target: torch.Tensor,
    min_ratio: float = 0.85,
    temperature: float = 0.08,
    over_ratio: float = 1.40,
    over_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Require the K-mode pool to contain at least one route with plausible length.

    This is the candidate-pool version of AnchorTraj's under-length hinge. The
    selected/oracle loss can still let every candidate collapse toward the OD
    chord; here the soft-best ratio makes the pool keep at least one non-chord
    route without forcing every mode to be long.
    """
    if candidates.ndim != 4 or candidates.shape[1] <= 1:
        zero = candidates.new_tensor(0.0)
        return zero, {
            "candidate_length_coverage": 0.0,
            "candidate_length_best_ratio": 0.0,
            "candidate_length_mean_ratio": 0.0,
            "candidate_length_over": 0.0,
        }
    batch, num_modes, _, _ = candidates.shape
    flat = candidates.reshape(batch * num_modes, candidates.shape[-2], candidates.shape[-1])
    cand_len = path_length(flat).view(batch, num_modes)
    target_len = path_length(target).view(batch, 1).clamp_min(1e-6)
    ratio = cand_len / target_len
    temp = max(float(temperature), 1e-6)
    weights = torch.softmax(ratio / temp, dim=1)
    soft_best = (weights * ratio).sum(dim=1)
    under = F.relu(float(min_ratio) - soft_best).pow(2).mean()
    over = F.relu(ratio - float(over_ratio)).pow(2).mean()
    loss = under + float(over_weight) * over
    stats = {
        "candidate_length_coverage": float(loss.detach().item()),
        "candidate_length_best_ratio": float(ratio.max(dim=1).values.detach().mean().item()),
        "candidate_length_mean_ratio": float(ratio.detach().mean().item()),
        "candidate_length_over": float(over.detach().item()),
    }
    return loss, stats


def load_global_transition_prior(path: str | Path | None, device: torch.device | None = None) -> dict | None:
    """Load aggregate city transition statistics for differentiable route regularization."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Global transition prior not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    grid_size = int(payload["grid_size"])
    num_cells = int(payload.get("num_cells", grid_size * grid_size))
    default_nll = float(payload.get("default_nll", 20.0))
    matrix = torch.full((num_cells, num_cells), default_nll, dtype=torch.float32, device=device)
    for key, item in dict(payload.get("transitions", {})).items():
        try:
            src_s, dst_s = str(key).split(":", 1)
            src, dst = int(src_s), int(dst_s)
        except ValueError:
            continue
        if 0 <= src < num_cells and 0 <= dst < num_cells:
            matrix[src, dst] = float(item["nll"] if isinstance(item, dict) else item)
    bounds = payload["bounds"]
    lon_bound = torch.tensor(bounds["lon_bound"], dtype=torch.float32, device=device)
    lat_bound = torch.tensor(bounds["lat_bound"], dtype=torch.float32, device=device)
    out = {
        "path": str(path),
        "grid_size": grid_size,
        "num_cells": num_cells,
        "default_nll": default_nll,
        "transition_nll": matrix,
        "lon_bound": lon_bound,
        "lat_bound": lat_bound,
        "num_unique_transitions": int(payload.get("num_unique_transitions", 0)),
        "total_transitions": int(payload.get("total_transitions", 0)),
    }
    od_payload = payload.get("od_cell_prior")
    if isinstance(od_payload, dict):
        global_cell_default = float(payload.get("cell_default_nll", default_nll))
        groups = dict(od_payload.get("groups", {}))
        bearing_bins = int(od_payload.get("bearing_bins", 1))
        distance_bins = int(od_payload.get("distance_bins", 1))
        cell_group_nll = torch.full(
            (max(1, bearing_bins * distance_bins), num_cells),
            global_cell_default,
            dtype=torch.float32,
            device=device,
        )
        group_default_nll = torch.full(
            (max(1, bearing_bins * distance_bins),),
            global_cell_default,
            dtype=torch.float32,
            device=device,
        )
        for sector in range(max(1, bearing_bins)):
            for dbin in range(max(1, distance_bins)):
                group_idx = sector * max(1, distance_bins) + dbin
                group = groups.get(f"s{sector}_d{dbin}", {})
                group_default_nll[group_idx] = float(group.get("default_nll", global_cell_default))
                for cell_key, item in dict(group.get("cells", {})).items():
                    try:
                        cell_id = int(cell_key)
                    except ValueError:
                        continue
                    if 0 <= cell_id < num_cells:
                        cell_group_nll[group_idx, cell_id] = float(
                            item["nll"] if isinstance(item, dict) else item
                        )
        out["od_cell_prior"] = {
            "bearing_bins": max(1, bearing_bins),
            "distance_bins": max(1, distance_bins),
            "distance_thresholds_m": [
                float(value) for value in od_payload.get("distance_thresholds_m", [])
            ],
            "cell_group_nll": cell_group_nll,
            "group_default_nll": group_default_nll,
        }
    length_payload = payload.get("od_length_prior")
    if isinstance(length_payload, dict):
        bearing_bins = int(length_payload.get("bearing_bins", 1))
        distance_bins = int(length_payload.get("distance_bins", 1))
        group_count = max(1, bearing_bins * distance_bins)
        global_stats = dict(length_payload.get("global", {}))
        global_p10 = float(global_stats.get("p10", 1.0))
        global_p50 = float(global_stats.get("p50", 1.0))
        global_p90 = float(global_stats.get("p90", max(global_p50, 1.0)))
        p10 = torch.full((group_count,), global_p10, dtype=torch.float32, device=device)
        p50 = torch.full((group_count,), global_p50, dtype=torch.float32, device=device)
        p90 = torch.full((group_count,), global_p90, dtype=torch.float32, device=device)
        for sector in range(max(1, bearing_bins)):
            for dbin in range(max(1, distance_bins)):
                group_idx = sector * max(1, distance_bins) + dbin
                group = dict(length_payload.get("groups", {}).get(f"s{sector}_d{dbin}", {}))
                p10[group_idx] = float(group.get("p10", global_p10))
                p50[group_idx] = float(group.get("p50", global_p50))
                p90[group_idx] = float(group.get("p90", global_p90))
        out["od_length_prior"] = {
            "bearing_bins": max(1, bearing_bins),
            "distance_bins": max(1, distance_bins),
            "distance_thresholds_m": [
                float(value) for value in length_payload.get("distance_thresholds_m", [])
            ],
            "p10": p10.clamp_min(1e-3),
            "p50": p50.clamp_min(1e-3),
            "p90": p90.clamp_min(1e-3),
        }
    return out


def _reference_meter_scale(reference_lonlat: torch.Tensor) -> torch.Tensor:
    lat_mean = reference_lonlat[..., 1].mean(dim=1)
    deg_to_rad = math.pi / 180.0
    scale_x = torch.tensor(111_320.0, device=reference_lonlat.device, dtype=reference_lonlat.dtype) * torch.cos(
        lat_mean * deg_to_rad
    )
    scale_y = torch.full_like(scale_x, 110_540.0)
    return torch.stack([scale_x, scale_y], dim=-1)


def _local_prediction_to_lonlat(
    points: torch.Tensor,
    reference_lonlat: torch.Tensor,
    route_scales: torch.Tensor,
) -> torch.Tensor:
    origin = reference_lonlat[:, :1, :]
    meter_scale = _reference_meter_scale(reference_lonlat).view(reference_lonlat.shape[0], 1, 2)
    local_m = points * route_scales.view(-1, 1, 1).to(device=points.device, dtype=points.dtype)
    return local_m / meter_scale.clamp_min(1e-6) + origin


def _od_group_index(reference_lonlat: torch.Tensor, prior: dict) -> torch.Tensor:
    od_prior = prior.get("od_cell_prior", {})
    bearing_bins = max(1, int(od_prior.get("bearing_bins", 1)))
    distance_bins = max(1, int(od_prior.get("distance_bins", 1)))
    start = reference_lonlat[:, 0]
    end = reference_lonlat[:, -1]
    lat_mid = (start[:, 1] + end[:, 1]) * 0.5
    dx = (end[:, 0] - start[:, 0]) * torch.cos(lat_mid * (math.pi / 180.0))
    dy = end[:, 1] - start[:, 1]
    angle = safe_atan2(dy, dx)
    sector = torch.floor(((angle + math.pi) / (2.0 * math.pi)) * bearing_bins).long()
    sector = sector.clamp(0, bearing_bins - 1)

    meter_scale = _reference_meter_scale(reference_lonlat)
    distance_m = torch.linalg.norm((end - start) * meter_scale, dim=-1)
    dbin = torch.zeros_like(sector)
    thresholds = od_prior.get("distance_thresholds_m", [])
    for threshold in thresholds:
        dbin = dbin + (distance_m > float(threshold)).long()
    dbin = dbin.clamp(0, distance_bins - 1)
    return sector * distance_bins + dbin


def _bilinear_cell_ids_weights(
    lonlat: torch.Tensor,
    prior: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid = int(prior["grid_size"])
    lon_bound = prior["lon_bound"].to(device=lonlat.device, dtype=lonlat.dtype)
    lat_bound = prior["lat_bound"].to(device=lonlat.device, dtype=lonlat.dtype)
    x = (lonlat[..., 0] - lon_bound[0]) / (lon_bound[1] - lon_bound[0]).clamp_min(1e-9) * (grid - 1)
    y = (lonlat[..., 1] - lat_bound[0]) / (lat_bound[1] - lat_bound[0]).clamp_min(1e-9) * (grid - 1)
    x = x.clamp(0.0, float(grid - 1))
    y = y.clamp(0.0, float(grid - 1))
    x0 = torch.floor(x).long().clamp(0, grid - 1)
    y0 = torch.floor(y).long().clamp(0, grid - 1)
    x1 = (x0 + 1).clamp(0, grid - 1)
    y1 = (y0 + 1).clamp(0, grid - 1)
    wx = (x - x0.to(dtype=lonlat.dtype)).clamp(0.0, 1.0)
    wy = (y - y0.to(dtype=lonlat.dtype)).clamp(0.0, 1.0)
    ids = torch.stack(
        [
            y0 * grid + x0,
            y0 * grid + x1,
            y1 * grid + x0,
            y1 * grid + x1,
        ],
        dim=-1,
    )
    weights = torch.stack(
        [
            (1.0 - wx) * (1.0 - wy),
            wx * (1.0 - wy),
            (1.0 - wx) * wy,
            wx * wy,
        ],
        dim=-1,
    )
    hard_ids = y0 * grid + x0
    return ids, weights, hard_ids


def global_transition_prior_loss(
    pred: torch.Tensor,
    reference_lonlat: torch.Tensor,
    route_scales: torch.Tensor,
    prior: dict | None,
    points: int = 32,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Soft route-transition prior from aggregate training-set grid links.

    This mirrors TrajGPT-R's link/connection emphasis without using the current
    sample's middle trajectory as a condition. Consecutive duplicate grid cells
    are ignored because the Table-2 tokenizer also collapses repeats.
    """
    if prior is None:
        zero = pred.new_tensor(0.0)
        return zero, {"global_transition_prior": 0.0, "global_transition_valid_rate": 0.0}
    original_ndim = pred.ndim
    if original_ndim == 4:
        bsz, num_modes, seq_len, dim = pred.shape
        pred_flat = pred.reshape(bsz * num_modes, seq_len, dim)
        ref_flat = reference_lonlat[:, None].expand(-1, num_modes, -1, -1).reshape(
            bsz * num_modes,
            reference_lonlat.shape[1],
            reference_lonlat.shape[2],
        )
        scale_flat = route_scales[:, None].expand(-1, num_modes).reshape(-1)
    elif original_ndim == 3:
        pred_flat = pred
        ref_flat = reference_lonlat
        scale_flat = route_scales
    else:
        zero = pred.new_tensor(0.0)
        return zero, {"global_transition_prior": 0.0, "global_transition_valid_rate": 0.0}

    sample_points = max(2, min(int(points), int(pred_flat.shape[1])))
    if sample_points != pred_flat.shape[1]:
        pred_flat = F.interpolate(
            pred_flat.transpose(1, 2),
            size=sample_points,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    lonlat = _local_prediction_to_lonlat(pred_flat.float(), ref_flat.float(), scale_flat.float())
    ids, weights, hard_ids = _bilinear_cell_ids_weights(lonlat, prior)
    src_ids, dst_ids = ids[:, :-1], ids[:, 1:]
    src_w, dst_w = weights[:, :-1], weights[:, 1:]
    transition_nll = prior["transition_nll"].to(device=pred.device, dtype=pred.dtype)
    pair_nll = transition_nll[src_ids[..., :, None], dst_ids[..., None, :]]
    pair_weight = src_w[..., :, None] * dst_w[..., None, :]
    expected_nll = (pair_weight.to(dtype=pred.dtype) * pair_nll).sum(dim=(-2, -1))
    changed = hard_ids[:, 1:] != hard_ids[:, :-1]
    if not bool(changed.any()):
        zero = pred.new_tensor(0.0)
        return zero, {"global_transition_prior": 0.0, "global_transition_valid_rate": 0.0}
    denom = changed.to(dtype=pred.dtype).sum().clamp_min(1.0)
    normalizer = max(float(prior.get("default_nll", 1.0)), 1e-6)
    loss = (expected_nll * changed.to(dtype=pred.dtype)).sum() / denom / normalizer
    stats = {
        "global_transition_prior": float(loss.detach().item()),
        "global_transition_valid_rate": float(changed.to(dtype=torch.float32).mean().detach().item()),
    }
    return loss, stats


def od_cell_prior_loss(
    pred: torch.Tensor,
    reference_lonlat: torch.Tensor,
    route_scales: torch.Tensor,
    prior: dict | None,
    points: int = 32,
) -> tuple[torch.Tensor, dict[str, float]]:
    """OD-conditioned aggregate cell prior for non-leaking city-route regularization.

    The prior is grouped only by origin/destination chord bearing and distance.
    It therefore provides a weak city-structure target without exposing the
    current sample's middle trajectory as a condition.
    """
    if prior is None or "od_cell_prior" not in prior:
        zero = pred.new_tensor(0.0)
        return zero, {"od_cell_prior": 0.0, "od_cell_valid_rate": 0.0}
    original_ndim = pred.ndim
    if original_ndim == 4:
        bsz, num_modes, seq_len, dim = pred.shape
        pred_flat = pred.reshape(bsz * num_modes, seq_len, dim)
        ref_flat = reference_lonlat[:, None].expand(-1, num_modes, -1, -1).reshape(
            bsz * num_modes,
            reference_lonlat.shape[1],
            reference_lonlat.shape[2],
        )
        scale_flat = route_scales[:, None].expand(-1, num_modes).reshape(-1)
    elif original_ndim == 3:
        pred_flat = pred
        ref_flat = reference_lonlat
        scale_flat = route_scales
    else:
        zero = pred.new_tensor(0.0)
        return zero, {"od_cell_prior": 0.0, "od_cell_valid_rate": 0.0}

    sample_points = max(2, min(int(points), int(pred_flat.shape[1])))
    if sample_points != pred_flat.shape[1]:
        pred_flat = F.interpolate(
            pred_flat.transpose(1, 2),
            size=sample_points,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    lonlat = _local_prediction_to_lonlat(pred_flat.float(), ref_flat.float(), scale_flat.float())
    ids, weights, hard_ids = _bilinear_cell_ids_weights(lonlat, prior)
    group_idx = _od_group_index(ref_flat.float(), prior)
    cell_group_nll = prior["od_cell_prior"]["cell_group_nll"].to(device=pred.device, dtype=pred.dtype)
    group_idx = group_idx.to(device=pred.device).clamp(0, cell_group_nll.shape[0] - 1)
    nll = cell_group_nll[group_idx[:, None, None], ids]
    expected_nll = (nll * weights.to(device=pred.device, dtype=pred.dtype)).sum(dim=-1)
    changed_or_endpoint = torch.ones_like(hard_ids, dtype=torch.bool)
    if hard_ids.shape[1] > 2:
        changed_or_endpoint[:, 1:-1] = (
            (hard_ids[:, 1:-1] != hard_ids[:, :-2])
            | (hard_ids[:, 1:-1] != hard_ids[:, 2:])
        )
    denom = changed_or_endpoint.to(dtype=pred.dtype).sum().clamp_min(1.0)
    group_default = prior["od_cell_prior"]["group_default_nll"].to(device=pred.device, dtype=pred.dtype)
    normalizer = group_default[group_idx].mean().clamp_min(1e-6)
    loss = (expected_nll * changed_or_endpoint.to(dtype=pred.dtype)).sum() / denom / normalizer
    stats = {
        "od_cell_prior": float(loss.detach().item()),
        "od_cell_valid_rate": float(changed_or_endpoint.to(dtype=torch.float32).mean().detach().item()),
    }
    return loss, stats


def route_grid_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coarse_points: int = 32,
    grid_size: int = 12,
    temperature: float = 0.08,
    padding: float = 0.10,
) -> torch.Tensor:
    """Differentiable local-grid route-token supervision.

    GT middle geometry is used only as a training target, not as model input.
    This is a continuous analogue of TrajGPT-R/link-token supervision: the
    generated coarse waypoints must fall into the same local route cells as the
    target waypoints.
    """
    coarse_points = int(coarse_points)
    grid_size = int(grid_size)
    if coarse_points <= 1 or grid_size <= 1:
        return pred.new_tensor(0.0)

    pred_c = _resample_like(pred.float(), coarse_points)
    target_c = _resample_like(target.float(), coarse_points)
    losses: List[torch.Tensor] = []
    pad = max(float(padding), 0.0)

    for pred_i, target_i in zip(pred_c, target_c):
        xy_min = target_i.amin(dim=0)
        xy_max = target_i.amax(dim=0)
        span = (xy_max - xy_min).clamp_min(0.05)
        xy_min = xy_min - pad * span
        xy_max = xy_max + pad * span
        span = (xy_max - xy_min).clamp_min(1e-6)

        target_norm = ((target_i - xy_min) / span).clamp(0.0, 1.0 - 1e-6)
        target_col = torch.floor(target_norm[:, 0] * grid_size).long().clamp(0, grid_size - 1)
        target_row = torch.floor(target_norm[:, 1] * grid_size).long().clamp(0, grid_size - 1)
        target_center = torch.stack(
            [
                (target_col.to(dtype=pred.dtype) + 0.5) / float(grid_size),
                (target_row.to(dtype=pred.dtype) + 0.5) / float(grid_size),
            ],
            dim=-1,
        )
        pred_norm = ((pred_i - xy_min) / span).clamp(0.0, 1.0)
        cell_delta = (pred_norm - target_center) * float(grid_size)
        beta = max(float(temperature), 1e-3)
        losses.append(
            F.smooth_l1_loss(
                cell_delta,
                torch.zeros_like(cell_delta),
                beta=beta,
                reduction="mean",
            )
        )

    return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)


def route_transition_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coarse_points: int = 32,
    grid_size: int = 12,
    beta: float = 0.08,
    padding: float = 0.10,
    stationary_weight: float = 0.25,
) -> torch.Tensor:
    """Align local-grid movement vectors with the target route.

    This is a differentiable counterpart of TrajGPT-R/ControlTraj-style
    transition metrics. The target middle trajectory is used only as
    supervision; the model still receives only the OD-chord reference.
    """
    coarse_points = int(coarse_points)
    grid_size = int(grid_size)
    if coarse_points <= 2 or grid_size <= 1:
        return pred.new_tensor(0.0)

    pred_c = _resample_like(pred.float(), coarse_points)
    target_c = _resample_like(target.float(), coarse_points)
    losses: List[torch.Tensor] = []
    pad = max(float(padding), 0.0)
    beta = max(float(beta), 1e-4)
    stationary_weight = max(float(stationary_weight), 0.0)

    for pred_i, target_i in zip(pred_c, target_c):
        xy_min = target_i.amin(dim=0)
        xy_max = target_i.amax(dim=0)
        span = (xy_max - xy_min).clamp_min(0.05)
        xy_min = xy_min - pad * span
        xy_max = xy_max + pad * span
        span = (xy_max - xy_min).clamp_min(1e-6)

        pred_norm = ((pred_i - xy_min) / span).clamp(0.0, 1.0)
        target_norm = ((target_i - xy_min) / span).clamp(0.0, 1.0)
        pred_step = (pred_norm[1:] - pred_norm[:-1]) * float(grid_size)
        target_step = (target_norm[1:] - target_norm[:-1]) * float(grid_size)

        target_cell = torch.floor(target_norm * float(grid_size)).long().clamp(0, grid_size - 1)
        changed = (target_cell[1:] != target_cell[:-1]).any(dim=-1).to(dtype=pred_step.dtype)
        weights = torch.where(changed > 0, torch.ones_like(changed), changed.new_full(changed.shape, stationary_weight))
        step_loss = F.smooth_l1_loss(pred_step, target_step, beta=beta, reduction="none").mean(dim=-1)
        denom = weights.sum().clamp_min(1.0)
        losses.append((step_loss * weights).sum() / denom)

    return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)


def _route_transition_alignment_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    coarse_points: int = 32,
    grid_size: int = 12,
    beta: float = 0.08,
    padding: float = 0.10,
    stationary_weight: float = 0.25,
) -> torch.Tensor:
    """Per-sample version of route transition alignment for candidate ranking."""
    coarse_points = int(coarse_points)
    grid_size = int(grid_size)
    if coarse_points <= 2 or grid_size <= 1:
        return pred.new_zeros(pred.shape[0])

    pred_c = _resample_like(pred.float(), coarse_points)
    target_c = _resample_like(target.float(), coarse_points)
    losses: List[torch.Tensor] = []
    pad = max(float(padding), 0.0)
    beta = max(float(beta), 1e-4)
    stationary_weight = max(float(stationary_weight), 0.0)

    for pred_i, target_i in zip(pred_c, target_c):
        xy_min = target_i.amin(dim=0)
        xy_max = target_i.amax(dim=0)
        span = (xy_max - xy_min).clamp_min(0.05)
        xy_min = xy_min - pad * span
        xy_max = xy_max + pad * span
        span = (xy_max - xy_min).clamp_min(1e-6)

        pred_norm = ((pred_i - xy_min) / span).clamp(0.0, 1.0)
        target_norm = ((target_i - xy_min) / span).clamp(0.0, 1.0)
        pred_step = (pred_norm[1:] - pred_norm[:-1]) * float(grid_size)
        target_step = (target_norm[1:] - target_norm[:-1]) * float(grid_size)

        target_cell = torch.floor(target_norm * float(grid_size)).long().clamp(0, grid_size - 1)
        changed = (target_cell[1:] != target_cell[:-1]).any(dim=-1).to(dtype=pred_step.dtype)
        weights = torch.where(
            changed > 0,
            torch.ones_like(changed),
            changed.new_full(changed.shape, stationary_weight),
        )
        step_loss = F.smooth_l1_loss(pred_step, target_step, beta=beta, reduction="none").mean(dim=-1)
        losses.append((step_loss * weights).sum() / weights.sum().clamp_min(1.0))

    return torch.stack(losses).to(device=pred.device, dtype=pred.dtype) if losses else pred.new_zeros(pred.shape[0])


def candidate_guided_route_direction_loss(
    candidates: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    tokens: torch.Tensor,
    route_scales: torch.Tensor | None = None,
    route_weight: float = 1.0,
    direction_weight: float = 1.0,
    temperature: float = 0.10,
    route_points: int = 32,
    route_grid_size: int = 12,
    route_beta: float = 0.08,
    class_counts: torch.Tensor | None = None,
    class_balance_beta: float = 0.999,
    mode: str = "joint_energy",
    direction_gate_temperature: float = 0.08,
    direction_gate_margin: float = 0.05,
    candidate_metrics: Dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Joint soft-best route/direction objective over K candidates.

    The loss follows the GTG-style idea of guiding generation with proxy
    objectives, but keeps the clean OD-chord protocol: GT middle trajectory is
    used only as target-side supervision. A candidate is rewarded only when it
    is simultaneously route-transition faithful and instruction-direction
    faithful, avoiding the post-hoc failure where a lower Route Transition JSD
    candidate breaks Direction F1.
    """
    if candidates.ndim != 4 or candidates.shape[1] <= 1:
        selected = candidates if candidates.ndim == 3 else candidates[:, 0]
        route = route_transition_alignment_loss(
            selected,
            target,
            coarse_points=route_points,
            grid_size=route_grid_size,
            beta=route_beta,
        )
        direction, direction_stats = contract_direction_shape_loss(
            selected,
            target,
            reference,
            tokens,
            route_scales=route_scales,
            class_counts=class_counts,
            class_balance_beta=class_balance_beta,
        )
        total = float(route_weight) * route + float(direction_weight) * direction
        return total.to(dtype=selected.dtype), {
            "candidate_guided_route_direction": float(total.detach().item()),
            "candidate_guided_route_direction_route": float(route.detach().item()),
            "candidate_guided_route_direction_direction": float(direction.detach().item()),
            "candidate_guided_route_direction_best": float(total.detach().item()),
            "candidate_guided_route_direction_route_best": float(route.detach().item()),
            "candidate_guided_route_direction_direction_best": float(direction.detach().item()),
            "candidate_guided_route_direction_route_top_match": 1.0,
            "candidate_guided_route_direction_direction_top_match": 1.0,
            "contract_direction_shape": direction_stats.get("contract_direction_shape", 0.0),
        }

    batch, num_modes, target_len, coord_dim = candidates.shape
    flat = candidates.reshape(batch * num_modes, target_len, coord_dim)
    target_rep = target.unsqueeze(1).expand(-1, num_modes, -1, -1).reshape(
        batch * num_modes,
        target.shape[-2],
        target.shape[-1],
    )
    reference_rep = reference.unsqueeze(1).expand(-1, num_modes, -1, -1).reshape(
        batch * num_modes,
        reference.shape[-2],
        reference.shape[-1],
    )
    tokens_rep = tokens.unsqueeze(1).expand(-1, num_modes, -1).reshape(batch * num_modes, tokens.shape[-1])
    scales_rep = None
    if route_scales is not None:
        scales_rep = route_scales.view(batch, 1).expand(-1, num_modes).reshape(-1)

    route_per = _route_transition_alignment_per_sample(
        flat,
        target_rep,
        coarse_points=route_points,
        grid_size=route_grid_size,
        beta=route_beta,
    ).view(batch, num_modes)
    direction_per, _ = _contract_direction_shape_per_sample(
        flat,
        target_rep,
        reference_rep,
        tokens_rep,
        route_scales=scales_rep,
        class_counts=class_counts,
        class_balance_beta=class_balance_beta,
        target_metrics=None,
        pred_metrics=candidate_metrics,
    )
    direction_per = direction_per.view(batch, num_modes)

    mode = str(mode)
    if mode not in {"joint_energy", "direction_gated_route"}:
        raise ValueError(f"Unsupported candidate guided route-direction mode: {mode}")
    energy = float(route_weight) * route_per + float(direction_weight) * direction_per
    temp = max(float(temperature), 1e-6)
    best_energy, best_idx = energy.min(dim=1)
    route_best, route_idx = route_per.min(dim=1)
    direction_best, direction_idx = direction_per.min(dim=1)
    if mode == "direction_gated_route":
        gate_temp = max(float(direction_gate_temperature), 1e-6)
        gate_margin = max(float(direction_gate_margin), 0.0)
        direction_gate = torch.sigmoid((direction_best.unsqueeze(1) + gate_margin - direction_per) / gate_temp)
        gated_route_logits = -route_per / temp + torch.log(direction_gate.clamp_min(1e-6))
        weights = torch.softmax(gated_route_logits, dim=1)
        route_soft = (weights * route_per).sum(dim=1)
        direction_soft = (weights * direction_per).sum(dim=1)
        gate_regularizer = (weights * direction_per).sum(dim=1)
        softbest = float(route_weight) * route_soft + float(direction_weight) * gate_regularizer
        best_idx = torch.argmin(route_per.masked_fill(direction_gate < 0.5, float("inf")), dim=1)
        no_valid = torch.isinf(route_per.masked_fill(direction_gate < 0.5, float("inf"))).all(dim=1)
        best_idx = torch.where(no_valid, direction_idx, best_idx)
        best_energy = energy.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    else:
        weights = torch.softmax(-energy / temp, dim=1)
        softbest = (weights * energy).sum(dim=1)
        route_soft = (weights * route_per).sum(dim=1)
        direction_soft = (weights * direction_per).sum(dim=1)
        direction_gate = torch.ones_like(direction_per)
    stats = {
        "candidate_guided_route_direction": float(softbest.detach().mean().item()),
        "candidate_guided_route_direction_route": float(route_soft.detach().mean().item()),
        "candidate_guided_route_direction_direction": float(direction_soft.detach().mean().item()),
        "candidate_guided_route_direction_best": float(best_energy.detach().mean().item()),
        "candidate_guided_route_direction_route_best": float(route_best.detach().mean().item()),
        "candidate_guided_route_direction_direction_best": float(direction_best.detach().mean().item()),
        "candidate_guided_route_direction_route_top_match": float((best_idx == route_idx).float().mean().detach().item()),
        "candidate_guided_route_direction_direction_top_match": float(
            (best_idx == direction_idx).float().mean().detach().item()
        ),
        "candidate_guided_route_direction_gate_mean": float(direction_gate.detach().mean().item()),
    }
    return softbest.mean().to(dtype=candidates.dtype), stats


def soft_reference_loss(
    pred: torch.Tensor,
    reference: torch.Tensor,
    margin: float = 0.03,
    beta: float = 0.01,
    norm: float = 0.08,
) -> torch.Tensor:
    if reference.shape[1] < 2:
        return pred.new_tensor(0.0)
    dist = point_segment_distance(pred, reference[:, :-1], reference[:, 1:])
    excess = F.softplus((dist - float(margin)) / max(float(beta), 1e-6)) * float(beta)
    return F.smooth_l1_loss(
        excess / max(float(norm), 1e-6),
        torch.zeros_like(excess),
        reduction="mean",
    )


def candidate_pairwise_distance_m(
    candidates: torch.Tensor,
    route_scales: torch.Tensor,
    mode: Literal["trajectory", "midpoint", "both"] = "trajectory",
) -> torch.Tensor:
    """Mean pairwise candidate separation in meters for [B, K, T, 2]."""
    if candidates.ndim != 4 or candidates.shape[1] < 2:
        return candidates.new_zeros(candidates.shape[0])
    route_scales = route_scales.view(-1, 1, 1).to(device=candidates.device, dtype=candidates.dtype)
    diff = candidates[:, :, None] - candidates[:, None, :]
    point_dist = torch.linalg.norm(diff, dim=-1) * route_scales
    if mode == "midpoint":
        t = torch.linspace(0.0, 1.0, candidates.shape[2], device=candidates.device, dtype=candidates.dtype)
        weights = torch.exp(-0.5 * ((t - 0.5) / 0.18).pow(2))
        weights = weights / weights.sum().clamp_min(1e-6)
        dist = (point_dist * weights.view(1, 1, 1, -1)).sum(dim=-1)
    elif mode == "both":
        traj_dist = point_dist.mean(dim=-1)
        mid_dist = candidate_pairwise_distance_m(candidates, route_scales.view(-1), mode="midpoint")
        k = candidates.shape[1]
        pair_mask = torch.triu(torch.ones(k, k, dtype=torch.bool, device=candidates.device), diagonal=1)
        return 0.5 * traj_dist[:, pair_mask].mean(dim=1) + 0.5 * mid_dist
    else:
        dist = point_dist.mean(dim=-1)
    k = candidates.shape[1]
    pair_mask = torch.triu(torch.ones(k, k, dtype=torch.bool, device=candidates.device), diagonal=1)
    return dist[:, pair_mask].mean(dim=1)


def candidate_endpoint_distance_m(candidates: torch.Tensor, route_scales: torch.Tensor) -> torch.Tensor:
    if candidates.ndim != 4 or candidates.shape[1] < 2:
        return candidates.new_zeros(candidates.shape[0])
    endpoints = candidates[:, :, -1, :]
    diff = endpoints[:, :, None] - endpoints[:, None, :]
    dist = torch.linalg.norm(diff, dim=-1) * route_scales.view(-1, 1, 1).to(
        device=candidates.device,
        dtype=candidates.dtype,
    )
    k = candidates.shape[1]
    pair_mask = torch.triu(torch.ones(k, k, dtype=torch.bool, device=candidates.device), diagonal=1)
    return dist[:, pair_mask].mean(dim=1)


def candidate_diversity_loss(
    candidates: torch.Tensor,
    route_scales: torch.Tensor,
    target_m: float = 30.0,
    max_m: float = 120.0,
    mode: Literal["trajectory", "midpoint", "both"] = "both",
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Mode-separation loss inspired by MTR-style multi-query candidate training.

    The lower hinge prevents K candidates from collapsing to the same shape.
    The optional upper hinge keeps non-oracle modes from drifting into unrealistic
    routes while the oracle candidate is still constrained by geometry loss.
    """
    if candidates.ndim != 4 or candidates.shape[1] < 2:
        zero = candidates.new_tensor(0.0)
        return zero, {
            "candidate_diversity_m": 0.0,
            "candidate_endpoint_diversity_m": 0.0,
            "candidate_diversity_loss": 0.0,
        }
    diversity = candidate_pairwise_distance_m(candidates.float(), route_scales.float(), mode=mode)
    endpoint_diversity = candidate_endpoint_distance_m(candidates.float(), route_scales.float())
    lower = F.relu(float(target_m) - diversity) / max(float(target_m), 1.0)
    if max_m and float(max_m) > float(target_m):
        upper = F.relu(diversity - float(max_m)) / max(float(max_m), 1.0)
    else:
        upper = torch.zeros_like(lower)
    loss = (lower.pow(2) + 0.25 * upper.pow(2)).mean()
    stats = {
        "candidate_diversity_m": float(diversity.detach().mean().item()),
        "candidate_endpoint_diversity_m": float(endpoint_diversity.detach().mean().item()),
        "candidate_diversity_loss": float(loss.detach().item()),
    }
    return loss, stats


MODE_ANCHOR_FEATURES = [
    "length_ratio",
    "progress",
    "curvature",
    "mean_reference_distance",
    "abs_heading_change",
]


CALIBRATION_METRIC_ALIASES = {
    "progress_ratio": "displacement_ratio",
    "pace_avg_speed_kmh": "avg_speed_kmh",
    "mean_reference_distance": "mean_reference_distance_norm",
    "direction_token_metric": "signed_net_heading_rad",
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


def load_token_geometry_calibration(
    path: str | Path | None,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict | None:
    """Load data-driven token geometry centers/scales for differentiable control loss."""
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_fields = payload.get("fields", {})
    calibration: dict = {}
    for field in DEFAULT_MOBILITY_VOCAB.fields:
        field_payload = raw_fields.get(field, {})
        default_metric = str(field_payload.get("metric", ""))
        default_metric = CALIBRATION_METRIC_ALIASES.get(default_metric, default_metric)
        token_metrics: List[str] = []
        centers: List[float] = []
        scales: List[float] = []
        supports: List[int] = []
        for token_name in DEFAULT_MOBILITY_VOCAB.token_to_id[field]:
            stats = field_payload.get("tokens", {}).get(token_name, {})
            metric = str(stats.get("direction_metric", default_metric))
            metric = CALIBRATION_METRIC_ALIASES.get(metric, metric)
            center = _safe_float(stats.get("p50", stats.get("mean", 0.0)), default=0.0)
            scale = _safe_float(stats.get("scale", stats.get("iqr", stats.get("std", 1.0))), default=1.0)
            token_metrics.append(metric)
            centers.append(center)
            scales.append(max(scale, 1e-6))
            supports.append(int(stats.get("count", 0)))
        calibration[field] = {
            "field": field,
            "metrics": token_metrics,
            "centers": torch.tensor(centers, device=device, dtype=dtype),
            "scales": torch.tensor(scales, device=device, dtype=dtype).clamp_min(1e-6),
            "supports": supports,
        }
    return calibration


def _safe_float(value, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _full_heading_metrics(points: torch.Tensor) -> Dict[str, torch.Tensor]:
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


def _reference_lateral_metrics(points: torch.Tensor, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
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


def _soft_reference_direction_descriptors(points: torch.Tensor, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Differentiable reference-relative direction descriptors.

    The evaluation/calibration code uses hard side ratios. Those are useful for
    reporting but weak for training because left/right side counts are almost
    non-differentiable. These soft descriptors keep the same geometric meaning
    while allowing counterfactual direction losses to push trajectories.
    """
    lateral_metrics = _reference_lateral_metrics(points, reference)
    lateral = None
    if points.shape[1] >= 2 and reference.shape[1] >= 2:
        diffs = torch.zeros_like(reference)
        diffs[:, 1:-1] = reference[:, 2:] - reference[:, :-2]
        diffs[:, 0] = reference[:, 1] - reference[:, 0]
        diffs[:, -1] = reference[:, -1] - reference[:, -2]
        tangent = diffs / torch.linalg.norm(diffs, dim=-1, keepdim=True).clamp_min(1e-8)
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
        lateral = ((points - reference) * normal).sum(dim=-1)
    if lateral is None:
        zeros = points.new_zeros(points.shape[0])
        lateral = points.new_zeros(points.shape[0], points.shape[1])
        soft_left = zeros
        soft_right = zeros
    else:
        # Local normalized coordinates have typical lateral magnitudes in
        # hundredths; this temperature keeps left/right gradients alive near 0.
        temp = lateral.abs().detach().median(dim=1).values.clamp_min(0.02).view(-1, 1)
        left_prob = torch.sigmoid(lateral / temp)
        right_prob = torch.sigmoid(-lateral / temp)
        soft_left = left_prob.mean(dim=1)
        soft_right = right_prob.mean(dim=1)

    heading = _full_heading_metrics(points)
    return {
        **lateral_metrics,
        "soft_left_side_ratio": soft_left,
        "soft_right_side_ratio": soft_right,
        "soft_left_minus_right": soft_left - soft_right,
        "soft_right_minus_left": soft_right - soft_left,
        "total_abs_heading_rad": heading["total_abs_heading_rad"],
        "max_abs_turn_rad": heading["max_abs_turn_rad"],
        "turn_count": heading["turn_count"],
    }


def _local_direction_descriptors(points: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Differentiable iMotion-style local-frame direction descriptors.

    The trajectory is rotated into its own initial motion frame, mirroring the
    local-coordinate reasoning used by iMotion-LLM direction rules while keeping
    the quantities differentiable for training.
    """
    diffs = points[:, 1:] - points[:, :-1]
    if diffs.shape[1] < 2:
        zeros = points.new_zeros(points.shape[0])
        return {
            "local_dx": zeros,
            "local_dy": zeros,
            "heading_diff": zeros,
            "signed_total_turn": zeros,
            "total_abs_turn": zeros,
            "soft_turn_count": zeros,
            "left_turn_ratio": zeros,
            "right_turn_ratio": zeros,
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
    turn_threshold = math.radians(15.0)
    turn_temp = math.radians(5.0)
    soft_turn = torch.sigmoid((abs_delta - turn_threshold) / max(turn_temp, 1e-6))
    left_soft = torch.sigmoid(delta / max(turn_temp, 1e-6)) * soft_turn
    right_soft = torch.sigmoid(-delta / max(turn_temp, 1e-6)) * soft_turn
    soft_count = soft_turn.sum(dim=1)
    denom = soft_count.clamp_min(1e-6)

    return {
        "local_dx": local_dx,
        "local_dy": local_dy,
        "local_heading_diff_rad": heading_diff,
        "local_abs_heading_diff_rad": heading_diff.abs(),
        "local_signed_total_turn_rad": delta.sum(dim=1),
        "local_total_abs_turn_rad": abs_delta.sum(dim=1),
        "local_left_turn_ratio": left_soft.sum(dim=1) / denom,
        "local_right_turn_ratio": right_soft.sum(dim=1) / denom,
        "local_max_abs_turn_rad": abs_delta.amax(dim=1),
        "local_turn_count": soft_count,
        "heading_diff": heading_diff,
        "signed_total_turn": delta.sum(dim=1),
        "total_abs_turn": abs_delta.sum(dim=1),
        "soft_turn_count": soft_count,
        "left_turn_ratio": left_soft.sum(dim=1) / denom,
        "right_turn_ratio": right_soft.sum(dim=1) / denom,
    }


def _smooth_route_shape_descriptors(
    points: torch.Tensor,
    route_scales: torch.Tensor | None = None,
    coarse_points: int = 16,
) -> Dict[str, torch.Tensor]:
    """Low-frequency descriptors approximating the official direction contract.

    The official OD-chord labels are mined from a simplified macro route, not
    from every high-frequency resampled step.  These differentiable coarse
    descriptors keep the training loss aligned with that contract while still
    giving useful gradients to the generator.
    """
    coarse = _resample_like(points.float(), int(max(coarse_points, 4)))
    diffs = coarse[:, 1:] - coarse[:, :-1]
    if diffs.shape[1] < 2:
        zeros = points.new_zeros(points.shape[0])
        return {
            "contract_net_heading_rad": zeros,
            "contract_abs_net_heading_rad": zeros,
            "contract_total_abs_heading_rad": zeros,
            "contract_soft_turn_count": zeros,
            "contract_displacement_ratio": zeros,
        }
    seg_len = torch.linalg.norm(diffs, dim=-1)
    if route_scales is not None:
        physical_len = seg_len * route_scales.view(-1, 1).to(device=points.device, dtype=points.dtype)
        valid_weight = torch.sigmoid((physical_len - 25.0) / 10.0).to(dtype=points.dtype)
    else:
        valid_weight = torch.ones_like(seg_len)
    headings = safe_atan2(diffs[..., 1], diffs[..., 0])
    delta = headings[:, 1:] - headings[:, :-1]
    delta = safe_atan2(torch.sin(delta), torch.cos(delta))
    pair_weight = torch.minimum(valid_weight[:, 1:], valid_weight[:, :-1])
    abs_delta = delta.abs()
    turn_threshold = math.radians(35.0)
    turn_temp = math.radians(10.0)
    soft_strong = torch.sigmoid((abs_delta - turn_threshold) / max(turn_temp, 1e-6)) * pair_weight
    signed_net = (delta * pair_weight).sum(dim=1)
    total_abs = (abs_delta * soft_strong).sum(dim=1)
    soft_count = soft_strong.sum(dim=1)
    displacement = torch.linalg.norm(points[:, -1] - points[:, 0], dim=-1)
    length = path_length(points).clamp_min(1e-6)
    return {
        "contract_net_heading_rad": signed_net,
        "contract_abs_net_heading_rad": signed_net.abs(),
        "contract_total_abs_heading_rad": total_abs,
        "contract_soft_turn_count": soft_count,
        "contract_displacement_ratio": displacement / length,
    }


def _calibrated_geometry_metrics(
    pred: torch.Tensor,
    reference: torch.Tensor,
    route_scales: torch.Tensor | None = None,
    trip_duration_min: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    heading = _full_heading_metrics(pred)
    local_direction = _local_direction_descriptors(pred)
    lateral = _reference_lateral_metrics(pred, reference)
    contract_direction = _smooth_route_shape_descriptors(pred, route_scales=route_scales)
    pred_len = path_length(pred)
    ref_len = path_length(reference).clamp_min(1e-6)
    ref_dist = mean_reference_distance(pred, reference)
    metrics = {
        "length_ratio": pred_len / ref_len,
        "curvature_energy": curvature_energy(pred),
        "contract_curvature_score": contract_curvature_score(pred, route_scales),
        "turn_density_score": turn_density_score(pred, route_scales),
        "displacement_ratio": displacement_ratio(pred),
        "mean_reference_distance_norm": ref_dist,
        **heading,
        **local_direction,
        **lateral,
        **contract_direction,
    }
    if trip_duration_min is not None:
        duration = trip_duration_min.view(-1).to(device=pred.device, dtype=pred.dtype).clamp_min(1e-6)
        if route_scales is None:
            length_m = pred_len
        else:
            length_m = pred_len * route_scales.view(-1).to(device=pred.device, dtype=pred.dtype)
        length_km = (length_m / 1000.0).clamp_min(0.0)
        avg_speed = (length_km / (duration / 60.0)).clamp(0.0, 120.0)
        metrics.update(
            {
                "trip_duration_min": duration,
                "generated_total_distance_km": length_km,
                "avg_speed_kmh": avg_speed,
                "time_per_km_min": duration / length_km.clamp_min(1e-6),
            }
        )
    return metrics


def calibrated_token_geometry_loss(
    pred: torch.Tensor,
    reference: torch.Tensor,
    tokens: torch.Tensor,
    calibration: dict | None,
    route_scales: torch.Tensor | None = None,
    metrics: Dict[str, torch.Tensor] | None = None,
    fields: Iterable[str] | None = None,
    temperature: float = 1.0,
    field_weights: Dict[str, float] | None = None,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    class_balance_beta: float = 0.999,
    class_counts: Dict[str, torch.Tensor] | None = None,
    reduction: str = "mean",
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Classification loss over geometry-derived token logits from calibration centers.

    This directly optimizes the same instruction attributes used by Table 3:
    each token receives a logit based on negative normalized distance between
    the generated trajectory geometry and that token's calibrated center.
    Balanced/focal variants are useful when a field collapses to frequent
    classes, such as Porto direction and reference-adherence labels.
    """
    if calibration is None:
        zero = pred.new_tensor(0.0)
        return zero, {"calibrated_token": 0.0, "calibrated_token_acc": 0.0}
    selected_fields = list(fields) if fields is not None else list(DEFAULT_MOBILITY_VOCAB.fields)
    if not selected_fields:
        zero = pred.new_tensor(0.0)
        return zero, {"calibrated_token": 0.0, "calibrated_token_acc": 0.0}

    metrics = (
        metrics
        if isinstance(metrics, dict)
        else _calibrated_geometry_metrics(pred.float(), reference.float(), route_scales=route_scales)
    )
    losses: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []
    accuracies: List[torch.Tensor] = []
    stats: Dict[str, float] = {}
    temp = max(float(temperature), 1e-6)
    field_weights = field_weights or {}
    reduction = str(reduction)
    if reduction not in {"mean", "none"}:
        raise ValueError(f"Unsupported reduction: {reduction}")

    field_index = {field: idx for idx, field in enumerate(DEFAULT_MOBILITY_VOCAB.fields)}
    for field in selected_fields:
        if field not in DEFAULT_MOBILITY_VOCAB.fields:
            raise ValueError(f"Unknown calibrated token field: {field}")
        field_calib = calibration[field]
        if field == "direction":
            # Direction labels are defined by the official macro geometry
            # contract, not by a single calibrated scalar.  Lower score means
            # more compatible, so use negative scores as classification logits.
            logits = -_field_all_token_scores(metrics, field_calib, dtype=pred.dtype) / temp
        else:
            centers = field_calib["centers"].to(device=pred.device, dtype=pred.dtype)
            scales = field_calib["scales"].to(device=pred.device, dtype=pred.dtype).clamp_min(1e-6)
            if any(metric_name not in metrics for metric_name in field_calib["metrics"]):
                stats[f"calibrated_token_{field}"] = 0.0
                stats[f"calibrated_token_{field}_acc"] = 0.0
                continue
            token_logits = []
            for token_idx, metric_name in enumerate(field_calib["metrics"]):
                value = metrics[metric_name].to(dtype=pred.dtype)
                z = (value - centers[token_idx]) / scales[token_idx]
                token_logits.append(-z.abs() / temp)
            logits = torch.stack(token_logits, dim=1)
        target = tokens[:, field_index[field]].long()
        class_weight = None
        if loss_type in {"balanced_ce", "balanced_focal"}:
            if class_counts is not None and field in class_counts:
                supports = class_counts[field].to(device=pred.device, dtype=pred.dtype)
            else:
                supports = torch.tensor(
                    field_calib.get("supports", [1 for _ in range(logits.shape[1])]),
                    device=pred.device,
                    dtype=pred.dtype,
                )
            supports = supports.clamp_min(1.0)
            beta = min(max(float(class_balance_beta), 0.0), 0.999999)
            if beta > 0.0:
                effective_num = 1.0 - torch.pow(supports.new_full(supports.shape, beta), supports)
                class_weight = (1.0 - beta) / effective_num.clamp_min(1e-8)
            else:
                class_weight = 1.0 / supports
            class_weight = class_weight / class_weight.mean().clamp_min(1e-8)

        if loss_type in {"focal", "balanced_focal"}:
            log_probs = F.log_softmax(logits, dim=1)
            probs = log_probs.exp()
            target_log_prob = log_probs.gather(1, target[:, None]).squeeze(1)
            target_prob = probs.gather(1, target[:, None]).squeeze(1).clamp(1e-6, 1.0)
            per_sample_loss = -target_log_prob * torch.pow(1.0 - target_prob, max(float(focal_gamma), 0.0))
            if class_weight is not None:
                per_sample_loss = per_sample_loss * class_weight.gather(0, target)
            field_loss = per_sample_loss
        elif loss_type in {"ce", "balanced_ce"}:
            field_loss = F.cross_entropy(logits, target, weight=class_weight, reduction="none")
        else:
            raise ValueError(f"Unknown calibrated token loss type: {loss_type}")
        field_acc = (logits.argmax(dim=1) == target).float()
        losses.append(field_loss)
        weights.append(pred.new_tensor(float(field_weights.get(field, 1.0))).clamp_min(0.0))
        accuracies.append(field_acc)
        stats[f"calibrated_token_{field}"] = float(field_loss.detach().mean().item())
        stats[f"calibrated_token_{field}_acc"] = float(field_acc.detach().mean().item())

    loss_tensor = torch.stack(losses, dim=1)
    weight_tensor = torch.stack(weights).to(device=loss_tensor.device, dtype=loss_tensor.dtype)
    if float(weight_tensor.sum().detach().item()) <= 0.0:
        weight_tensor = torch.ones_like(weight_tensor)
    if reduction == "none":
        total = (loss_tensor * weight_tensor.view(1, -1)).sum(dim=1) / weight_tensor.sum().clamp_min(1e-6)
        mean_acc = torch.stack([acc.float().mean() for acc in accuracies]).mean()
        stats["calibrated_token"] = float(total.detach().mean().item())
    else:
        per_sample_total = (
            (loss_tensor * weight_tensor.view(1, -1)).sum(dim=1)
            / weight_tensor.sum().clamp_min(1e-6)
        )
        total = per_sample_total.mean()
        mean_acc = torch.stack([acc.float().mean() for acc in accuracies]).mean()
        stats["calibrated_token"] = float(total.detach().item())
    stats["calibrated_token_acc"] = float(mean_acc.detach().item())
    return total, stats


def _contract_direction_shape_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    tokens: torch.Tensor,
    route_scales: torch.Tensor | None = None,
    class_counts: torch.Tensor | None = None,
    class_balance_beta: float = 0.999,
    target_metrics: Dict[str, torch.Tensor] | None = None,
    pred_metrics: Dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pred_m = (
        pred_metrics
        if isinstance(pred_metrics, dict)
        else _calibrated_geometry_metrics(pred.float(), reference.float(), route_scales=route_scales)
    )
    target_m = (
        target_metrics
        if isinstance(target_metrics, dict)
        else _calibrated_geometry_metrics(target.float(), reference.float(), route_scales=route_scales)
    )
    field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("direction")
    direction_ids = tokens[:, field_idx].long()
    weights = pred.new_ones(direction_ids.shape, dtype=pred.dtype)
    if class_counts is not None:
        counts = class_counts.to(device=pred.device, dtype=pred.dtype).clamp_min(1.0)
        beta = min(max(float(class_balance_beta), 0.0), 0.999999)
        if beta > 0.0:
            effective_num = 1.0 - torch.pow(counts.new_full(counts.shape, beta), counts)
            class_weight = (1.0 - beta) / effective_num.clamp_min(1e-8)
        else:
            class_weight = 1.0 / counts
        class_weight = class_weight / class_weight.mean().clamp_min(1e-8)
        weights = class_weight.gather(0, direction_ids).to(dtype=pred.dtype)

    def smooth_l1_vec(value: torch.Tensor, target_value: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        diff_loss = F.smooth_l1_loss(value / max(float(scale), 1e-6), target_value / max(float(scale), 1e-6), reduction="none")
        return diff_loss.to(dtype=pred.dtype)

    angle_diff = safe_atan2(
        torch.sin(pred_m["contract_net_heading_rad"] - target_m["contract_net_heading_rad"]),
        torch.cos(pred_m["contract_net_heading_rad"] - target_m["contract_net_heading_rad"]),
    )
    zero_angle = torch.zeros_like(angle_diff)
    net_loss = smooth_l1_vec(angle_diff, zero_angle, scale=math.pi)
    total_abs_loss = smooth_l1_vec(
        pred_m["contract_total_abs_heading_rad"],
        target_m["contract_total_abs_heading_rad"],
        scale=math.pi,
    )
    turn_count_loss = smooth_l1_vec(
        pred_m["contract_soft_turn_count"],
        target_m["contract_soft_turn_count"],
        scale=3.0,
    )
    progress_loss = smooth_l1_vec(
        pred_m["contract_displacement_ratio"],
        target_m["contract_displacement_ratio"],
        scale=1.0,
    )
    lateral_loss = smooth_l1_vec(
        pred_m["mean_signed_lateral_offset"],
        target_m["mean_signed_lateral_offset"],
        scale=0.10,
    )
    side_loss = 0.5 * (
        smooth_l1_vec(pred_m["left_side_ratio"], target_m["left_side_ratio"], scale=1.0)
        + smooth_l1_vec(pred_m["right_side_ratio"], target_m["right_side_ratio"], scale=1.0)
    )
    per_sample = (
        1.0 * net_loss
        + 0.8 * total_abs_loss
        + 0.8 * turn_count_loss
        + 0.4 * progress_loss
        + 0.8 * lateral_loss
        + 0.4 * side_loss
    ) * weights
    terms = {
        "net": net_loss,
        "turn": total_abs_loss,
        "count": turn_count_loss,
        "progress": progress_loss,
        "lateral": lateral_loss,
        "side": side_loss,
        "weights": weights,
    }
    return per_sample, terms


def contract_direction_shape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    tokens: torch.Tensor,
    route_scales: torch.Tensor | None = None,
    class_counts: torch.Tensor | None = None,
    class_balance_beta: float = 0.999,
    target_metrics: Dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Supervise macro direction descriptors against the GT trajectory."""
    per_sample, terms = _contract_direction_shape_per_sample(
        pred,
        target,
        reference,
        tokens,
        route_scales=route_scales,
        class_counts=class_counts,
        class_balance_beta=class_balance_beta,
        target_metrics=target_metrics,
        pred_metrics=None,
    )
    weights = terms["weights"]
    total = per_sample.sum() / weights.sum().clamp_min(1e-6)
    stats = {
        "contract_direction_shape": float(total.detach().item()),
        "contract_direction_shape_net": float(terms["net"].detach().mean().item()),
        "contract_direction_shape_turn": float(terms["turn"].detach().mean().item()),
        "contract_direction_shape_count": float(terms["count"].detach().mean().item()),
        "contract_direction_shape_progress": float(terms["progress"].detach().mean().item()),
        "contract_direction_shape_lateral": float(terms["lateral"].detach().mean().item()),
        "contract_direction_shape_side": float(terms["side"].detach().mean().item()),
    }
    return total.to(dtype=pred.dtype), stats


def candidate_contract_direction_shape_loss(
    candidates: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    tokens: torch.Tensor,
    route_scales: torch.Tensor | None = None,
    class_counts: torch.Tensor | None = None,
    class_balance_beta: float = 0.999,
    softmin_temperature: float = 0.08,
    target_metrics: Dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """MTR-style candidate-pool direction shape coverage loss."""
    if candidates.ndim != 4 or candidates.shape[1] <= 1:
        return contract_direction_shape_loss(
            candidates if candidates.ndim == 3 else candidates[:, 0],
            target,
            reference,
            tokens,
            route_scales=route_scales,
            class_counts=class_counts,
            class_balance_beta=class_balance_beta,
            target_metrics=target_metrics,
        )
    batch, num_modes, target_len, coord_dim = candidates.shape
    flat = candidates.reshape(batch * num_modes, target_len, coord_dim)
    target_rep = target.unsqueeze(1).expand(-1, num_modes, -1, -1).reshape(batch * num_modes, target.shape[-2], target.shape[-1])
    reference_rep = reference.unsqueeze(1).expand(-1, num_modes, -1, -1).reshape(batch * num_modes, reference.shape[-2], reference.shape[-1])
    tokens_rep = tokens.unsqueeze(1).expand(-1, num_modes, -1).reshape(batch * num_modes, tokens.shape[-1])
    scales_rep = None
    if route_scales is not None:
        scales_rep = route_scales.view(batch, 1).expand(-1, num_modes).reshape(-1)
    per_sample, _ = _contract_direction_shape_per_sample(
        flat,
        target_rep,
        reference_rep,
        tokens_rep,
        route_scales=scales_rep,
        class_counts=class_counts,
        class_balance_beta=class_balance_beta,
        target_metrics=(
            {
                key: value.unsqueeze(1).expand(-1, num_modes).reshape(batch * num_modes)
                for key, value in target_metrics.items()
            }
            if isinstance(target_metrics, dict)
            else None
        ),
        pred_metrics=(
            {
                key: value
                for key, value in pred_metrics.items()
            }
            if isinstance(pred_metrics, dict)
            else None
        ),
    )
    per_mode = per_sample.view(batch, num_modes)
    temp = max(float(softmin_temperature), 1e-6)
    weights = torch.softmax(-per_mode / temp, dim=1)
    softbest = (weights * per_mode).sum(dim=1)
    best = per_mode.min(dim=1).values
    total = softbest.mean()
    stats = {
        "candidate_contract_direction_shape": float(total.detach().item()),
        "candidate_contract_direction_shape_best": float(best.detach().mean().item()),
        "candidate_contract_direction_shape_mean": float(per_mode.detach().mean().item()),
    }
    return total.to(dtype=candidates.dtype), stats


def _direction_rule_loss_per_sample(
    pred: torch.Tensor,
    reference: torch.Tensor,
    tokens: torch.Tensor,
    route_scales: torch.Tensor | None = None,
    anti_uturn_weight: float = 1.0,
    metrics: Dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    metrics = (
        metrics
        if isinstance(metrics, dict)
        else _calibrated_geometry_metrics(pred.float(), reference.float(), route_scales=route_scales)
    )
    field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("direction")
    target = tokens[:, field_idx].long()
    ids = DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]
    dtype = pred.dtype

    # Keep generator-side candidate coverage aligned with the official Table-3
    # OD-chord direction contract. Left/right are signed-turn semantics; lateral
    # side is only a weak auxiliary inside _contract_direction_all_token_scores.
    score_by_id = _contract_direction_all_token_scores(metrics, dtype=dtype)
    per_sample = score_by_id.gather(1, target.view(-1, 1)).squeeze(1)

    total_abs = metrics["contract_total_abs_heading_rad"].to(dtype=dtype)
    turn_count = metrics["contract_soft_turn_count"].to(dtype=dtype)
    displacement_ratio = metrics["contract_displacement_ratio"].to(dtype=dtype)
    left_mask = target == int(ids["left_turn"])
    right_mask = target == int(ids["right_turn"])
    single_turn_mask = left_mask | right_mask
    anti_uturn = (
        F.relu(displacement_ratio.new_tensor(0.35) - displacement_ratio) / 0.35
        + F.relu(total_abs - math.radians(150.0)) / math.radians(150.0)
        + F.relu(turn_count - 2.0) / 2.0
    )
    per_sample = per_sample + single_turn_mask.to(dtype=dtype) * float(anti_uturn_weight) * anti_uturn
    stats = {
        "best_score": per_sample,
        "anti_uturn": anti_uturn * single_turn_mask.to(dtype=dtype),
        "u_strength": -score_by_id[:, int(ids["u_turn"])],
        "multi_strength": -score_by_id[:, int(ids["multi_turn"])],
        "left_strength": -score_by_id[:, int(ids["left_turn"])],
        "right_strength": -score_by_id[:, int(ids["right_turn"])],
    }
    return per_sample, stats


def candidate_direction_rule_coverage_loss(
    candidates: torch.Tensor,
    reference: torch.Tensor,
    tokens: torch.Tensor,
    route_scales: torch.Tensor | None = None,
    softmin_temperature: float = 0.08,
    anti_uturn_weight: float = 1.0,
    metrics: Dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Soft-best candidate coverage for the official OD-chord direction rule.

    Unlike candidate_contract_direction_shape_loss, this objective does not
    ask a mode to match the full GT trajectory shape.  It asks the K-candidate
    pool to contain at least one trajectory compatible with the target
    direction token under the Table-3 direction contract, with an explicit
    anti-U-turn guard for left/right single-turn instructions.
    """
    if candidates.ndim != 4 or candidates.shape[1] <= 1:
        sample = candidates if candidates.ndim == 3 else candidates[:, 0]
        per_sample, term_stats = _direction_rule_loss_per_sample(
            sample,
            reference,
            tokens,
            route_scales=route_scales,
            anti_uturn_weight=anti_uturn_weight,
            metrics=metrics,
        )
        total = per_sample.mean()
        return total.to(dtype=sample.dtype), {
            "candidate_direction_rule_coverage": float(total.detach().item()),
            "candidate_direction_rule_coverage_best": float(per_sample.detach().mean().item()),
            "candidate_direction_rule_coverage_mean": float(per_sample.detach().mean().item()),
            "candidate_direction_rule_anti_uturn": float(term_stats["anti_uturn"].detach().mean().item()),
        }

    batch, num_modes, target_len, coord_dim = candidates.shape
    flat = candidates.reshape(batch * num_modes, target_len, coord_dim)
    reference_rep = reference.unsqueeze(1).expand(-1, num_modes, -1, -1).reshape(
        batch * num_modes,
        reference.shape[-2],
        reference.shape[-1],
    )
    tokens_rep = tokens.unsqueeze(1).expand(-1, num_modes, -1).reshape(batch * num_modes, tokens.shape[-1])
    scales_rep = None
    if route_scales is not None:
        scales_rep = route_scales.view(batch, 1).expand(-1, num_modes).reshape(-1)
    per_sample, term_stats = _direction_rule_loss_per_sample(
        flat,
        reference_rep,
        tokens_rep,
        route_scales=scales_rep,
        anti_uturn_weight=anti_uturn_weight,
        metrics=metrics,
    )
    per_mode = per_sample.view(batch, num_modes)
    temp = max(float(softmin_temperature), 1e-6)
    weights = torch.softmax(-per_mode / temp, dim=1)
    softbest = (weights * per_mode).sum(dim=1)
    best = per_mode.min(dim=1).values
    best_idx = per_mode.argmin(dim=1)
    anti_uturn = term_stats["anti_uturn"].view(batch, num_modes)
    total = softbest.mean()
    stats = {
        "candidate_direction_rule_coverage": float(total.detach().item()),
        "candidate_direction_rule_coverage_best": float(best.detach().mean().item()),
        "candidate_direction_rule_coverage_mean": float(per_mode.detach().mean().item()),
        "candidate_direction_rule_anti_uturn": float(
            anti_uturn.gather(1, best_idx.view(-1, 1)).squeeze(1).detach().mean().item()
        ),
    }
    return total.to(dtype=candidates.dtype), stats


def imotion_direction_loss(
    pred: torch.Tensor,
    tokens: torch.Tensor,
    straight_heading_rad: float = math.pi / 6.0,
    straight_lateral: float = 0.08,
    turn_margin_rad: float = 0.35,
    uturn_heading_rad: float = 1.50,
    uturn_longitudinal: float = -0.05,
    multiturn_total_abs_rad: float = 1.20,
    multiturn_soft_count: float = 2.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Instruction-direction loss using iMotion-style local-frame descriptors.

    This complements calibration CE. Calibration compares each class to a single
    scalar center; this loss encodes the expected local-frame inequalities for
    straight, left/right, U-turn, and multi-turn behavior.
    """
    if pred.shape[1] < 3:
        zero = pred.new_tensor(0.0)
        return zero, {"imotion_direction": 0.0, "imotion_direction_acc_proxy": 0.0}

    desc = _local_direction_descriptors(pred.float())
    field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("direction")
    target = tokens[:, field_idx].long()
    ids = DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]
    losses: List[torch.Tensor] = []

    def hinge(value: torch.Tensor, scale: float) -> torch.Tensor:
        normalized = F.relu(value) / max(float(scale), 1e-6)
        return F.smooth_l1_loss(normalized, torch.zeros_like(normalized), reduction="none")

    def add(mask: torch.Tensor, values: List[torch.Tensor]) -> None:
        if bool(mask.any()):
            losses.append(torch.stack([value[mask].mean() for value in values]).mean())

    abs_heading = desc["heading_diff"].abs()
    local_dy = desc["local_dy"]
    local_dx = desc["local_dx"]
    signed_turn = desc["signed_total_turn"]
    total_abs = desc["total_abs_turn"]
    turn_count = desc["soft_turn_count"]

    straight = target == int(ids["straight"])
    add(
        straight,
        [
            hinge(abs_heading - float(straight_heading_rad), straight_heading_rad),
            hinge(local_dy.abs() - float(straight_lateral), straight_lateral),
            hinge(total_abs - float(multiturn_total_abs_rad), multiturn_total_abs_rad) * 0.25,
        ],
    )

    left = target == int(ids["left_turn"])
    add(
        left,
        [
            hinge(float(turn_margin_rad) - signed_turn, turn_margin_rad),
            hinge(-local_dy, straight_lateral),
        ],
    )

    right = target == int(ids["right_turn"])
    add(
        right,
        [
            hinge(signed_turn + float(turn_margin_rad), turn_margin_rad),
            hinge(local_dy, straight_lateral),
        ],
    )

    u_turn = target == int(ids["u_turn"])
    add(
        u_turn,
        [
            hinge(float(uturn_heading_rad) - abs_heading, uturn_heading_rad),
            hinge(local_dx - float(uturn_longitudinal), max(abs(float(uturn_longitudinal)), 0.05)) * 0.25,
        ],
    )

    multi_turn = target == int(ids["multi_turn"])
    add(
        multi_turn,
        [
            hinge(float(multiturn_total_abs_rad) - total_abs, multiturn_total_abs_rad),
            hinge(float(multiturn_soft_count) - turn_count, multiturn_soft_count) * 0.25,
        ],
    )

    if not losses:
        total = pred.new_tensor(0.0)
    else:
        total = torch.stack(losses).mean().to(dtype=pred.dtype)

    pred_labels = torch.full_like(target, fill_value=int(ids["multi_turn"]))
    pred_labels = torch.where(
        (abs_heading < float(straight_heading_rad)) & (local_dy.abs() < float(straight_lateral)),
        torch.full_like(pred_labels, int(ids["straight"])),
        pred_labels,
    )
    pred_labels = torch.where(
        (signed_turn > float(turn_margin_rad)) & (local_dy > 0),
        torch.full_like(pred_labels, int(ids["left_turn"])),
        pred_labels,
    )
    pred_labels = torch.where(
        (signed_turn < -float(turn_margin_rad)) & (local_dy < 0),
        torch.full_like(pred_labels, int(ids["right_turn"])),
        pred_labels,
    )
    pred_labels = torch.where(
        (abs_heading > float(uturn_heading_rad)) & (local_dx < float(uturn_longitudinal)),
        torch.full_like(pred_labels, int(ids["u_turn"])),
        pred_labels,
    )

    stats = {
        "imotion_direction": float(total.detach().item()),
        "imotion_direction_acc_proxy": float((pred_labels == target).float().mean().detach().item()),
        "imotion_direction_abs_heading": float(abs_heading.detach().mean().item()),
        "imotion_direction_total_abs_turn": float(total_abs.detach().mean().item()),
        "imotion_direction_soft_turn_count": float(turn_count.detach().mean().item()),
        "imotion_direction_local_dy_abs": float(local_dy.detach().abs().mean().item()),
    }
    return total, stats


def _positive_part_tensor(value: torch.Tensor) -> torch.Tensor:
    return F.relu(value)


def _direction_rule_all_token_scores(
    metrics: Dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Differentiable version of eval.instruction_geometry.direction_rule_scores.

    Table-4 evaluation treats direction as a composite rule over local-frame
    heading, lateral movement, turn count, and total turning.  Using the same
    score family in the loss keeps counterfactual direction training aligned
    with the metric that decides success.
    """
    cfg = DIRECTION_RULE_DEFAULTS
    sample = next(iter(metrics.values()))
    device = sample.device
    signed_heading = metrics.get("local_heading_diff_rad", metrics.get("signed_net_heading_rad", sample.new_zeros(sample.shape)))
    abs_heading = metrics.get("local_abs_heading_diff_rad", signed_heading.abs()).abs().to(dtype=dtype)
    signed_turn = metrics.get("local_signed_total_turn_rad", metrics.get("signed_total_turn_rad", sample.new_zeros(sample.shape))).to(dtype=dtype)
    total_abs = metrics.get("local_total_abs_turn_rad", metrics.get("total_abs_heading_rad", sample.new_zeros(sample.shape))).to(dtype=dtype)
    max_abs_turn = metrics.get("local_max_abs_turn_rad", metrics.get("max_abs_turn_rad", sample.new_zeros(sample.shape))).to(dtype=dtype)
    turn_count = metrics.get("local_turn_count", metrics.get("turn_count", sample.new_zeros(sample.shape))).to(dtype=dtype)
    local_dx = metrics.get("local_dx", sample.new_zeros(sample.shape)).to(dtype=dtype)
    local_dy = metrics.get(
        "local_dy",
        metrics.get("mean_signed_lateral_offset", sample.new_zeros(sample.shape)),
    ).to(dtype=dtype)
    left_ratio = metrics.get("local_left_turn_ratio", metrics.get("left_turn_ratio", sample.new_zeros(sample.shape))).to(dtype=dtype)
    right_ratio = metrics.get("local_right_turn_ratio", metrics.get("right_turn_ratio", sample.new_zeros(sample.shape))).to(dtype=dtype)

    straight_heading = max(float(cfg["straight_heading_rad"]), 1e-6)
    straight_lateral = max(float(cfg["straight_lateral"]), 1e-6)
    straight_total_abs = max(float(cfg["straight_total_abs_rad"]), 1e-6)
    turn_margin = max(float(cfg["turn_margin_rad"]), 1e-6)
    uturn_heading = max(float(cfg["uturn_heading_rad"]), 1e-6)
    uturn_longitudinal = float(cfg["uturn_longitudinal"])
    uturn_max_abs = max(float(cfg["uturn_max_abs_turn_rad"]), 1e-6)
    multiturn_abs = max(float(cfg["multiturn_total_abs_rad"]), 1e-6)
    multiturn_count = max(float(cfg["multiturn_count"]), 1e-6)

    left_strength = torch.maximum(torch.maximum(local_dy / straight_lateral, signed_turn / turn_margin), left_ratio)
    right_strength = torch.maximum(torch.maximum(-local_dy / straight_lateral, -signed_turn / turn_margin), right_ratio)
    uturn_strength = torch.maximum(abs_heading / uturn_heading, max_abs_turn / uturn_max_abs)
    uturn_backward = 1.0 + (uturn_longitudinal - local_dx) / 0.20
    uturn_strength = torch.where(local_dx < uturn_longitudinal, torch.maximum(uturn_strength, uturn_backward), uturn_strength)
    multiturn_strength = torch.maximum(total_abs / multiturn_abs, turn_count / multiturn_count)
    left_clear = left_strength.clamp(min=0.0, max=1.0)
    right_clear = right_strength.clamp(min=0.0, max=1.0)
    directional_multiturn_penalty = _positive_part_tensor(torch.maximum(left_strength, right_strength) - 1.0)

    straight = (
        abs_heading / straight_heading
        + local_dy.abs() / straight_lateral
        + _positive_part_tensor(total_abs - straight_total_abs) / straight_total_abs
        + _positive_part_tensor(turn_count - 1.0)
        + 0.03 * total_abs / multiturn_abs
    )
    left = (
        _positive_part_tensor(1.0 - left_strength)
        + 0.35 * _positive_part_tensor(multiturn_strength - 1.0) * (1.0 - 0.70 * left_clear)
        + 0.25 * _positive_part_tensor(uturn_strength - 1.0)
        - 0.03 * signed_turn / turn_margin
        - 0.02 * local_dy / straight_lateral
        - 0.02 * (left_ratio - right_ratio)
    )
    right = (
        _positive_part_tensor(1.0 - right_strength)
        + 0.35 * _positive_part_tensor(multiturn_strength - 1.0) * (1.0 - 0.70 * right_clear)
        + 0.25 * _positive_part_tensor(uturn_strength - 1.0)
        + 0.03 * signed_turn / turn_margin
        + 0.02 * local_dy / straight_lateral
        - 0.02 * (right_ratio - left_ratio)
    )
    u_turn = (
        _positive_part_tensor(1.0 - uturn_strength)
        + 0.25 * _positive_part_tensor(multiturn_strength - 1.0)
        - 0.03 * uturn_strength
    )
    multi_turn = (
        _positive_part_tensor(1.0 - multiturn_strength)
        + 0.20 * _positive_part_tensor(uturn_strength - 1.0)
        + 0.45 * directional_multiturn_penalty
        - 0.03 * multiturn_strength
    )

    ids = DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]
    ordered = torch.empty(
        (sample.shape[0], len(ids)),
        dtype=dtype,
        device=device,
    )
    ordered[:, int(ids["straight"])] = straight
    ordered[:, int(ids["left_turn"])] = left
    ordered[:, int(ids["right_turn"])] = right
    ordered[:, int(ids["u_turn"])] = u_turn
    ordered[:, int(ids["multi_turn"])] = multi_turn
    return ordered


def _contract_direction_all_token_scores(
    metrics: Dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Differentiable score family for the official OD-chord direction labels."""
    sample = next(iter(metrics.values()))
    device = sample.device
    net = metrics.get("contract_net_heading_rad", metrics.get("signed_net_heading_rad", sample.new_zeros(sample.shape))).to(dtype=dtype)
    abs_net = metrics.get("contract_abs_net_heading_rad", net.abs()).abs().to(dtype=dtype)
    total_abs = metrics.get("contract_total_abs_heading_rad", metrics.get("total_abs_heading_rad", sample.new_zeros(sample.shape))).to(dtype=dtype)
    turn_count = metrics.get("contract_soft_turn_count", metrics.get("turn_count", sample.new_zeros(sample.shape))).to(dtype=dtype)
    displacement_ratio = metrics.get("contract_displacement_ratio", metrics.get("displacement_ratio", sample.new_zeros(sample.shape))).to(dtype=dtype)
    signed_turn = metrics.get("local_signed_total_turn_rad", metrics.get("signed_total_turn_rad", net)).to(dtype=dtype)
    local_dy = metrics.get("local_dy", sample.new_zeros(sample.shape)).to(dtype=dtype)
    left_ratio = metrics.get("local_left_turn_ratio", metrics.get("left_turn_ratio", sample.new_zeros(sample.shape))).to(dtype=dtype)
    right_ratio = metrics.get("local_right_turn_ratio", metrics.get("right_turn_ratio", sample.new_zeros(sample.shape))).to(dtype=dtype)

    deg35 = math.radians(35.0)
    deg135 = math.radians(135.0)
    deg120 = math.radians(120.0)
    eps = 1e-6

    u_heading_strength = torch.minimum(
        abs_net / max(deg135, eps),
        torch.relu(0.55 - displacement_ratio) / 0.55,
    )
    u_loop_strength = torch.minimum(
        torch.relu(0.25 - displacement_ratio) / 0.25,
        total_abs / max(deg120, eps),
    )
    u_strength = torch.maximum(u_heading_strength, u_loop_strength)
    multi_strength = torch.maximum(
        turn_count / 3.0,
        total_abs / max(deg120, eps),
    )
    left_strength = torch.maximum(
        torch.relu(net) / max(deg35, eps),
        torch.maximum(
            torch.relu(signed_turn) / max(deg35, eps),
            torch.maximum(0.35 * torch.relu(local_dy - 0.02) / 0.12, left_ratio),
        ),
    )
    right_strength = torch.maximum(
        torch.relu(-net) / max(deg35, eps),
        torch.maximum(
            torch.relu(-signed_turn) / max(deg35, eps),
            torch.maximum(0.35 * torch.relu(-local_dy - 0.02) / 0.12, right_ratio),
        ),
    )
    left_clear = left_strength.clamp(min=0.0, max=1.0)
    right_clear = right_strength.clamp(min=0.0, max=1.0)
    directional_strength = torch.maximum(left_strength, right_strength)

    straight = (
        abs_net / deg35
        + 0.4 * total_abs / max(deg120, eps)
        + torch.relu(turn_count - 1.0)
        + 0.4 * torch.maximum(left_strength, right_strength)
    )
    left = (
        torch.relu(1.0 - left_strength)
        + 0.5 * torch.relu(multi_strength - 1.0) * (1.0 - 0.75 * left_clear)
        + 0.5 * torch.relu(u_strength - 1.0)
        - 0.08 * left_strength
    )
    right = (
        torch.relu(1.0 - right_strength)
        + 0.5 * torch.relu(multi_strength - 1.0) * (1.0 - 0.75 * right_clear)
        + 0.5 * torch.relu(u_strength - 1.0)
        - 0.08 * right_strength
    )
    u_turn = (
        torch.relu(1.0 - u_strength)
        + 0.35 * torch.relu(multi_strength - 1.0)
        - 0.08 * u_strength
    )
    multi_turn = (
        torch.relu(1.0 - multi_strength)
        + 0.35 * torch.relu(u_strength - 1.0)
        + 0.25 * multi_strength * directional_strength.clamp(min=0.0, max=1.0)
        + 0.75 * torch.relu(directional_strength - 1.0)
        - 0.08 * multi_strength
    )

    ids = DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]
    ordered = torch.empty((sample.shape[0], len(ids)), dtype=dtype, device=device)
    ordered[:, int(ids["straight"])] = straight
    ordered[:, int(ids["left_turn"])] = left
    ordered[:, int(ids["right_turn"])] = right
    ordered[:, int(ids["u_turn"])] = u_turn
    ordered[:, int(ids["multi_turn"])] = multi_turn
    return ordered


def _raw_field_token_score(
    metrics: Dict[str, torch.Tensor],
    field_calib: dict,
    token_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    centers = field_calib["centers"].to(device=token_ids.device, dtype=dtype)
    scales = field_calib["scales"].to(device=token_ids.device, dtype=dtype).clamp_min(1e-6)
    values: List[torch.Tensor] = []
    for sample_idx, token_id in enumerate(token_ids.long()):
        idx = int(token_id.detach().item())
        metric_name = field_calib["metrics"][idx]
        if metric_name not in metrics:
            raise ValueError(f"Calibration metric '{metric_name}' is not available.")
        value = metrics[metric_name][sample_idx].to(dtype=dtype)
        values.append((value - centers[idx]).abs() / scales[idx])
    return torch.stack(values, dim=0)


def _raw_field_all_token_scores(
    metrics: Dict[str, torch.Tensor],
    field_calib: dict,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_tokens = int(field_calib["centers"].numel())
    batch_size = next(iter(metrics.values())).shape[0]
    scores: List[torch.Tensor] = []
    for token_idx in range(num_tokens):
        token_ids = torch.full(
            (batch_size,),
            int(token_idx),
            dtype=torch.long,
            device=next(iter(metrics.values())).device,
        )
        scores.append(_raw_field_token_score(metrics, field_calib, token_ids, dtype=dtype))
    return torch.stack(scores, dim=1)


def _field_token_score(
    metrics: Dict[str, torch.Tensor],
    field_calib: dict,
    token_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if str(field_calib.get("field", "")) == "direction":
        all_scores = _contract_direction_all_token_scores(metrics, dtype=dtype)
        return all_scores.gather(1, token_ids.long().view(-1, 1)).squeeze(1)
    return _raw_field_token_score(metrics, field_calib, token_ids, dtype=dtype)


def _field_all_token_scores(
    metrics: Dict[str, torch.Tensor],
    field_calib: dict,
    dtype: torch.dtype,
) -> torch.Tensor:
    if str(field_calib.get("field", "")) == "direction":
        return _contract_direction_all_token_scores(metrics, dtype=dtype)
    return _raw_field_all_token_scores(metrics, field_calib, dtype=dtype)


def _calibration_metrics_available(metrics: Dict[str, torch.Tensor], field_calib: dict) -> bool:
    if str(field_calib.get("field", "")) == "direction":
        return True
    return all(str(metric_name) in metrics for metric_name in field_calib.get("metrics", []))


def _field_transition_metric_delta(
    base_metrics: Dict[str, torch.Tensor],
    cf_metrics: Dict[str, torch.Tensor],
    field_calib: dict,
    source_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Signed metric movement from source token toward target token.

    Table-4 evaluation reports whether the counterfactual output moves the
    target geometry metric in the expected direction. The calibrated
    target-score loss is necessary but not sufficient for that: a trajectory
    can reduce distance-to-target without a stable signed move.

    The stable contract is base-aware: use the target token's metric, then
    reward movement from the actual base value toward the target token center.
    This is important for cross-metric direction transitions such as
    multi_turn -> straight, where the target metric should usually decrease
    rather than blindly increase.

    Returns:
        raw_delta: signed movement in the metric's native units, matching
            Table-4's `signed_target_metric_delta`.
        normalized_delta: raw_delta divided by the target token scale for a
            field-comparable training margin.
        valid: whether the target metric is available.
    """
    if str(field_calib.get("field", "")) == "direction":
        base_target_score = _field_token_score(base_metrics, field_calib, target_token_ids, dtype=dtype)
        cf_target_score = _field_token_score(cf_metrics, field_calib, target_token_ids, dtype=dtype)
        signed_delta = base_target_score - cf_target_score
        valid = torch.ones_like(signed_delta, dtype=torch.bool)
        return signed_delta, signed_delta, valid

    device = source_token_ids.device
    centers = field_calib["centers"].to(device=device, dtype=dtype)
    scales = field_calib["scales"].to(device=device, dtype=dtype).clamp_min(1e-6)
    source_ids = source_token_ids.long().to(device=device)
    target_ids = target_token_ids.long().to(device=device)
    raw_delta = torch.zeros_like(target_ids, dtype=dtype)
    normalized_delta = torch.zeros_like(target_ids, dtype=dtype)
    valid = torch.zeros_like(target_ids, dtype=torch.bool)
    metric_names = list(field_calib["metrics"])
    for tgt_idx, metric_name in enumerate(metric_names):
        if metric_name not in base_metrics or metric_name not in cf_metrics:
            continue
        mask = target_ids == int(tgt_idx)
        if not bool(mask.any()):
            continue
        base_value = base_metrics[metric_name].to(device=device, dtype=dtype)[mask]
        cf_value = cf_metrics[metric_name].to(device=device, dtype=dtype)[mask]
        center_delta = centers[int(tgt_idx)] - base_value
        fallback_delta = centers[int(tgt_idx)] - centers[source_ids[mask]]
        center_delta = torch.where(center_delta.abs() < 1e-8, fallback_delta, center_delta)
        direction = torch.where(
            center_delta >= 0,
            torch.ones_like(center_delta),
            -torch.ones_like(center_delta),
        )
        signed_raw_delta = direction * (cf_value - base_value)
        raw_delta[mask] = signed_raw_delta
        normalized_delta[mask] = signed_raw_delta / scales[int(tgt_idx)]
        valid[mask] = True
    return raw_delta, normalized_delta, valid


def _field_collateral_metric_delta(
    base_metrics: Dict[str, torch.Tensor],
    cf_metrics: Dict[str, torch.Tensor],
    field_calib: dict,
    token_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalized non-target metric drift used by Table-4 collateral.

    For each sample, use the metric attached to its original token and measure
    how far the counterfactual trajectory moved away from the base trajectory
    in that metric, normalized by the token's calibration scale.
    """
    device = token_ids.device
    scales = field_calib["scales"].to(device=device, dtype=dtype).clamp_min(1e-6)
    token_ids = token_ids.long().to(device=device)
    values = torch.zeros_like(token_ids, dtype=dtype)
    for token_idx, metric_name in enumerate(field_calib["metrics"]):
        if metric_name not in base_metrics or metric_name not in cf_metrics:
            continue
        mask = token_ids == int(token_idx)
        if not bool(mask.any()):
            continue
        base_value = base_metrics[metric_name].to(device=device, dtype=dtype)[mask]
        cf_value = cf_metrics[metric_name].to(device=device, dtype=dtype)[mask]
        values[mask] = (cf_value - base_value).abs() / scales[int(token_idx)]
    return values


def direction_structural_collateral_loss(
    base_pred: torch.Tensor,
    cf_pred: torch.Tensor,
    reference: torch.Tensor,
    base_tokens: torch.Tensor,
    calibration: dict | None,
    fields: Iterable[str] | None = None,
    field_weights: Dict[str, float] | None = None,
    margin_z: float = 0.08,
    route_scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Guard direction edits against raw structural drift.

    Table-4 independent collateral intentionally excludes distance/progress/
    reference-adherence for direction edits under the OD-chord protocol because
    these fields are structurally coupled.  For training, however, unconstrained
    direction edits can exploit that exclusion by changing route scale or
    reference distance.  This auxiliary loss is a training-only guard: it keeps
    selected structural fields close to the base prediction while the ordinary
    direction objective handles target direction movement.
    """
    requested_fields = [
        str(field)
        for field in (fields or ("distance", "progress", "reference_adherence"))
        if str(field) in DEFAULT_MOBILITY_VOCAB.fields and str(field) != "direction"
    ]
    if calibration is None or not requested_fields:
        zero = cf_pred.new_tensor(0.0)
        return zero, {
            "direction_structural_collateral": 0.0,
            "direction_structural_collateral_drift": 0.0,
            "direction_structural_collateral_excess": 0.0,
            "direction_structural_collateral_feasible": 0.0,
            "direction_structural_collateral_active_fields": 0.0,
        }

    base_metrics = _calibrated_geometry_metrics(base_pred.float(), reference.float(), route_scales=route_scales)
    cf_metrics = _calibrated_geometry_metrics(cf_pred.float(), reference.float(), route_scales=route_scales)
    field_index = {field: idx for idx, field in enumerate(DEFAULT_MOBILITY_VOCAB.fields)}
    weights = field_weights or {}
    margin = float(margin_z)

    loss_terms: List[torch.Tensor] = []
    drift_terms: List[torch.Tensor] = []
    per_sample_drifts: List[torch.Tensor] = []
    total_weight = cf_pred.new_tensor(0.0)
    active_fields = 0
    for field in requested_fields:
        field_calib = calibration.get(field) if isinstance(calibration, dict) else None
        if field_calib is None or not _calibration_metrics_available(base_metrics, field_calib):
            continue
        weight_value = float(weights.get(field, 1.0))
        if weight_value <= 0.0:
            continue
        token_ids = base_tokens[:, field_index[field]].to(device=cf_pred.device)
        drift = _field_collateral_metric_delta(
            base_metrics,
            cf_metrics,
            field_calib,
            token_ids,
            dtype=cf_pred.dtype,
        )
        weight = cf_pred.new_tensor(weight_value)
        loss_terms.append(weight * F.relu(drift - margin).mean())
        drift_terms.append(weight * drift.mean())
        per_sample_drifts.append(drift)
        total_weight = total_weight + weight
        active_fields += 1

    if not loss_terms:
        zero = cf_pred.new_tensor(0.0)
        return zero, {
            "direction_structural_collateral": 0.0,
            "direction_structural_collateral_drift": 0.0,
            "direction_structural_collateral_excess": 0.0,
            "direction_structural_collateral_feasible": 0.0,
            "direction_structural_collateral_active_fields": 0.0,
        }

    denom = total_weight.clamp_min(1e-6)
    loss = torch.stack(loss_terms).sum() / denom
    drift_mean = torch.stack(drift_terms).sum() / denom
    sample_mean_drift = torch.stack(per_sample_drifts, dim=0).mean(dim=0)
    feasible = (sample_mean_drift <= margin).to(dtype=cf_pred.dtype).mean()
    stats = {
        "direction_structural_collateral": float(loss.detach().item()),
        "direction_structural_collateral_drift": float(drift_mean.detach().item()),
        "direction_structural_collateral_excess": float(loss.detach().item()),
        "direction_structural_collateral_feasible": float(feasible.detach().item()),
        "direction_structural_collateral_active_fields": float(active_fields),
    }
    return loss, stats


def _field_signed_metric_delta(
    base_metrics: Dict[str, torch.Tensor],
    cf_metrics: Dict[str, torch.Tensor],
    field_calib: dict,
    source_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible normalized signed movement used by older callers."""
    _, normalized_delta, valid = _field_transition_metric_delta(
        base_metrics,
        cf_metrics,
        field_calib,
        source_token_ids,
        target_token_ids,
        dtype=dtype,
    )
    return normalized_delta, valid


def calibration_counterfactual_control_loss(
    base_pred: torch.Tensor,
    cf_pred: torch.Tensor,
    reference: torch.Tensor,
    base_tokens: torch.Tensor,
    cf_tokens: torch.Tensor,
    changed_field: str,
    calibration: dict | None,
    target_margin: float = 0.05,
    collateral_margin: float = 0.05,
    target_weight: float = 1.0,
    collateral_weight: float = 1.0,
    token_margin: float = 0.05,
    token_margin_weight: float = 0.0,
    token_boundary_margin: float = 0.05,
    token_boundary_weight: float = 0.0,
    signed_metric_margin: float = 0.0,
    signed_metric_weight: float = 0.0,
    success_margin: float = 0.0,
    success_weight: float = 0.0,
    success_focal_gamma: float = 0.0,
    target_effect_cap: float = 0.0,
    target_effect_cap_weight: float = 0.0,
    sample_weights: torch.Tensor | None = None,
    collateral_mode: str = "score",
    base_metrics: Dict[str, torch.Tensor] | None = None,
    cf_metrics: Dict[str, torch.Tensor] | None = None,
    return_stats: bool = True,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Paired Table-4-style controllability loss.

    The target term pushes the counterfactual trajectory closer to the changed
    token's calibrated geometry than the base trajectory. The optional token
    margin term requires the counterfactual geometry to classify closer to the
    target token than to the source token. The collateral term keeps all
    unchanged token-field geometry near the base prediction.
    """
    if calibration is None:
        zero = cf_pred.new_tensor(0.0)
        return zero, {
            "cf_target_effect": 0.0,
            "cf_target_loss": 0.0,
            "cf_token_margin_effect": 0.0,
            "cf_token_margin_loss": 0.0,
            "cf_token_margin_success": 0.0,
            "cf_token_boundary_effect": 0.0,
            "cf_token_boundary_loss": 0.0,
            "cf_token_boundary_success": 0.0,
            "cf_signed_metric_delta": 0.0,
            "cf_signed_target_metric_delta": 0.0,
            "cf_signed_metric_loss": 0.0,
            "cf_signed_metric_success": 0.0,
            "cf_signed_metric_valid_rate": 0.0,
            "cf_success_loss": 0.0,
            "cf_target_effect_cap_loss": 0.0,
            "cf_collateral": 0.0,
            "cf_success": 0.0,
            "cf_sample_weight": 0.0,
        }
    if changed_field not in DEFAULT_MOBILITY_VOCAB.fields:
        raise ValueError(f"Unknown counterfactual field: {changed_field}")

    def _weighted_mean(values: torch.Tensor) -> torch.Tensor:
        if sample_weights is None:
            return values.mean()
        weights = sample_weights.to(device=values.device, dtype=values.dtype).view(-1)
        if weights.numel() != values.numel():
            raise ValueError(
                f"sample_weights has {weights.numel()} values but loss tensor has {values.numel()} values."
            )
        weights = weights.clamp_min(0.0)
        denom = weights.sum().clamp_min(1e-6)
        return (values * weights).sum() / denom

    base_metrics = (
        base_metrics
        if isinstance(base_metrics, dict)
        else _calibrated_geometry_metrics(base_pred.float(), reference.float())
    )
    cf_metrics = (
        cf_metrics
        if isinstance(cf_metrics, dict)
        else _calibrated_geometry_metrics(cf_pred.float(), reference.float())
    )
    field_index = {field: idx for idx, field in enumerate(DEFAULT_MOBILITY_VOCAB.fields)}
    field_idx = field_index[changed_field]
    source_token_ids = base_tokens[:, field_idx]
    target_token_ids = cf_tokens[:, field_idx]
    if not _calibration_metrics_available(base_metrics, calibration[changed_field]):
        zero = cf_pred.new_tensor(0.0)
        return zero, {
            "cf_target_effect": 0.0,
            "cf_target_loss": 0.0,
            "cf_token_margin_effect": 0.0,
            "cf_token_margin_loss": 0.0,
            "cf_token_margin_success": 0.0,
            "cf_token_boundary_effect": 0.0,
            "cf_token_boundary_loss": 0.0,
            "cf_token_boundary_success": 0.0,
            "cf_signed_metric_delta": 0.0,
            "cf_signed_target_metric_delta": 0.0,
            "cf_signed_metric_loss": 0.0,
            "cf_signed_metric_success": 0.0,
            "cf_signed_metric_valid_rate": 0.0,
            "cf_success_loss": 0.0,
            "cf_target_effect_cap_loss": 0.0,
            "cf_collateral": 0.0,
            "cf_success": 0.0,
            "cf_sample_weight": 0.0,
            "cf_field": changed_field,
            "cf_skipped_unavailable_metric": 1.0,
        }

    base_target_score = _field_token_score(
        base_metrics,
        calibration[changed_field],
        target_token_ids,
        dtype=cf_pred.dtype,
    ).detach()
    cf_target_score = _field_token_score(
        cf_metrics,
        calibration[changed_field],
        target_token_ids,
        dtype=cf_pred.dtype,
    )
    target_effect = base_target_score - cf_target_score
    target_loss = _weighted_mean(F.softplus(float(target_margin) - target_effect))
    if float(success_weight) > 0.0:
        success_residual = float(success_margin) - target_effect
        success_loss_values = F.softplus(success_residual)
        if float(success_focal_gamma) > 0.0:
            success_focus = torch.sigmoid(success_residual.detach()).pow(float(success_focal_gamma))
            success_loss_values = success_loss_values * success_focus
        success_loss = _weighted_mean(success_loss_values)
    else:
        success_loss = cf_pred.new_tensor(0.0)
    if float(target_effect_cap_weight) > 0.0 and float(target_effect_cap) > 0.0:
        cap_loss = _weighted_mean(F.relu(target_effect - float(target_effect_cap)).pow(2))
    else:
        cap_loss = cf_pred.new_tensor(0.0)
    if float(token_margin_weight) > 0.0:
        cf_source_score = _field_token_score(
            cf_metrics,
            calibration[changed_field],
            source_token_ids,
            dtype=cf_pred.dtype,
        )
        token_margin_effect = cf_source_score - cf_target_score
        token_margin_loss = _weighted_mean(F.softplus(float(token_margin) - token_margin_effect))
    else:
        token_margin_effect = cf_pred.new_zeros(cf_pred.shape[0])
        token_margin_loss = cf_pred.new_tensor(0.0)
    if float(token_boundary_weight) > 0.0:
        cf_all_scores = _field_all_token_scores(
            cf_metrics,
            calibration[changed_field],
            dtype=cf_pred.dtype,
        )
        target_mask = torch.zeros_like(cf_all_scores, dtype=torch.bool)
        target_mask.scatter_(1, target_token_ids.long().view(-1, 1), True)
        competitor_scores = cf_all_scores.masked_fill(target_mask, float("inf"))
        best_competitor_score = competitor_scores.amin(dim=1)
        token_boundary_effect = best_competitor_score - cf_target_score
        token_boundary_loss = _weighted_mean(F.softplus(float(token_boundary_margin) - token_boundary_effect))
    else:
        token_boundary_effect = cf_pred.new_zeros(cf_pred.shape[0])
        token_boundary_loss = cf_pred.new_tensor(0.0)
    if float(signed_metric_weight) > 0.0:
        signed_target_metric_delta, signed_metric_delta, signed_metric_valid = _field_transition_metric_delta(
            base_metrics,
            cf_metrics,
            calibration[changed_field],
            source_token_ids,
            target_token_ids,
            dtype=cf_pred.dtype,
        )
        if bool(signed_metric_valid.any()):
            signed_metric_active = signed_metric_delta[signed_metric_valid]
            signed_target_metric_active = signed_target_metric_delta[signed_metric_valid]
            active_weights = sample_weights[signed_metric_valid] if sample_weights is not None else None
            if active_weights is None:
                signed_metric_loss = F.softplus(float(signed_metric_margin) - signed_metric_active).mean()
                signed_metric_success = (signed_metric_active.detach() > float(signed_metric_margin)).float().mean()
                signed_metric_mean = signed_metric_active.detach().mean()
                signed_target_metric_mean = signed_target_metric_active.detach().mean()
            else:
                active_weights = active_weights.to(device=signed_metric_active.device, dtype=signed_metric_active.dtype)
                active_denom = active_weights.clamp_min(0.0).sum().clamp_min(1e-6)
                signed_metric_loss = (
                    F.softplus(float(signed_metric_margin) - signed_metric_active) * active_weights
                ).sum() / active_denom
                signed_metric_success = (
                    (signed_metric_active.detach() > float(signed_metric_margin)).float() * active_weights
                ).sum() / active_denom
                signed_metric_mean = (signed_metric_active.detach() * active_weights).sum() / active_denom
                signed_target_metric_mean = (
                    signed_target_metric_active.detach() * active_weights
                ).sum() / active_denom
        else:
            signed_metric_loss = cf_pred.new_tensor(0.0)
            signed_metric_success = cf_pred.new_tensor(0.0)
            signed_metric_mean = cf_pred.new_tensor(0.0)
            signed_target_metric_mean = cf_pred.new_tensor(0.0)
    else:
        signed_metric_valid = torch.zeros_like(target_effect, dtype=torch.bool)
        signed_metric_loss = cf_pred.new_tensor(0.0)
        signed_metric_success = cf_pred.new_tensor(0.0)
        signed_metric_mean = cf_pred.new_tensor(0.0)
        signed_target_metric_mean = cf_pred.new_tensor(0.0)

    collateral_terms: List[torch.Tensor] = []
    for field in independent_collateral_fields(changed_field):
        if not _calibration_metrics_available(base_metrics, calibration[field]):
            continue
        token_ids = base_tokens[:, field_index[field]]
        if str(collateral_mode) == "metric_delta":
            drift = _field_collateral_metric_delta(
                base_metrics,
                cf_metrics,
                calibration[field],
                token_ids,
                dtype=cf_pred.dtype,
            )
            collateral_terms.append(_weighted_mean(F.relu(drift - float(collateral_margin))))
        else:
            base_score = _field_token_score(
                base_metrics,
                calibration[field],
                token_ids,
                dtype=cf_pred.dtype,
            ).detach()
            cf_score = _field_token_score(
                cf_metrics,
                calibration[field],
                token_ids,
                dtype=cf_pred.dtype,
            )
            collateral_terms.append(_weighted_mean(F.relu((cf_score - base_score).abs() - float(collateral_margin))))
    collateral = torch.stack(collateral_terms).mean() if collateral_terms else cf_pred.new_tensor(0.0)
    total = (
        float(target_weight) * target_loss
        + float(token_margin_weight) * token_margin_loss
        + float(token_boundary_weight) * token_boundary_loss
        + float(signed_metric_weight) * signed_metric_loss
        + float(success_weight) * success_loss
        + float(target_effect_cap_weight) * cap_loss
        + float(collateral_weight) * collateral
    )
    if not bool(return_stats):
        return total, {}
    stats = {
        "cf_target_effect": float(_weighted_mean(target_effect.detach()).item()),
        "cf_target_loss": float(target_loss.detach().item()),
        "cf_token_margin_effect": float(_weighted_mean(token_margin_effect.detach()).item()),
        "cf_token_margin_loss": float(token_margin_loss.detach().item()),
        "cf_token_margin_success": float(
            _weighted_mean((token_margin_effect.detach() > float(token_margin)).float()).item()
        ),
        "cf_token_boundary_effect": float(_weighted_mean(token_boundary_effect.detach()).item()),
        "cf_token_boundary_loss": float(token_boundary_loss.detach().item()),
        "cf_token_boundary_success": float(
            _weighted_mean((token_boundary_effect.detach() > float(token_boundary_margin)).float()).item()
        ),
        "cf_field": changed_field,
        "cf_signed_metric_delta": float(signed_metric_mean.item()),
        "cf_signed_target_metric_delta": float(signed_target_metric_mean.item()),
        "cf_signed_metric_loss": float(signed_metric_loss.detach().item()),
        "cf_signed_metric_success": float(signed_metric_success.item()),
        "cf_signed_metric_valid_rate": float(_weighted_mean(signed_metric_valid.detach().float()).item()),
        "cf_success_loss": float(success_loss.detach().item()),
        "cf_target_effect_cap_loss": float(cap_loss.detach().item()),
        "cf_collateral": float(collateral.detach().item()),
        "cf_success": float(_weighted_mean((target_effect.detach() > 0.0).float()).item()),
        "cf_sample_weight": float(
            sample_weights.detach().float().mean().item() if sample_weights is not None else 1.0
        ),
    }
    return total, stats


def curvature_counterfactual_control_loss(
    base_pred: torch.Tensor,
    cf_pred: torch.Tensor,
    reference: torch.Tensor,
    base_tokens: torch.Tensor,
    cf_tokens: torch.Tensor,
    calibration: dict | None,
    target_margin: float = 0.10,
    target_weight: float = 1.0,
    collateral_margin: float = 0.08,
    collateral_weight: float = 1.0,
    collateral_mode: str = "independent",
    collateral_field_weights: Dict[str, float] | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Curvature-specific paired loss with explicit local-shape collateral.

    Curvature counterfactuals are easy to satisfy by making the whole route
    longer, slower, or directionally different. This objective keeps the target
    term on the calibrated turn-density metric while measuring non-target drift
    with the same z-normalized metric deltas used by Table 4.
    """
    if calibration is None:
        zero = cf_pred.new_tensor(0.0)
        return zero, {
            "curvature_cf": 0.0,
            "curvature_cf_target": 0.0,
            "curvature_cf_collateral": 0.0,
            "curvature_cf_collateral_drift": 0.0,
            "curvature_cf_success": 0.0,
        }
    field = "curvature"
    field_idx = DEFAULT_MOBILITY_VOCAB.fields.index(field)
    source_ids = base_tokens[:, field_idx]
    target_ids = cf_tokens[:, field_idx]
    changed = source_ids.long() != target_ids.long()
    if not bool(changed.any()):
        zero = cf_pred.new_tensor(0.0)
        return zero, {
            "curvature_cf": 0.0,
            "curvature_cf_target": 0.0,
            "curvature_cf_collateral": 0.0,
            "curvature_cf_collateral_drift": 0.0,
            "curvature_cf_success": 0.0,
        }

    base_metrics = _calibrated_geometry_metrics(base_pred.float(), reference.float())
    cf_metrics = _calibrated_geometry_metrics(cf_pred.float(), reference.float())
    _, signed_delta, valid_metric = _field_transition_metric_delta(
        base_metrics,
        cf_metrics,
        calibration[field],
        source_ids,
        target_ids,
        dtype=cf_pred.dtype,
    )
    base_target_score = _field_token_score(
        base_metrics,
        calibration[field],
        target_ids,
        dtype=cf_pred.dtype,
    )
    cf_target_score = _field_token_score(
        cf_metrics,
        calibration[field],
        target_ids,
        dtype=cf_pred.dtype,
    )
    active = changed & valid_metric
    if not bool(active.any()):
        zero = cf_pred.new_tensor(0.0)
        return zero, {
            "curvature_cf": 0.0,
            "curvature_cf_target": 0.0,
            "curvature_cf_collateral": 0.0,
            "curvature_cf_collateral_drift": 0.0,
            "curvature_cf_success": 0.0,
        }

    target_delta = signed_delta[active]
    target_effect = (base_target_score - cf_target_score)[active]
    movement_loss = F.softplus(float(target_margin) - target_delta).mean()
    center_loss = F.softplus(float(target_margin) - target_effect).mean()
    target_loss = 0.5 * movement_loss + 0.5 * center_loss
    collateral_terms: List[torch.Tensor] = []
    collateral_values: List[torch.Tensor] = []
    collateral_weights: List[float] = []
    collateral_field_weights = collateral_field_weights or {}
    field_index = {name: idx for idx, name in enumerate(DEFAULT_MOBILITY_VOCAB.fields)}
    for collateral_field in collateral_fields_for_mode("curvature", collateral_mode):
        token_ids = base_tokens[:, field_index[collateral_field]]
        drift = _field_collateral_metric_delta(
            base_metrics,
            cf_metrics,
            calibration[collateral_field],
            token_ids,
            dtype=cf_pred.dtype,
        )[active]
        field_weight = max(float(collateral_field_weights.get(collateral_field, 1.0)), 0.0)
        if field_weight <= 0.0:
            continue
        field_collateral = F.relu(drift - float(collateral_margin)).mean()
        collateral_terms.append(field_collateral * field_weight)
        collateral_values.append(drift.detach().mean())
        collateral_weights.append(field_weight)
    weight_denom = float(sum(collateral_weights))
    collateral = (
        torch.stack(collateral_terms).sum() / max(weight_denom, 1e-6)
        if collateral_terms
        else cf_pred.new_tensor(0.0)
    )
    collateral_drift = (
        torch.stack(collateral_values).mean()
        if collateral_values
        else cf_pred.new_tensor(0.0)
    )
    total = float(target_weight) * target_loss + float(collateral_weight) * collateral
    stats = {
        "curvature_cf": float(total.detach().item()),
        "curvature_cf_target": float(target_delta.detach().mean().item()),
        "curvature_cf_collateral": float(collateral.detach().item()),
        "curvature_cf_collateral_drift": float(collateral_drift.detach().item()),
        "curvature_cf_success": float((target_delta.detach() > float(target_margin)).float().mean().item()),
    }
    return total, stats


def reference_adherence_counterfactual_control_loss(
    base_pred: torch.Tensor,
    cf_pred: torch.Tensor,
    reference: torch.Tensor,
    base_tokens: torch.Tensor,
    cf_tokens: torch.Tensor,
    calibration: dict | None,
    target_margin: float = 0.05,
    target_weight: float = 1.0,
    collateral_margin: float = 0.06,
    collateral_weight: float = 0.5,
    base_metrics: Dict[str, torch.Tensor] | None = None,
    cf_metrics: Dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Reference-adherence loss aligned with Table-4 signed movement.

    OD-chord conditioning makes this field especially easy to under-train: the
    generator often already stays near the chord, so a generic token-distance
    loss has a weak gradient. This objective directly rewards movement of
    mean-reference-distance toward the changed token while guarding the
    independent non-target fields used by Table 4.
    """
    if calibration is None:
        zero = cf_pred.new_tensor(0.0)
        return zero, {
            "reference_cf": 0.0,
            "reference_cf_target": 0.0,
            "reference_cf_center_effect": 0.0,
            "reference_cf_collateral": 0.0,
            "reference_cf_success": 0.0,
        }
    field = "reference_adherence"
    field_idx = DEFAULT_MOBILITY_VOCAB.fields.index(field)
    source_ids = base_tokens[:, field_idx]
    target_ids = cf_tokens[:, field_idx]
    changed = source_ids.long() != target_ids.long()
    if not bool(changed.any()):
        zero = cf_pred.new_tensor(0.0)
        return zero, {
            "reference_cf": 0.0,
            "reference_cf_target": 0.0,
            "reference_cf_center_effect": 0.0,
            "reference_cf_collateral": 0.0,
            "reference_cf_success": 0.0,
        }

    base_metrics = (
        base_metrics
        if isinstance(base_metrics, dict)
        else _calibrated_geometry_metrics(base_pred.float(), reference.float())
    )
    cf_metrics = (
        cf_metrics
        if isinstance(cf_metrics, dict)
        else _calibrated_geometry_metrics(cf_pred.float(), reference.float())
    )
    _, signed_delta, valid_metric = _field_transition_metric_delta(
        base_metrics,
        cf_metrics,
        calibration[field],
        source_ids,
        target_ids,
        dtype=cf_pred.dtype,
    )
    base_target_score = _field_token_score(
        base_metrics,
        calibration[field],
        target_ids,
        dtype=cf_pred.dtype,
    )
    cf_target_score = _field_token_score(
        cf_metrics,
        calibration[field],
        target_ids,
        dtype=cf_pred.dtype,
    )
    active = changed & valid_metric
    if not bool(active.any()):
        zero = cf_pred.new_tensor(0.0)
        return zero, {
            "reference_cf": 0.0,
            "reference_cf_target": 0.0,
            "reference_cf_center_effect": 0.0,
            "reference_cf_collateral": 0.0,
            "reference_cf_success": 0.0,
        }

    target_delta = signed_delta[active]
    target_effect = (base_target_score - cf_target_score)[active]
    movement_loss = F.softplus(float(target_margin) - target_delta).mean()
    center_loss = F.softplus(float(target_margin) - target_effect).mean()
    target_loss = 0.5 * movement_loss + 0.5 * center_loss
    collateral_terms: List[torch.Tensor] = []
    field_index = {name: idx for idx, name in enumerate(DEFAULT_MOBILITY_VOCAB.fields)}
    for collateral_field in independent_collateral_fields(field):
        token_ids = base_tokens[:, field_index[collateral_field]]
        drift = _field_collateral_metric_delta(
            base_metrics,
            cf_metrics,
            calibration[collateral_field],
            token_ids,
            dtype=cf_pred.dtype,
        )[active]
        collateral_terms.append(F.relu(drift - float(collateral_margin)).mean())
    collateral = torch.stack(collateral_terms).mean() if collateral_terms else cf_pred.new_tensor(0.0)
    total = float(target_weight) * target_loss + float(collateral_weight) * collateral
    stats = {
        "reference_cf": float(total.detach().item()),
        "reference_cf_target": float(target_delta.detach().mean().item()),
        "reference_cf_center_effect": float(target_effect.detach().mean().item()),
        "reference_cf_collateral": float(collateral.detach().item()),
        "reference_cf_success": float((target_effect.detach() > float(target_margin)).float().mean().item()),
    }
    return total, stats


def _counterfactual_preference_score(
    base_metrics: Dict[str, torch.Tensor],
    cf_metrics: Dict[str, torch.Tensor],
    base_tokens: torch.Tensor,
    cf_tokens: torch.Tensor,
    changed_field: str,
    calibration: dict,
    dtype: torch.dtype,
    collateral_weight: float = 1.0,
    signed_metric_weight: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    field_index = {field: idx for idx, field in enumerate(DEFAULT_MOBILITY_VOCAB.fields)}
    field_idx = field_index[changed_field]
    source_token_ids = base_tokens[:, field_idx]
    target_token_ids = cf_tokens[:, field_idx]
    base_target_score = _field_token_score(
        base_metrics,
        calibration[changed_field],
        target_token_ids,
        dtype=dtype,
    ).detach()
    cf_target_score = _field_token_score(
        cf_metrics,
        calibration[changed_field],
        target_token_ids,
        dtype=dtype,
    )
    target_effect = base_target_score - cf_target_score
    _, signed_metric_delta, signed_metric_valid = _field_transition_metric_delta(
        base_metrics,
        cf_metrics,
        calibration[changed_field],
        source_token_ids,
        target_token_ids,
        dtype=dtype,
    )
    signed_metric_term = torch.where(
        signed_metric_valid,
        signed_metric_delta,
        torch.zeros_like(signed_metric_delta),
    )

    collateral_terms: List[torch.Tensor] = []
    for field in independent_collateral_fields(changed_field):
        token_ids = base_tokens[:, field_index[field]]
        base_score = _field_token_score(
            base_metrics,
            calibration[field],
            token_ids,
            dtype=dtype,
        ).detach()
        cf_score = _field_token_score(
            cf_metrics,
            calibration[field],
            token_ids,
            dtype=dtype,
        )
        collateral_terms.append((cf_score - base_score).abs())
    collateral = torch.stack(collateral_terms, dim=0).mean(dim=0) if collateral_terms else torch.zeros_like(target_effect)
    score = target_effect + float(signed_metric_weight) * signed_metric_term - float(collateral_weight) * collateral
    return score, {
        "target_effect": target_effect,
        "collateral": collateral,
        "signed_metric_delta": signed_metric_delta,
        "signed_metric_valid": signed_metric_valid.to(dtype=dtype),
    }


def calibration_counterfactual_preference_loss(
    base_pred: torch.Tensor,
    chosen_pred: torch.Tensor,
    rejected_pred: torch.Tensor,
    reference: torch.Tensor,
    base_tokens: torch.Tensor,
    chosen_tokens: torch.Tensor,
    rejected_tokens: torch.Tensor,
    changed_field: str,
    calibration: dict | None,
    margin: float = 0.05,
    collateral_weight: float = 1.0,
    signed_metric_weight: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Preference-style counterfactual objective.

    A supported changed-token trajectory should have a better Table-4 utility
    than an unsupported or low-quality changed-token trajectory under the same
    reference and source instruction.  This turns transition diagnostics into
    chosen/rejected supervision instead of using them only for sampling.
    """
    if calibration is None:
        zero = chosen_pred.new_tensor(0.0)
        return zero, {
            "cf_preference": 0.0,
            "cf_preference_gap": 0.0,
            "cf_preference_acc": 0.0,
            "cf_preference_chosen_score": 0.0,
            "cf_preference_rejected_score": 0.0,
            "cf_preference_chosen_effect": 0.0,
            "cf_preference_rejected_effect": 0.0,
            "cf_preference_chosen_collateral": 0.0,
            "cf_preference_rejected_collateral": 0.0,
        }
    if changed_field not in DEFAULT_MOBILITY_VOCAB.fields:
        raise ValueError(f"Unknown counterfactual field: {changed_field}")

    base_metrics = _calibrated_geometry_metrics(base_pred.float(), reference.float())
    chosen_metrics = _calibrated_geometry_metrics(chosen_pred.float(), reference.float())
    rejected_metrics = _calibrated_geometry_metrics(rejected_pred.float(), reference.float())
    chosen_score, chosen_stats = _counterfactual_preference_score(
        base_metrics,
        chosen_metrics,
        base_tokens,
        chosen_tokens,
        changed_field,
        calibration,
        dtype=chosen_pred.dtype,
        collateral_weight=collateral_weight,
        signed_metric_weight=signed_metric_weight,
    )
    rejected_score, rejected_stats = _counterfactual_preference_score(
        base_metrics,
        rejected_metrics,
        base_tokens,
        rejected_tokens,
        changed_field,
        calibration,
        dtype=chosen_pred.dtype,
        collateral_weight=collateral_weight,
        signed_metric_weight=signed_metric_weight,
    )
    gap = chosen_score - rejected_score
    loss = F.softplus(float(margin) - gap).mean()
    stats = {
        "cf_preference": float(loss.detach().item()),
        "cf_preference_gap": float(gap.detach().mean().item()),
        "cf_preference_acc": float((gap.detach() > 0.0).float().mean().item()),
        "cf_preference_chosen_score": float(chosen_score.detach().mean().item()),
        "cf_preference_rejected_score": float(rejected_score.detach().mean().item()),
        "cf_preference_chosen_effect": float(chosen_stats["target_effect"].detach().mean().item()),
        "cf_preference_rejected_effect": float(rejected_stats["target_effect"].detach().mean().item()),
        "cf_preference_chosen_collateral": float(chosen_stats["collateral"].detach().mean().item()),
        "cf_preference_rejected_collateral": float(rejected_stats["collateral"].detach().mean().item()),
        "cf_preference_chosen_signed_delta": float(chosen_stats["signed_metric_delta"].detach().mean().item()),
        "cf_preference_rejected_signed_delta": float(rejected_stats["signed_metric_delta"].detach().mean().item()),
    }
    return loss, stats


def masked_counterfactual_local_proposal_loss(
    base_pred: torch.Tensor,
    cf_pred: torch.Tensor,
    reference: torch.Tensor,
    base_tokens: torch.Tensor,
    changed_tokens: torch.Tensor,
    changed_field: str,
    calibration: dict | None = None,
    edit_mask: torch.Tensor | None = None,
    preserve_mask: torch.Tensor | None = None,
    target_margin: float = 0.03,
    collateral_margin: float = 0.02,
    target_weight: float = 1.0,
    collateral_weight: float = 1.0,
    anchor_weight: float = 0.5,
    direction_collateral_weight: float = 0.0,
    direction_collateral_margin: float = 0.08,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Edited-span target gain plus preserved-span collateral suppression."""
    if changed_field not in DEFAULT_MOBILITY_VOCAB.fields:
        raise ValueError(f"Unknown counterfactual field: {changed_field}")
    if edit_mask is None or preserve_mask is None:
        edit_mask, preserve_mask, _, _, _ = build_counterfactual_edit_masks(
            base_tokens,
            changed_tokens,
            changed_field,
            int(cf_pred.shape[1]),
        )
    edit_mask = edit_mask.to(device=cf_pred.device, dtype=cf_pred.dtype)
    preserve_mask = preserve_mask.to(device=cf_pred.device, dtype=cf_pred.dtype)
    edit_denom = edit_mask.sum(dim=1).clamp_min(1e-6)
    preserve_denom = preserve_mask.sum(dim=1).clamp_min(1e-6)

    base_metrics = _calibrated_geometry_metrics(base_pred.float(), reference.float())
    cf_metrics = _calibrated_geometry_metrics(cf_pred.float(), reference.float())

    base_step = torch.linalg.norm(base_pred[:, 1:] - base_pred[:, :-1], dim=-1)
    cf_step = torch.linalg.norm(cf_pred[:, 1:] - cf_pred[:, :-1], dim=-1)
    step_edit_mask = 0.5 * (edit_mask[:, 1:] + edit_mask[:, :-1])
    step_preserve_mask = 0.5 * (preserve_mask[:, 1:] + preserve_mask[:, :-1])
    base_heading = heading_change(base_pred.float())
    cf_heading = heading_change(cf_pred.float())
    base_curv = curvature_energy(base_pred.float())
    cf_curv = curvature_energy(cf_pred.float())
    local_step_gain = ((cf_step - base_step).abs() * step_edit_mask).sum(dim=1) / step_edit_mask.sum(dim=1).clamp_min(1e-6)

    if changed_field == "direction":
        target_token = changed_tokens[:, DEFAULT_MOBILITY_VOCAB.fields.index("direction")].to(device=cf_pred.device)
        multi_turn_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["multi_turn"])
        if bool((target_token == multi_turn_id).any()):
            direction_gain = (cf_curv - base_curv) + 0.35 * local_step_gain
        else:
            direction_gain = (cf_heading - base_heading).abs() + 0.15 * local_step_gain
        target_signal = direction_gain
    elif changed_field == "curvature":
        target_token = changed_tokens[:, DEFAULT_MOBILITY_VOCAB.fields.index("curvature")].to(device=cf_pred.device)
        winding_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["curvature"]["winding"])
        smooth_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["curvature"]["smooth"])
        target_signal = cf_curv - base_curv
        target_signal = torch.where(target_token == smooth_id, -target_signal, target_signal)
        target_signal = torch.where(target_token == winding_id, target_signal + 0.25 * local_step_gain, target_signal)
    else:
        base_ref = base_metrics["mean_reference_distance_norm"]
        cf_ref = cf_metrics["mean_reference_distance_norm"]
        target_signal = base_ref - cf_ref

    target_loss = F.softplus(float(target_margin) - target_signal).mean()

    preserve_drift = torch.linalg.norm(cf_pred - base_pred, dim=-1)
    preserve_drift = (preserve_drift * preserve_mask).sum(dim=1) / preserve_denom
    preserved_collateral = F.relu(preserve_drift - float(collateral_margin)).mean()
    direction_collateral = cf_pred.new_tensor(0.0)
    if (
        changed_field == "curvature"
        and calibration is not None
        and "direction" in calibration
        and float(direction_collateral_weight) > 0.0
    ):
        direction_field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("direction")
        direction_token_ids = base_tokens[:, direction_field_idx].to(device=cf_pred.device, dtype=torch.long)
        direction_delta = _field_collateral_metric_delta(
            base_metrics,
            cf_metrics,
            calibration["direction"],
            direction_token_ids,
            dtype=cf_pred.dtype,
        )
        direction_desc_base = _local_direction_descriptors(base_pred.float())
        direction_desc_cf = _local_direction_descriptors(cf_pred.float())
        signed_turn_drift = (
            direction_desc_cf["local_signed_total_turn_rad"] - direction_desc_base["local_signed_total_turn_rad"]
        ).abs()
        total_abs_turn_drift = (
            direction_desc_cf["local_total_abs_turn_rad"] - direction_desc_base["local_total_abs_turn_rad"]
        ).abs()
        direction_penalty = torch.stack(
            [
                F.relu(direction_delta - float(direction_collateral_margin)),
                0.35 * F.relu(signed_turn_drift - float(direction_collateral_margin)),
                0.20 * F.relu(total_abs_turn_drift - float(direction_collateral_margin) * 2.0),
            ],
            dim=1,
        ).mean(dim=1)
        direction_collateral = direction_penalty.mean()

    endpoint_drift = (
        torch.linalg.norm(cf_pred[:, 0] - base_pred[:, 0], dim=-1)
        + torch.linalg.norm(cf_pred[:, -1] - base_pred[:, -1], dim=-1)
    ) * 0.5
    length_drift = (
        path_length(cf_pred.float()) - path_length(base_pred.float())
    ).abs() / path_length(base_pred.float()).clamp_min(1e-6)
    anchor_loss = (endpoint_drift + length_drift.to(dtype=endpoint_drift.dtype)).mean()

    total = (
        float(target_weight) * target_loss
        + float(collateral_weight) * preserved_collateral
        + float(anchor_weight) * anchor_loss
        + float(direction_collateral_weight) * direction_collateral
    )
    stats = {
        "cf_local_masked": float(total.detach().item()),
        "cf_local_target_gain": float(target_signal.detach().mean().item()),
        "cf_local_target_loss": float(target_loss.detach().item()),
        "cf_local_preserved_collateral": float(preserved_collateral.detach().item()),
        "cf_local_anchor": float(anchor_loss.detach().item()),
        "cf_local_direction_collateral": float(direction_collateral.detach().item()),
        "cf_local_edit_fraction": float(edit_mask.detach().mean().item()),
    }
    return total, stats


def direction_counterfactual_control_loss(
    base_pred: torch.Tensor,
    cf_pred: torch.Tensor,
    reference: torch.Tensor,
    base_tokens: torch.Tensor,
    cf_tokens: torch.Tensor,
    target_margin: float = 0.15,
    collateral_margin: float = 0.08,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Direction-specific paired control loss with differentiable descriptors.

    This is an auxiliary Table-4 loss for the hardest field. It complements the
    calibration score loss by pushing the geometry feature that each direction
    token actually uses: left/right side dominance, U-turn peak heading, or
    multi-turn count. Unchanged non-direction fields are still guarded by the
    caller's generic collateral loss.
    """
    del base_tokens
    ids = DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]
    field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("direction")
    target = cf_tokens[:, field_idx].long()
    base_desc = _local_direction_descriptors(base_pred.float())
    cf_desc = _local_direction_descriptors(cf_pred.float())

    effects: List[torch.Tensor] = []
    losses: List[torch.Tensor] = []

    def add(mask: torch.Tensor, effect: torch.Tensor, margin: float) -> None:
        if bool(mask.any()):
            selected = effect[mask]
            effects.append(selected)
            losses.append(F.softplus(float(margin) - selected).mean())

    straight = target == int(ids["straight"])
    straight_effect = (
        base_desc["local_abs_heading_diff_rad"]
        - cf_desc["local_abs_heading_diff_rad"]
        + 0.25 * (base_desc["local_total_abs_turn_rad"] - cf_desc["local_total_abs_turn_rad"])
    )
    add(straight, straight_effect, target_margin)

    left = target == int(ids["left_turn"])
    left_effect = (
        cf_desc["local_signed_total_turn_rad"]
        - base_desc["local_signed_total_turn_rad"]
        + 0.25 * (cf_desc["local_dy"] - base_desc["local_dy"])
    )
    add(left, left_effect, target_margin)

    right = target == int(ids["right_turn"])
    right_effect = (
        base_desc["local_signed_total_turn_rad"]
        - cf_desc["local_signed_total_turn_rad"]
        + 0.25 * (base_desc["local_dy"] - cf_desc["local_dy"])
    )
    add(right, right_effect, target_margin)

    u_turn = target == int(ids["u_turn"])
    u_effect = (
        cf_desc["local_abs_heading_diff_rad"]
        - base_desc["local_abs_heading_diff_rad"]
        + 0.25 * (base_desc["local_dx"] - cf_desc["local_dx"])
    )
    add(u_turn, u_effect, target_margin)

    multi_turn = target == int(ids["multi_turn"])
    multi_effect = (
        cf_desc["local_turn_count"]
        - base_desc["local_turn_count"]
        + 0.25 * (cf_desc["local_total_abs_turn_rad"] - base_desc["local_total_abs_turn_rad"])
    )
    add(multi_turn, multi_effect, target_margin)

    if losses:
        target_loss = torch.stack(losses).mean().to(dtype=cf_pred.dtype)
        effect_tensor = torch.cat([item.reshape(-1) for item in effects]).to(dtype=cf_pred.dtype)
    else:
        target_loss = cf_pred.new_tensor(0.0)
        effect_tensor = cf_pred.new_zeros(1)

    # Keep a direction counterfactual from changing overall motion scale too
    # wildly unless the target token explicitly asks for it.
    collateral = F.relu(
        (cf_desc["local_total_abs_turn_rad"] - base_desc["local_total_abs_turn_rad"]).abs()
        - float(collateral_margin)
    ).mean()
    total = target_loss + 0.25 * collateral
    stats = {
        "direction_cf": float(total.detach().item()),
        "direction_cf_target_loss": float(target_loss.detach().item()),
        "direction_cf_effect": float(effect_tensor.detach().mean().item()),
        "direction_cf_success": float((effect_tensor.detach() > 0.0).float().mean().item()),
        "direction_cf_collateral": float(collateral.detach().item()),
    }
    return total, stats


def _eval_aligned_direction_logits(
    desc: Dict[str, torch.Tensor],
    straight_heading_rad: float,
    straight_lateral: float,
    turn_margin_rad: float,
    uturn_heading_rad: float,
    uturn_longitudinal: float,
    multiturn_total_abs_rad: float,
    multiturn_soft_count: float,
) -> torch.Tensor:
    """Soft logits for the Table-3 hybrid direction contract.

    The scores mirror the local-frame rule family used by the direction
    diagnostic: straight is low heading/lateral drift, left/right are signed
    turn plus lateral side, U-turn is large heading and backward progress, and
    multi-turn is high accumulated turning. They are not probabilities; they are
    differentiable ranking scores for counterfactual supervision.
    """
    eps = 1e-6
    abs_heading = desc["local_abs_heading_diff_rad"]
    local_dy = desc["local_dy"]
    local_dx = desc["local_dx"]
    signed_turn = desc["local_signed_total_turn_rad"]
    total_abs = desc["local_total_abs_turn_rad"]
    turn_count = desc["local_turn_count"]
    left_ratio = desc["local_left_turn_ratio"]
    right_ratio = desc["local_right_turn_ratio"]

    straight = -(
        abs_heading / max(float(straight_heading_rad), eps)
        + local_dy.abs() / max(float(straight_lateral), eps)
        + 0.25 * total_abs / max(float(multiturn_total_abs_rad), eps)
    )
    left = (
        signed_turn / max(float(turn_margin_rad), eps)
        + local_dy / max(float(straight_lateral), eps)
        + 0.50 * (left_ratio - right_ratio)
    )
    right = (
        -signed_turn / max(float(turn_margin_rad), eps)
        - local_dy / max(float(straight_lateral), eps)
        + 0.50 * (right_ratio - left_ratio)
    )
    u_turn = (
        abs_heading / max(float(uturn_heading_rad), eps)
        + (-local_dx) / max(abs(float(uturn_longitudinal)), 0.05)
        + 0.25 * total_abs / max(float(multiturn_total_abs_rad), eps)
    )
    multi_turn = (
        total_abs / max(float(multiturn_total_abs_rad), eps)
        + 0.35 * turn_count / max(float(multiturn_soft_count), eps)
    )
    ids = DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]
    logits = torch.stack([straight, left, right, u_turn, multi_turn], dim=1)
    ordered = torch.empty_like(logits)
    ordered[:, int(ids["straight"])] = straight
    ordered[:, int(ids["left_turn"])] = left
    ordered[:, int(ids["right_turn"])] = right
    ordered[:, int(ids["u_turn"])] = u_turn
    ordered[:, int(ids["multi_turn"])] = multi_turn
    return ordered


def _target_aware_direction_collateral(
    base_desc: Dict[str, torch.Tensor],
    cf_desc: Dict[str, torch.Tensor],
    target: torch.Tensor,
    ids: Dict[str, int],
    margin: float,
) -> torch.Tensor:
    """Collateral for direction edits that does not penalize target evidence.

    A uniform collateral on total turn and turn count conflicts with the
    objective for `multi_turn` and `u_turn` targets: the model is asked to
    increase turning evidence while being penalized for increasing the same
    descriptors.  This helper keeps collateral on non-target descriptors only.
    """
    if target.numel() == 0:
        return target.new_tensor(0.0, dtype=cf_desc["local_dx"].dtype)

    margin = float(margin)
    dx_drift = (cf_desc["local_dx"] - base_desc["local_dx"]).abs()
    dy_drift = (cf_desc["local_dy"] - base_desc["local_dy"]).abs()
    total_abs_drift = (
        cf_desc["local_total_abs_turn_rad"] - base_desc["local_total_abs_turn_rad"]
    ).abs()
    turn_count_drift = (cf_desc["local_turn_count"] - base_desc["local_turn_count"]).abs()
    signed_turn_drift = (
        cf_desc["local_signed_total_turn_rad"] - base_desc["local_signed_total_turn_rad"]
    ).abs()

    losses: List[torch.Tensor] = []

    def add(mask: torch.Tensor, parts: List[torch.Tensor]) -> None:
        if bool(mask.any()):
            losses.append(
                torch.stack([F.relu(part[mask] - margin).mean() for part in parts]).mean()
            )

    straight = target == int(ids["straight"])
    add(
        straight,
        [
            dx_drift,
            dy_drift,
            F.relu(cf_desc["local_total_abs_turn_rad"] - base_desc["local_total_abs_turn_rad"]),
            F.relu(cf_desc["local_turn_count"] - base_desc["local_turn_count"]),
        ],
    )

    left = target == int(ids["left_turn"])
    right = target == int(ids["right_turn"])
    add(left | right, [dx_drift, total_abs_drift, turn_count_drift])

    u_turn = target == int(ids["u_turn"])
    add(u_turn, [dy_drift, turn_count_drift])

    multi_turn = target == int(ids["multi_turn"])
    add(multi_turn, [dx_drift, dy_drift, signed_turn_drift])

    if not losses:
        return cf_desc["local_dx"].new_tensor(0.0)
    return torch.stack(losses).mean()


def _direction_metric_scores(
    metrics: Dict[str, torch.Tensor],
    field_calib: dict,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return calibration distance scores for every direction token."""
    return _field_all_token_scores(metrics, field_calib, dtype=dtype)


def hybrid_direction_counterfactual_loss(
    base_pred: torch.Tensor,
    cf_pred: torch.Tensor,
    reference: torch.Tensor,
    base_tokens: torch.Tensor,
    cf_tokens: torch.Tensor,
    calibration: dict | None,
    auto_margin: float = 0.15,
    rule_margin: float = 0.10,
    effect_margin: float = 0.10,
    collateral_margin: float = 0.20,
    auto_weight: float = 1.0,
    rule_weight: float = 1.0,
    effect_weight: float = 0.5,
    collateral_weight: float = 0.25,
    local_collateral_weight: float = 0.0,
    local_collateral_margin: float = 0.05,
    local_endpoint_margin: float = 0.01,
    straight_heading_rad: float = math.pi / 6.0,
    straight_lateral: float = 0.08,
    uturn_heading_rad: float = 2.0,
    uturn_longitudinal: float = -0.05,
    multiturn_total_abs_rad: float = 4.5,
    multiturn_soft_count: float = 3.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Transition-aware direction loss matching the Table-3 hybrid contract.

    The default direction diagnostic uses a hybrid policy: left/right may be
    recovered by calibration-style auto prediction when iMotion predicts
    multi-turn, while straight/u-turn/multi-turn must satisfy iMotion-style
    local-frame rules. This loss mirrors that contract instead of treating all
    direction tokens as one generic scalar movement.
    """
    field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("direction")
    ids = DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]
    target = cf_tokens[:, field_idx].long()
    source = base_tokens[:, field_idx].long()
    valid = target != source
    if not bool(valid.any()):
        zero = cf_pred.new_tensor(0.0)
        return zero, {
            "hybrid_direction_cf": 0.0,
            "hybrid_direction_cf_auto": 0.0,
            "hybrid_direction_cf_rule": 0.0,
            "hybrid_direction_cf_effect": 0.0,
            "hybrid_direction_cf_success_proxy": 0.0,
            "hybrid_direction_cf_auto_success": 0.0,
            "hybrid_direction_cf_rule_success": 0.0,
            "hybrid_direction_cf_collateral": 0.0,
        }

    base_desc = _local_direction_descriptors(base_pred.float())
    cf_desc = _local_direction_descriptors(cf_pred.float())
    base_metrics = _calibrated_geometry_metrics(base_pred.float(), reference.float())
    cf_metrics = _calibrated_geometry_metrics(cf_pred.float(), reference.float())
    direction_calib = None if calibration is None else calibration.get("direction")
    target = target[valid]
    source = source[valid]

    left_right_mask = (target == int(ids["left_turn"])) | (target == int(ids["right_turn"]))
    rule_mask = ~left_right_mask
    losses: List[torch.Tensor] = []
    proxy_hits: List[torch.Tensor] = []
    auto_loss = cf_pred.new_tensor(0.0)
    auto_success = cf_pred.new_tensor(0.0)
    effect_loss = cf_pred.new_tensor(0.0)
    local_collateral = cf_pred.new_tensor(0.0)

    if bool(left_right_mask.any()) and direction_calib is not None:
        base_scores = _direction_metric_scores(base_metrics, direction_calib, dtype=cf_pred.dtype)[valid]
        cf_scores = _direction_metric_scores(cf_metrics, direction_calib, dtype=cf_pred.dtype)[valid]
        lr_cf_scores = cf_scores[left_right_mask]
        lr_base_scores = base_scores[left_right_mask]
        lr_target = target[left_right_mask]
        lr_source = source[left_right_mask]
        target_score = lr_cf_scores.gather(1, lr_target[:, None]).squeeze(1)
        source_score = lr_cf_scores.gather(1, lr_source[:, None]).squeeze(1)
        competitor_scores = lr_cf_scores.masked_fill(
            F.one_hot(lr_target, num_classes=lr_cf_scores.shape[1]).bool(),
            torch.finfo(lr_cf_scores.dtype).max,
        )
        best_other = competitor_scores.min(dim=1).values
        target_base_score = lr_base_scores.gather(1, lr_target[:, None]).squeeze(1)
        auto_margin_loss = F.softplus(float(auto_margin) + target_score - best_other).mean()
        source_drop_loss = F.softplus(float(auto_margin) + target_score - source_score).mean()
        auto_loss = auto_margin_loss + 0.5 * source_drop_loss
        effect = target_base_score - target_score
        effect_loss = F.softplus(float(effect_margin) - effect).mean()
        auto_pred = lr_cf_scores.argmin(dim=1)
        auto_success = (auto_pred == lr_target).to(dtype=cf_pred.dtype).mean()
        proxy_hits.append(auto_pred == lr_target)
        losses.append(float(auto_weight) * auto_loss)
        losses.append(float(effect_weight) * effect_loss)
        if float(local_collateral_weight) > 0.0:
            lr_valid = valid.nonzero(as_tuple=False).view(-1)[left_right_mask]
            metric_drifts = [
                (cf_metrics["length_ratio"][lr_valid] - base_metrics["length_ratio"][lr_valid]).abs(),
                (cf_metrics["displacement_ratio"][lr_valid] - base_metrics["displacement_ratio"][lr_valid]).abs(),
                (cf_metrics["curvature_energy"][lr_valid] - base_metrics["curvature_energy"][lr_valid]).abs(),
                (
                    cf_metrics["mean_reference_distance_norm"][lr_valid]
                    - base_metrics["mean_reference_distance_norm"][lr_valid]
                ).abs(),
            ]
            start_drift = torch.linalg.norm(
                cf_pred.float()[lr_valid, 0] - base_pred.float()[lr_valid, 0],
                dim=-1,
            )
            end_drift = torch.linalg.norm(
                cf_pred.float()[lr_valid, -1] - base_pred.float()[lr_valid, -1],
                dim=-1,
            )
            endpoint_drift = torch.maximum(start_drift, end_drift)
            metric_loss = torch.stack(
                [
                    F.relu(item - float(local_collateral_margin))
                    for item in metric_drifts
                ],
                dim=0,
            ).mean()
            endpoint_loss = F.relu(endpoint_drift - float(local_endpoint_margin)).mean()
            local_collateral = metric_loss + endpoint_loss
            losses.append(float(local_collateral_weight) * local_collateral)

    valid_abs_heading = cf_desc["local_abs_heading_diff_rad"][valid]
    valid_local_dy = cf_desc["local_dy"][valid]
    valid_local_dx = cf_desc["local_dx"][valid]
    valid_total_abs = cf_desc["local_total_abs_turn_rad"][valid]
    valid_turn_count = cf_desc["local_turn_count"][valid]
    valid_target = target
    valid_source = source

    rule_parts: List[torch.Tensor] = []
    rule_hits: List[torch.Tensor] = []

    def add_rule(mask: torch.Tensor, parts: List[torch.Tensor], hit: torch.Tensor) -> None:
        if bool(mask.any()):
            rule_parts.append(torch.stack([part[mask].mean() for part in parts]).mean())
            rule_hits.append(hit[mask])

    straight_mask = valid_target == int(ids["straight"])
    straight_hit = (
        (valid_abs_heading < float(straight_heading_rad))
        & (valid_local_dy.abs() < float(straight_lateral))
        & (valid_turn_count < float(multiturn_soft_count))
        & (valid_total_abs < float(multiturn_total_abs_rad))
    )
    add_rule(
        straight_mask,
        [
            F.relu(valid_abs_heading - float(straight_heading_rad)) / max(float(straight_heading_rad), 1e-6),
            F.relu(valid_local_dy.abs() - float(straight_lateral)) / max(float(straight_lateral), 1e-6),
            F.relu(valid_turn_count - float(multiturn_soft_count)) / max(float(multiturn_soft_count), 1e-6),
            F.relu(valid_total_abs - float(multiturn_total_abs_rad)) / max(float(multiturn_total_abs_rad), 1e-6),
        ],
        straight_hit,
    )

    u_turn_mask = valid_target == int(ids["u_turn"])
    u_turn_hit = (
        (valid_local_dx < float(uturn_longitudinal))
        & (valid_abs_heading >= float(uturn_heading_rad))
    )
    add_rule(
        u_turn_mask,
        [
            F.relu(float(uturn_heading_rad) - valid_abs_heading) / max(float(uturn_heading_rad), 1e-6),
            F.relu(valid_local_dx - float(uturn_longitudinal)) / max(abs(float(uturn_longitudinal)), 0.05),
        ],
        u_turn_hit,
    )

    multi_mask = valid_target == int(ids["multi_turn"])
    multi_score = torch.maximum(
        valid_turn_count / max(float(multiturn_soft_count), 1e-6),
        valid_total_abs / max(float(multiturn_total_abs_rad), 1e-6),
    )
    multi_hit = multi_score >= 1.0
    add_rule(
        multi_mask,
        [F.relu(1.0 + float(rule_margin) - multi_score)],
        multi_hit,
    )

    if bool(rule_mask.any()) and rule_parts:
        rule_loss = torch.stack(rule_parts).mean().to(dtype=cf_pred.dtype)
        losses.append(float(rule_weight) * rule_loss)
        rule_success = torch.cat([item.to(dtype=cf_pred.dtype) for item in rule_hits], dim=0).mean()
        proxy_hits.extend(rule_hits)
    else:
        rule_loss = cf_pred.new_tensor(0.0)
        rule_success = cf_pred.new_tensor(0.0)

    # Penalize unchanged source-rule still dominating for rule tokens. This is
    # especially useful for multi_turn -> straight/u_turn transitions.
    source_rule_parts: List[torch.Tensor] = []
    if bool(rule_mask.any()):
        source_rule = valid_source[rule_mask]
        rule_abs_heading = valid_abs_heading[rule_mask]
        rule_local_dx = valid_local_dx[rule_mask]
        rule_local_dy = valid_local_dy[rule_mask]
        rule_total_abs = valid_total_abs[rule_mask]
        rule_turn_count = valid_turn_count[rule_mask]
        source_multi = source_rule == int(ids["multi_turn"])
        if bool(source_multi.any()):
            source_multi_score = torch.maximum(
                rule_turn_count[source_multi] / max(float(multiturn_soft_count), 1e-6),
                rule_total_abs[source_multi] / max(float(multiturn_total_abs_rad), 1e-6),
            )
            source_rule_parts.append(F.relu(source_multi_score - (1.0 - float(rule_margin))).mean())
        source_straight = source_rule == int(ids["straight"])
        if bool(source_straight.any()):
            source_straight_score = (
                rule_abs_heading[source_straight] / max(float(straight_heading_rad), 1e-6)
                + rule_local_dy[source_straight].abs() / max(float(straight_lateral), 1e-6)
            )
            source_rule_parts.append(F.relu(1.0 - source_straight_score).mean())
        source_uturn = source_rule == int(ids["u_turn"])
        if bool(source_uturn.any()):
            source_uturn_score = (
                rule_abs_heading[source_uturn] / max(float(uturn_heading_rad), 1e-6)
                + (-rule_local_dx[source_uturn]) / max(abs(float(uturn_longitudinal)), 0.05)
            )
            source_rule_parts.append(F.relu(source_uturn_score - (1.0 - float(rule_margin))).mean())
    if source_rule_parts:
        losses.append(0.5 * float(rule_weight) * torch.stack(source_rule_parts).mean())

    valid_base_desc = {key: value[valid] for key, value in base_desc.items()}
    valid_cf_desc = {key: value[valid] for key, value in cf_desc.items()}
    collateral = _target_aware_direction_collateral(
        valid_base_desc,
        valid_cf_desc,
        valid_target,
        ids,
        margin=collateral_margin,
    ).to(dtype=cf_pred.dtype)
    if float(collateral_weight) > 0:
        losses.append(float(collateral_weight) * collateral)

    if losses:
        total = torch.stack([loss.to(dtype=cf_pred.dtype) for loss in losses]).sum()
    else:
        total = cf_pred.new_tensor(0.0)
    if proxy_hits:
        proxy_success = torch.cat([item.to(dtype=cf_pred.dtype) for item in proxy_hits], dim=0).mean()
    else:
        proxy_success = cf_pred.new_tensor(0.0)
    stats = {
        "hybrid_direction_cf": float(total.detach().item()),
        "hybrid_direction_cf_auto": float(auto_loss.detach().item()),
        "hybrid_direction_cf_rule": float(rule_loss.detach().item()),
        "hybrid_direction_cf_effect": float(effect_loss.detach().item()),
        "hybrid_direction_cf_success_proxy": float(proxy_success.detach().item()),
        "hybrid_direction_cf_auto_success": float(auto_success.detach().item()),
        "hybrid_direction_cf_rule_success": float(rule_success.detach().item()),
        "hybrid_direction_cf_collateral": float(collateral.detach().item()),
        "hybrid_direction_cf_local_collateral": float(local_collateral.detach().item()),
    }
    return total, stats


def eval_aligned_direction_counterfactual_loss(
    base_pred: torch.Tensor,
    cf_pred: torch.Tensor,
    reference: torch.Tensor,
    base_tokens: torch.Tensor,
    cf_tokens: torch.Tensor,
    target_margin: float = 0.50,
    class_margin: float = 0.25,
    collateral_margin: float = 0.20,
    ce_weight: float = 1.0,
    effect_weight: float = 1.0,
    collateral_weight: float = 0.25,
    straight_heading_rad: float = math.pi / 6.0,
    straight_lateral: float = 0.08,
    turn_margin_rad: float = 0.35,
    uturn_heading_rad: float = 2.0,
    uturn_longitudinal: float = -0.05,
    multiturn_total_abs_rad: float = 4.5,
    multiturn_soft_count: float = 3.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Counterfactual direction loss aligned with Table-3 hybrid semantics.

    The generic Table-4 loss can reward movement toward a calibrated scalar
    center even when the final trajectory is still not classified as the target
    direction. This loss adds a paired objective in the same local-frame
    descriptor space used by the direction diagnostic: the changed-token
    trajectory must rank the target direction above competing directions and
    improve the target score relative to the unchanged branch.
    """
    del reference
    field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("direction")
    target = cf_tokens[:, field_idx].long()
    source = base_tokens[:, field_idx].long()
    base_desc = _local_direction_descriptors(base_pred.float())
    cf_desc = _local_direction_descriptors(cf_pred.float())
    base_logits = _eval_aligned_direction_logits(
        base_desc,
        straight_heading_rad=straight_heading_rad,
        straight_lateral=straight_lateral,
        turn_margin_rad=turn_margin_rad,
        uturn_heading_rad=uturn_heading_rad,
        uturn_longitudinal=uturn_longitudinal,
        multiturn_total_abs_rad=multiturn_total_abs_rad,
        multiturn_soft_count=multiturn_soft_count,
    ).to(dtype=cf_pred.dtype)
    cf_logits = _eval_aligned_direction_logits(
        cf_desc,
        straight_heading_rad=straight_heading_rad,
        straight_lateral=straight_lateral,
        turn_margin_rad=turn_margin_rad,
        uturn_heading_rad=uturn_heading_rad,
        uturn_longitudinal=uturn_longitudinal,
        multiturn_total_abs_rad=multiturn_total_abs_rad,
        multiturn_soft_count=multiturn_soft_count,
    ).to(dtype=cf_pred.dtype)

    valid = target != source
    if not bool(valid.any()):
        zero = cf_pred.new_tensor(0.0)
        return zero, {
            "eval_direction_cf": 0.0,
            "eval_direction_cf_ce": 0.0,
            "eval_direction_cf_effect": 0.0,
            "eval_direction_cf_rank": 0.0,
            "eval_direction_cf_success": 0.0,
            "eval_direction_cf_collateral": 0.0,
        }

    base_logits = base_logits[valid]
    cf_logits = cf_logits[valid]
    target = target[valid]
    source = source[valid]

    ce = F.cross_entropy(cf_logits, target, reduction="mean")
    target_base = base_logits.gather(1, target[:, None]).squeeze(1)
    target_cf = cf_logits.gather(1, target[:, None]).squeeze(1)
    source_base = base_logits.gather(1, source[:, None]).squeeze(1)
    source_cf = cf_logits.gather(1, source[:, None]).squeeze(1)
    target_effect = target_cf - target_base
    source_drop = source_base - source_cf
    effect_loss = F.softplus(float(target_margin) - target_effect).mean()
    source_loss = F.softplus(float(target_margin) * 0.5 - source_drop).mean()

    competitor_logits = cf_logits.masked_fill(
        F.one_hot(target, num_classes=cf_logits.shape[1]).bool(),
        torch.finfo(cf_logits.dtype).min,
    )
    rank_gap = target_cf - competitor_logits.max(dim=1).values
    rank_loss = F.softplus(float(class_margin) - rank_gap).mean()

    ids = DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]
    rule_losses: List[torch.Tensor] = []

    def add_rule(mask: torch.Tensor, parts: List[torch.Tensor]) -> None:
        if bool(mask.any()):
            rule_losses.append(torch.stack([part[mask].mean() for part in parts]).mean())

    valid_abs_heading = cf_desc["local_abs_heading_diff_rad"][valid]
    valid_local_dy = cf_desc["local_dy"][valid]
    valid_local_dx = cf_desc["local_dx"][valid]
    valid_signed_turn = cf_desc["local_signed_total_turn_rad"][valid]
    valid_total_abs = cf_desc["local_total_abs_turn_rad"][valid]
    valid_turn_count = cf_desc["local_turn_count"][valid]

    straight_mask = target == int(ids["straight"])
    add_rule(
        straight_mask,
        [
            F.relu(valid_abs_heading - float(straight_heading_rad)) / max(float(straight_heading_rad), 1e-6),
            F.relu(valid_local_dy.abs() - float(straight_lateral)) / max(float(straight_lateral), 1e-6),
            F.relu(valid_total_abs - float(multiturn_total_abs_rad)) / max(float(multiturn_total_abs_rad), 1e-6),
            F.relu(valid_turn_count - float(multiturn_soft_count)) / max(float(multiturn_soft_count), 1e-6),
        ],
    )

    u_turn_mask = target == int(ids["u_turn"])
    add_rule(
        u_turn_mask,
        [
            F.relu(float(uturn_heading_rad) - valid_abs_heading) / max(float(uturn_heading_rad), 1e-6),
            F.relu(valid_local_dx - float(uturn_longitudinal)) / max(abs(float(uturn_longitudinal)), 0.05),
        ],
    )

    left_mask = target == int(ids["left_turn"])
    add_rule(
        left_mask,
        [
            F.relu(float(turn_margin_rad) - valid_signed_turn) / max(float(turn_margin_rad), 1e-6),
            F.relu(-valid_local_dy) / max(float(straight_lateral), 1e-6),
        ],
    )

    right_mask = target == int(ids["right_turn"])
    add_rule(
        right_mask,
        [
            F.relu(valid_signed_turn + float(turn_margin_rad)) / max(float(turn_margin_rad), 1e-6),
            F.relu(valid_local_dy) / max(float(straight_lateral), 1e-6),
        ],
    )

    multi_mask = target == int(ids["multi_turn"])
    add_rule(
        multi_mask,
        [
            F.relu(float(multiturn_total_abs_rad) - valid_total_abs) / max(float(multiturn_total_abs_rad), 1e-6),
            F.relu(float(multiturn_soft_count) - valid_turn_count) / max(float(multiturn_soft_count), 1e-6),
        ],
    )
    rule_loss = (
        torch.stack(rule_losses).mean().to(dtype=cf_pred.dtype)
        if rule_losses
        else cf_pred.new_tensor(0.0)
    )

    # Avoid solving direction by making unrelated descriptors explode while not
    # penalizing the target evidence itself, such as total turn for multi_turn.
    valid_base_desc = {key: value[valid] for key, value in base_desc.items()}
    valid_cf_desc = {key: value[valid] for key, value in cf_desc.items()}
    collateral = _target_aware_direction_collateral(
        valid_base_desc,
        valid_cf_desc,
        target,
        ids,
        margin=collateral_margin,
    ).to(dtype=cf_pred.dtype)

    total = (
        float(ce_weight) * ce
        + float(effect_weight) * (effect_loss + 0.5 * source_loss + rank_loss + rule_loss)
        + float(collateral_weight) * collateral
    )
    pred_label = cf_logits.argmax(dim=1)
    stats = {
        "eval_direction_cf": float(total.detach().item()),
        "eval_direction_cf_ce": float(ce.detach().item()),
        "eval_direction_cf_rule": float(rule_loss.detach().item()),
        "eval_direction_cf_effect": float(target_effect.detach().mean().item()),
        "eval_direction_cf_rank": float(rank_gap.detach().mean().item()),
        "eval_direction_cf_success": float((pred_label == target).float().mean().detach().item()),
        "eval_direction_cf_collateral": float(collateral.detach().item()),
    }
    return total, stats


def default_mode_anchor_targets(
    num_candidates: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Weak geometry roles for candidate modes.

    The anchors are descriptors, not target trajectories. They are meant to make
    candidate IDs stable enough for an MTR-style scorer to learn each mode's
    role, while oracle regression still decides which candidate matches the GT.
    """
    base = torch.tensor(
        [
            [0.85, 0.85, 0.012, 0.006, 0.12],  # compact, high-progress, tight/straight
            [1.00, 0.68, 0.030, 0.012, 0.35],  # medium, moderate curve
            [1.15, 0.80, 0.025, 0.014, 0.25],  # longer, high-progress
            [1.05, 0.45, 0.075, 0.020, 1.00],  # curved or low-progress mode
            [0.95, 0.60, 0.045, 0.008, 0.55],  # reference-following turning mode
            [1.20, 0.55, 0.055, 0.030, 0.75],  # loose-reference / exploratory mode
        ],
        device=device,
        dtype=dtype,
    )
    if int(num_candidates) <= base.shape[0]:
        anchors = base[: int(num_candidates)].clone()
    else:
        extra_count = int(num_candidates) - base.shape[0]
        t = torch.linspace(0.0, 1.0, extra_count, device=device, dtype=dtype).view(-1, 1)
        extra = torch.cat(
            [
                0.90 + 0.35 * t,
                0.45 + 0.40 * (1.0 - t),
                0.025 + 0.060 * t,
                0.010 + 0.030 * t,
                0.25 + 1.00 * t,
            ],
            dim=1,
        )
        anchors = torch.cat([base, extra], dim=0)
    scales = torch.tensor([0.25, 0.25, 0.050, 0.030, 0.75], device=device, dtype=dtype)
    return anchors, scales, list(MODE_ANCHOR_FEATURES)


def load_mode_anchor_targets(
    path: str | Path | None,
    num_candidates: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, List[str], str]:
    """Load mode anchors from JSON or fall back to weak defaults.

    Supported JSON formats:
    - {"features": [...], "anchors": [[...]], "scales": [...]}
    - eval_mode_specialization.py metrics containing a "per_mode" object.
    """
    if path is None:
        anchors, scales, features = default_mode_anchor_targets(num_candidates, device=device, dtype=dtype)
        return anchors, scales, features, "default"

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = payload.get("features") or payload.get("feature_names") or list(MODE_ANCHOR_FEATURES)
    if list(features) != list(MODE_ANCHOR_FEATURES):
        raise ValueError(f"Unsupported mode anchor features: {features}. Expected {MODE_ANCHOR_FEATURES}.")

    if "anchors" in payload:
        anchors_raw = payload["anchors"]
    elif "per_mode" in payload:
        per_mode = payload["per_mode"]
        anchors_raw = []
        for mode_idx in range(int(num_candidates)):
            mode_payload = per_mode[str(mode_idx)]
            anchors_raw.append([float(mode_payload[name]) for name in MODE_ANCHOR_FEATURES])
    else:
        raise ValueError(f"Mode anchor file {path} must contain 'anchors' or 'per_mode'.")

    anchors = torch.tensor(anchors_raw, device=device, dtype=dtype)
    if anchors.shape != (int(num_candidates), len(MODE_ANCHOR_FEATURES)):
        raise ValueError(
            f"Mode anchor shape {tuple(anchors.shape)} does not match "
            f"({int(num_candidates)}, {len(MODE_ANCHOR_FEATURES)})."
        )
    default_anchors, default_scales, _ = default_mode_anchor_targets(num_candidates, device=device, dtype=dtype)
    del default_anchors
    scales_raw = payload.get("scales") or payload.get("default_anchor_scales")
    scales = (
        torch.tensor(scales_raw, device=device, dtype=dtype)
        if scales_raw is not None
        else default_scales
    )
    if scales.numel() != len(MODE_ANCHOR_FEATURES):
        raise ValueError(f"Mode anchor scales must have {len(MODE_ANCHOR_FEATURES)} values, got {scales.numel()}.")
    return anchors, scales.clamp_min(1e-6), list(MODE_ANCHOR_FEATURES), str(path)


def mode_geometry_descriptors(candidates: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return [B, K, F] mode geometry descriptors in normalized local units."""
    if candidates.ndim != 4:
        candidates = candidates[:, None]
    batch, num_modes, target_len, coord_dim = candidates.shape
    flat = candidates.reshape(batch * num_modes, target_len, coord_dim)
    ref_flat = reference[:, None].expand(-1, num_modes, -1, -1).reshape(batch * num_modes, target_len, coord_dim)
    ref_len = path_length(ref_flat).view(batch, num_modes).clamp_min(1e-6)
    length_ratio_value = (path_length(flat).view(batch, num_modes) / ref_len).clamp(max=5.0)
    progress_value = displacement_ratio(flat).view(batch, num_modes).clamp(0.0, 2.0)
    curvature_value = curvature_energy(flat).view(batch, num_modes).clamp(max=math.pi)
    ref_dist_value = mean_reference_distance(flat, ref_flat).view(batch, num_modes).clamp(max=2.0)
    heading_value = heading_change(flat).abs().view(batch, num_modes).clamp(max=math.pi)
    return torch.stack(
        [
            length_ratio_value,
            progress_value,
            curvature_value,
            ref_dist_value,
            heading_value,
        ],
        dim=-1,
    )


def mode_anchor_loss(
    candidates: torch.Tensor,
    reference: torch.Tensor,
    anchors: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Weak mode-role regularization over candidate geometry descriptors."""
    if candidates.ndim != 4 or candidates.shape[1] < 2:
        zero = candidates.new_tensor(0.0)
        return zero, {"mode_anchor": 0.0, "mode_anchor_mean_abs_z": 0.0}
    descriptors = mode_geometry_descriptors(candidates.float(), reference.float())
    anchors = anchors.to(device=candidates.device, dtype=descriptors.dtype)
    scales = scales.to(device=candidates.device, dtype=descriptors.dtype).clamp_min(1e-6)
    if anchors.shape[0] != candidates.shape[1]:
        raise ValueError(f"Mode anchor count {anchors.shape[0]} does not match num_candidates {candidates.shape[1]}.")
    if anchors.shape[1] != descriptors.shape[-1] or scales.numel() != descriptors.shape[-1]:
        raise ValueError(
            f"Mode anchor feature shape mismatch: anchors={tuple(anchors.shape)}, scales={tuple(scales.shape)}, "
            f"expected feature dim {descriptors.shape[-1]}."
        )
    z = (descriptors - anchors.view(1, anchors.shape[0], anchors.shape[1])) / scales.view(1, 1, -1)
    loss = F.smooth_l1_loss(z, torch.zeros_like(z), reduction="mean")
    stats = {
        "mode_anchor": float(loss.detach().item()),
        "mode_anchor_mean_abs_z": float(z.detach().abs().mean().item()),
    }
    for idx, name in enumerate(MODE_ANCHOR_FEATURES):
        stats[f"mode_anchor_{name}_abs_z"] = float(z[..., idx].detach().abs().mean().item())
    return loss, stats


def mode_role_contrastive_geometry_loss(
    candidates: torch.Tensor,
    reference: torch.Tensor,
    margin: float = 0.15,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Relative candidate-role regularization for K-mode geometry.

    Absolute anchors can be brittle across cities or calibration variants. This
    loss only asks role-specific modes to be ordered correctly inside the same
    candidate set: mode 0 should be smoother/closer than loose or winding
    modes, mode 3 should be more winding than smooth/normal modes, and progress
    modes should separate along progress/length. This follows the MTR-style
    idea that motion modes need stable identities before score calibration.
    """
    if candidates.ndim != 4 or candidates.shape[1] < 6:
        zero = candidates.new_tensor(0.0)
        return zero, {
            "mode_role_contrast": 0.0,
            "mode_role_contrast_success": 0.0,
            "mode_role_contrast_close": 0.0,
            "mode_role_contrast_winding": 0.0,
            "mode_role_contrast_progress": 0.0,
        }
    descriptors = mode_geometry_descriptors(candidates.float(), reference.float())
    length = descriptors[..., 0]
    progress = descriptors[..., 1]
    curvature = descriptors[..., 2]
    ref_dist = descriptors[..., 3]
    heading = descriptors[..., 4]
    margin = float(margin)

    terms: List[torch.Tensor] = []
    effects: List[torch.Tensor] = []

    def add(effect: torch.Tensor) -> torch.Tensor:
        effects.append(effect)
        loss_item = F.softplus(margin - effect).mean()
        terms.append(loss_item)
        return loss_item

    close_effects = [
        ref_dist[:, 2] - ref_dist[:, 0],
        ref_dist[:, 3] - ref_dist[:, 0],
        curvature[:, 3] - curvature[:, 0],
        heading[:, 3] - heading[:, 0],
    ]
    close_loss = torch.stack([add(effect) for effect in close_effects]).mean()

    winding_effects = [
        curvature[:, 3] - curvature[:, 0],
        curvature[:, 3] - curvature[:, 1],
        heading[:, 3] - heading[:, 0],
        heading[:, 3] - heading[:, 1],
    ]
    winding_loss = torch.stack([add(effect) for effect in winding_effects]).mean()

    progress_effects = [
        progress[:, 4] - progress[:, 5],
        length[:, 5] - length[:, 0],
        length[:, 2] - length[:, 0],
    ]
    progress_loss = torch.stack([add(effect) for effect in progress_effects]).mean()

    total = torch.stack(terms).mean() if terms else candidates.new_tensor(0.0)
    effect_tensor = torch.stack(effects, dim=1) if effects else candidates.new_zeros(candidates.shape[0], 1)
    stats = {
        "mode_role_contrast": float(total.detach().item()),
        "mode_role_contrast_success": float((effect_tensor.detach() > 0.0).float().mean().item()),
        "mode_role_contrast_close": float(close_loss.detach().item()),
        "mode_role_contrast_winding": float(winding_loss.detach().item()),
        "mode_role_contrast_progress": float(progress_loss.detach().item()),
    }
    return total.to(dtype=candidates.dtype), stats


def mode_role_token_compatibility_loss(
    candidate_logits: torch.Tensor,
    tokens: torch.Tensor,
    temperature: float = 0.7,
    label_smoothing: float = 0.03,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Align candidate scores with instruction-token mode roles.

    The mapping follows the current K=6 role design:
    0 close/smooth, 1 normal/medium, 2 loose-reference, 3 winding/multi-turn,
    4 fast-progress, 5 slow-progress. It is a soft distribution rather than a
    hard mode penalty, so ADE-aware reranking can still override it later.
    """
    if candidate_logits.ndim != 2 or candidate_logits.shape[1] < 2:
        zero = candidate_logits.new_tensor(0.0)
        return zero, {
            "mode_role_compat": 0.0,
            "mode_role_compat_acc": 0.0,
            "mode_role_target_entropy": 0.0,
        }
    fields = DEFAULT_MOBILITY_VOCAB.fields
    field_idx = {field: idx for idx, field in enumerate(fields)}
    ids = DEFAULT_MOBILITY_VOCAB.token_to_id
    batch, num_modes = candidate_logits.shape
    scores = candidate_logits.new_zeros(batch, max(num_modes, 6))
    scores[:, 1] += 0.20

    distance = tokens[:, field_idx["distance"]]
    direction = tokens[:, field_idx["direction"]]
    curvature = tokens[:, field_idx["curvature"]]
    progress = tokens[:, field_idx["progress"]]
    reference = tokens[:, field_idx["reference_adherence"]]

    scores[:, 5] += 0.40 * (
        (distance == ids["distance"]["very_short"]) | (distance == ids["distance"]["short"])
    ).to(scores.dtype)
    scores[:, 1] += 0.25 * (distance == ids["distance"]["medium"]).to(scores.dtype)
    scores[:, 4] += 0.35 * (distance == ids["distance"]["long"]).to(scores.dtype)

    scores[:, 0] += 0.50 * (direction == ids["direction"]["straight"]).to(scores.dtype)
    scores[:, 1] += 0.20 * (
        (direction == ids["direction"]["left_turn"]) | (direction == ids["direction"]["right_turn"])
    ).to(scores.dtype)
    scores[:, 3] += 0.75 * (
        (direction == ids["direction"]["u_turn"]) | (direction == ids["direction"]["multi_turn"])
    ).to(scores.dtype)

    scores[:, 0] += 0.60 * (curvature == ids["curvature"]["smooth"]).to(scores.dtype)
    scores[:, 1] += 0.45 * (curvature == ids["curvature"]["moderate"]).to(scores.dtype)
    scores[:, 3] += 0.85 * (curvature == ids["curvature"]["winding"]).to(scores.dtype)

    scores[:, 5] += 0.70 * (progress == ids["progress"]["slow"]).to(scores.dtype)
    scores[:, 1] += 0.35 * (progress == ids["progress"]["normal"]).to(scores.dtype)
    scores[:, 4] += 0.80 * (progress == ids["progress"]["fast"]).to(scores.dtype)

    scores[:, 2] += 0.65 * (
        (reference == ids["reference_adherence"]["unknown"])
        | (reference == ids["reference_adherence"]["loose"])
    ).to(scores.dtype)
    scores[:, 0] += 0.65 * (reference == ids["reference_adherence"]["tight"]).to(scores.dtype)
    scores[:, 1] += 0.45 * (reference == ids["reference_adherence"]["moderate"]).to(scores.dtype)

    scores = scores[:, :num_modes]
    temp = max(float(temperature), 1e-6)
    target = torch.softmax(scores / temp, dim=1)
    smooth = max(0.0, min(float(label_smoothing), 0.5))
    if smooth > 0.0:
        target = (1.0 - smooth) * target + smooth / float(num_modes)
    log_probs = F.log_softmax(candidate_logits, dim=1)
    loss = -(target.detach() * log_probs).sum(dim=1).mean()
    target_idx = target.argmax(dim=1)
    stats = {
        "mode_role_compat": float(loss.detach().item()),
        "mode_role_compat_acc": float((candidate_logits.argmax(dim=1) == target_idx).float().mean().detach().item()),
        "mode_role_target_entropy": float((-(target * target.clamp_min(1e-8).log()).sum(dim=1).mean()).detach().item()),
    }
    for mode_idx in range(num_modes):
        stats[f"mode_role_target_rate_{mode_idx}"] = float((target_idx == mode_idx).float().mean().detach().item())
    return loss, stats


def geometry_control_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: float = 1.0,
    velocity_weight: float = 0.2,
    coarse_weight: float = 0.5,
    length_weight: float = 0.05,
    reference_weight: float = 0.02,
    coarse_points: int = 32,
    reference_margin: float = 0.03,
    reference_beta: float = 0.01,
    reference_norm: float = 0.08,
    route_grid_weight: float = 0.0,
    route_grid_points: int = 32,
    route_grid_size: int = 12,
    route_grid_temperature: float = 0.08,
    route_transition_weight: float = 0.0,
    route_transition_points: int = 32,
    route_transition_grid_size: int = 12,
    route_transition_beta: float = 0.08,
    return_stats: bool = True,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """AnchorTraj-inspired geometry loss in normalized local-meter coordinates."""
    pred = pred.float()
    target = target.float()
    reference = reference.float()
    pos = masked_smooth_l1(pred, target, mask)
    vel = velocity_smooth_l1_loss(pred, target, mask)
    coarse = coarse_skeleton_loss(pred, target, coarse_points, mask)
    length = length_ratio_loss(pred, target)
    route_grid = (
        route_grid_alignment_loss(
            pred,
            target,
            coarse_points=route_grid_points,
            grid_size=route_grid_size,
            temperature=route_grid_temperature,
        )
        if float(route_grid_weight) > 0.0
        else pred.new_tensor(0.0)
    )
    route_transition = (
        route_transition_alignment_loss(
            pred,
            target,
            coarse_points=route_transition_points,
            grid_size=route_transition_grid_size,
            beta=route_transition_beta,
        )
        if float(route_transition_weight) > 0.0
        else pred.new_tensor(0.0)
    )
    ref = (
        soft_reference_loss(
            pred,
            reference,
            margin=reference_margin,
            beta=reference_beta,
            norm=reference_norm,
        )
        if float(reference_weight) > 0.0
        else pred.new_tensor(0.0)
    )
    total = (
        float(pos_weight) * pos
        + float(velocity_weight) * vel
        + float(coarse_weight) * coarse
        + float(length_weight) * length
        + float(route_grid_weight) * route_grid
        + float(route_transition_weight) * route_transition
        + float(reference_weight) * ref
    )
    if not bool(return_stats):
        return total, {}
    stats = {
        "pos": float(pos.detach().item()),
        "velocity": float(vel.detach().item()),
        "coarse": float(coarse.detach().item()),
        "length": float(length.detach().item()),
        "route_grid": float(route_grid.detach().item()),
        "route_transition": float(route_transition.detach().item()),
        "reference": float(ref.detach().item()),
        "geometry": float(total.detach().item()),
    }
    return total, stats


def _field_ids(field: str) -> Dict[str, int]:
    return DEFAULT_MOBILITY_VOCAB.token_to_id[field]


def _token_mask(tokens: torch.Tensor, field: str, token_name: str) -> torch.Tensor:
    field_idx = _FIELD_INDEX[field]
    token_id = _field_ids(field)[token_name]
    return tokens[:, field_idx] == int(token_id)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return values.new_tensor(0.0)
    return values[mask].mean()


def token_consistency_loss(pred: torch.Tensor, reference: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
    """Differentiable rule loss that ties mobility tokens to geometry."""
    losses: List[torch.Tensor] = []
    stats: Dict[str, float] = {}

    ref_dist = mean_reference_distance(pred, reference)
    curv = turn_density_score(pred)
    progress = displacement_ratio(pred)
    turn = heading_change(pred)

    tight = _token_mask(tokens, "reference_adherence", "tight")
    moderate_ref = _token_mask(tokens, "reference_adherence", "moderate")
    loose = _token_mask(tokens, "reference_adherence", "loose")
    losses.append(_masked_mean(ref_dist.pow(2), tight))
    losses.append(_masked_mean((ref_dist - 0.015).pow(2), moderate_ref))
    losses.append(_masked_mean(F.relu(0.035 - ref_dist).pow(2), loose))

    smooth = _token_mask(tokens, "curvature", "smooth")
    moderate_curv = _token_mask(tokens, "curvature", "moderate")
    winding = _token_mask(tokens, "curvature", "winding")
    losses.append(_masked_mean(curv.pow(2), smooth))
    losses.append(_masked_mean((curv - 0.03).pow(2), moderate_curv))
    losses.append(_masked_mean(F.relu(0.06 - curv).pow(2), winding))

    slow = _token_mask(tokens, "progress", "slow")
    normal = _token_mask(tokens, "progress", "normal")
    fast = _token_mask(tokens, "progress", "fast")
    losses.append(_masked_mean(F.relu(progress - 0.45).pow(2), slow))
    losses.append(_masked_mean((progress - 0.65).pow(2), normal))
    losses.append(_masked_mean(F.relu(0.75 - progress).pow(2), fast))

    straight = _token_mask(tokens, "direction", "straight")
    left = _token_mask(tokens, "direction", "left_turn")
    right = _token_mask(tokens, "direction", "right_turn")
    u_turn = _token_mask(tokens, "direction", "u_turn")
    multi_turn = _token_mask(tokens, "direction", "multi_turn")
    losses.append(_masked_mean(F.relu(turn.abs() - 0.35).pow(2), straight))
    losses.append(_masked_mean(F.relu(0.35 - turn).pow(2), left))
    losses.append(_masked_mean(F.relu(turn + 0.35).pow(2), right))
    losses.append(_masked_mean(F.relu(1.5 - turn.abs()).pow(2), u_turn))
    losses.append(_masked_mean(F.relu(0.06 - curv).pow(2), multi_turn))

    total = torch.stack(losses).mean()
    stats["token_loss"] = float(total.detach().item())
    stats["ref_dist_norm"] = float(ref_dist.detach().mean().item())
    stats["curvature"] = float(curv.detach().mean().item())
    stats["progress"] = float(progress.detach().mean().item())
    stats["turn_abs"] = float(turn.detach().abs().mean().item())
    return total, stats


def _resample_like(points: torch.Tensor, target_len: int) -> torch.Tensor:
    if points.shape[1] == target_len:
        return points
    return F.interpolate(points.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)


def _line_between_endpoints(reference: torch.Tensor) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, reference.shape[1], device=reference.device, dtype=reference.dtype)
    t = t.view(1, -1, 1)
    return reference[:, :1] * (1.0 - t) + reference[:, -1:] * t


def _sine_offset(reference: torch.Tensor, amplitude: torch.Tensor, cycles: float) -> torch.Tensor:
    tangent, normal = AnchorResidualDecoder.tangent_normal(reference)
    t = torch.linspace(0.0, 1.0, reference.shape[1], device=reference.device, dtype=reference.dtype)
    wave = torch.sin(t * cycles * torch.pi).view(1, -1, 1)
    return reference + amplitude.view(-1, 1, 1) * wave * normal


def build_counterfactual_target(reference: torch.Tensor, field: str, token_ids: torch.Tensor) -> torch.Tensor:
    if field == "reference_adherence":
        ids = _field_ids(field)
        amp = torch.zeros(reference.shape[0], device=reference.device, dtype=reference.dtype)
        amp = torch.where(token_ids == ids["moderate"], torch.full_like(amp, 0.015), amp)
        amp = torch.where(token_ids == ids["loose"], torch.full_like(amp, 0.040), amp)
        amp = torch.where(token_ids == ids["unknown"], torch.full_like(amp, 0.025), amp)
        return _sine_offset(reference, amp, cycles=1.0)

    if field == "curvature":
        ids = _field_ids(field)
        line = _line_between_endpoints(reference)
        smooth = 0.7 * line + 0.3 * reference
        moderate = 0.5 * reference + 0.5 * _sine_offset(reference, reference.new_full((reference.shape[0],), 0.012), cycles=2.0)
        winding = _sine_offset(reference, reference.new_full((reference.shape[0],), 0.030), cycles=4.0)
        target = reference
        target = torch.where((token_ids == ids["smooth"]).view(-1, 1, 1), smooth, target)
        target = torch.where((token_ids == ids["moderate"]).view(-1, 1, 1), moderate, target)
        target = torch.where((token_ids == ids["winding"]).view(-1, 1, 1), winding, target)
        return target

    if field == "progress":
        ids = _field_ids(field)
        line = _line_between_endpoints(reference)
        slow = _sine_offset(reference, reference.new_full((reference.shape[0],), 0.030), cycles=2.0)
        fast = 0.85 * line + 0.15 * reference
        target = reference
        target = torch.where((token_ids == ids["slow"]).view(-1, 1, 1), slow, target)
        target = torch.where((token_ids == ids["fast"]).view(-1, 1, 1), fast, target)
        return target

    return reference


def sample_counterfactual_tokens(tokens: torch.Tensor, field: str) -> torch.Tensor:
    field_idx = DEFAULT_MOBILITY_VOCAB.fields.index(field)
    vocab_size = len(DEFAULT_MOBILITY_VOCAB.token_to_id[field])
    sampled = torch.randint(0, vocab_size - 1, (tokens.shape[0],), device=tokens.device)
    current = tokens[:, field_idx]
    sampled = sampled + (sampled >= current).long()
    counter_tokens = tokens.clone()
    counter_tokens[:, field_idx] = sampled
    return counter_tokens


def load_counterfactual_transition_table(
    path: str | Path | None,
    min_success_rate: float = 0.5,
    min_effect_z: float = 0.0,
    min_pairs: int = 1,
    weight_temperature: float = 1.0,
    min_token_match_rate: float = 0.0,
    max_collateral_z: float = 0.0,
    field_min_success_rate: Dict[str, float] | None = None,
    field_min_effect_z: Dict[str, float] | None = None,
    field_min_pairs: Dict[str, int] | None = None,
    field_min_token_match_rate: Dict[str, float] | None = None,
    field_max_collateral_z: Dict[str, float] | None = None,
    allow_transitions: str | None = None,
) -> Dict[str, Dict[int, List[tuple[int, float]]]]:
    """Build per-field source-token -> valid target-token sampler from diagnostics.

    The diagnostics file is produced by
    ``langdirecttraj.data.diagnose_counterfactual_transitions``.  We keep only
    transitions whose pseudo target effect and success rate indicate a usable
    positive counterfactual pair.  This follows the contrastive-data lesson from
    iMotion-LLM: train on semantically meaningful alternatives instead of random
    token flips.
    """
    if path is None or str(path) == "":
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_transition = payload.get("by_transition", {})
    table: Dict[str, Dict[int, List[tuple[int, float]]]] = {}
    allow_set: set[str] | None = None
    if allow_transitions is not None and str(allow_transitions).strip():
        allow_set = {
            item.strip()
            for item in str(allow_transitions).split(",")
            if item.strip() and item.strip().lower() not in {"none", "all"}
        }
        if not allow_set:
            allow_set = None

    def _finite_or_none(value: object) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def _summary_mean(stats: dict, preferred: str, fallback: str) -> float:
        value = stats.get(preferred)
        if not isinstance(value, dict):
            value = stats.get(fallback)
        if isinstance(value, dict):
            parsed = _finite_or_none(value.get("mean"))
            if parsed is not None:
                return parsed
        fallback_value = stats.get(fallback)
        if isinstance(fallback_value, dict):
            parsed = _finite_or_none(fallback_value.get("mean"))
            if parsed is not None:
                return parsed
        parsed = _finite_or_none(value)
        return parsed if parsed is not None else 0.0

    for key, stats in by_transition.items():
        if ":" not in key or "->" not in key:
            continue
        if allow_set is not None and key not in allow_set:
            continue
        field, transition = key.split(":", 1)
        source_token, target_token = transition.split("->", 1)
        if field not in DEFAULT_MOBILITY_VOCAB.fields:
            continue
        token_to_id = DEFAULT_MOBILITY_VOCAB.token_to_id[field]
        if source_token not in token_to_id or target_token not in token_to_id:
            continue
        num_pairs = int(stats.get("num_pairs", stats.get("num_cases", 0)))
        success_rate = _finite_or_none(stats.get("pseudo_target_success_rate", stats.get("success_rate", 0.0))) or 0.0
        effect = _summary_mean(stats, "pseudo_target_effect_z", "target_effect_z")
        collateral = _summary_mean(stats, "independent_collateral_drift_z", "collateral_drift_z")
        token_match_rate = _finite_or_none(stats.get("predicted_target_match_rate", 1.0)) or 1.0
        required_pairs = int((field_min_pairs or {}).get(field, min_pairs))
        required_success = float((field_min_success_rate or {}).get(field, min_success_rate))
        required_effect = float((field_min_effect_z or {}).get(field, min_effect_z))
        required_token_match = float((field_min_token_match_rate or {}).get(field, min_token_match_rate))
        allowed_collateral = float((field_max_collateral_z or {}).get(field, max_collateral_z))
        if num_pairs < required_pairs:
            continue
        if success_rate < required_success or effect < required_effect:
            continue
        if token_match_rate < required_token_match:
            continue
        if allowed_collateral > 0.0 and collateral > allowed_collateral:
            continue
        # Keep a small positive floor so low-effect but valid transitions remain
        # sampleable when explicitly allowed by thresholds.
        collateral_quality = 1.0 / (1.0 + max(collateral, 0.0))
        quality = max(effect, 0.0) * max(success_rate, 0.0) * max(token_match_rate, 1e-3) * collateral_quality
        if not math.isfinite(quality):
            continue
        weight = max(quality, 1e-3) ** max(float(weight_temperature), 1e-6)
        if not math.isfinite(weight) or weight <= 0.0:
            continue
        table.setdefault(field, {}).setdefault(int(token_to_id[source_token]), []).append(
            (int(token_to_id[target_token]), float(weight))
        )
    return table


def sample_transition_aware_counterfactual_tokens(
    tokens: torch.Tensor,
    field: str,
    transition_table: Dict[str, Dict[int, List[tuple[int, float]]]] | None,
    fallback: Literal["random", "skip"] = "random",
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, dict]:
    """Sample changed tokens from a transition-quality table.

    Returns ``(counter_tokens, valid_mask, quality_weights)``.  When
    ``fallback='skip'``, samples without a valid transition are marked invalid
    and left unchanged; callers can skip the field if all examples are invalid.
    When ``fallback='random'``, old uniform sampling is used for missing source
    tokens with unit quality weights.
    """
    if not transition_table or field not in transition_table:
        counter_tokens = sample_counterfactual_tokens(tokens, field)
        valid = torch.ones(tokens.shape[0], device=tokens.device, dtype=torch.bool)
        quality = torch.ones(tokens.shape[0], device=tokens.device, dtype=torch.float32)
        edit_mask, preserve_mask, slot_mask, proposal_families, mask_families = build_counterfactual_edit_masks(
            tokens,
            counter_tokens,
            field,
            16,
        )
        return counter_tokens, valid, quality, {
            "proposal_family": proposal_families,
            "mask_family": mask_families,
            "edit_mask": edit_mask,
            "preserve_mask": preserve_mask,
            "slot_mask": slot_mask,
        }

    field_idx = _FIELD_INDEX[field]
    counter_tokens = tokens.clone()
    valid_mask = torch.zeros(tokens.shape[0], device=tokens.device, dtype=torch.bool)
    quality_weights = torch.zeros(tokens.shape[0], device=tokens.device, dtype=torch.float32)
    field_table = transition_table.get(field, {})
    source_ids = tokens[:, field_idx].long()
    unique_source_ids = torch.unique(source_ids.detach()).cpu().tolist()
    for source_id in unique_source_ids:
        candidates = field_table.get(int(source_id), [])
        row_mask = source_ids == int(source_id)
        row_indices = row_mask.nonzero(as_tuple=False).view(-1)
        if row_indices.numel() == 0:
            continue
        if candidates:
            target_ids = torch.tensor([item[0] for item in candidates], device=tokens.device, dtype=torch.long)
            weights = torch.tensor([item[1] for item in candidates], device=tokens.device, dtype=torch.float32)
            finite = torch.isfinite(weights) & (weights > 0)
            if not bool(finite.any()):
                continue
            target_ids = target_ids[finite]
            weights = weights[finite].clamp_min(1e-8)
            probs = weights / weights.sum().clamp_min(1e-8)
            choices = torch.multinomial(probs, int(row_indices.numel()), replacement=True)
            counter_tokens[row_indices, field_idx] = target_ids[choices]
            valid_mask[row_indices] = True
            quality_weights[row_indices] = weights[choices].detach()
        elif fallback == "random":
            random_tokens = sample_counterfactual_tokens(tokens[row_indices], field)
            counter_tokens[row_indices, field_idx] = random_tokens[:, field_idx]
            valid_mask[row_indices] = True
            quality_weights[row_indices] = 1.0
    if not bool(valid_mask.any()):
        return None, valid_mask, quality_weights, {}
    edit_mask, preserve_mask, slot_mask, proposal_families, mask_families = build_counterfactual_edit_masks(
        tokens,
        counter_tokens,
        field,
        16,
    )
    return counter_tokens, valid_mask, quality_weights, {
        "proposal_family": proposal_families,
        "mask_family": mask_families,
        "edit_mask": edit_mask,
        "preserve_mask": preserve_mask,
        "slot_mask": slot_mask,
    }


def sample_preference_counterfactual_tokens(
    tokens: torch.Tensor,
    field: str,
    transition_table: Dict[str, Dict[int, List[tuple[int, float]]]] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Sample chosen/rejected changed-token pairs for preference training.

    ``chosen`` comes from transition-supported targets. ``rejected`` is a
    different target token that is not in the supported target set for the same
    source token when possible. Samples without a supported chosen transition
    are marked invalid and should be skipped by the caller.
    """
    if not transition_table or field not in transition_table:
        valid = torch.zeros(tokens.shape[0], device=tokens.device, dtype=torch.bool)
        return None, None, valid
    field_idx = DEFAULT_MOBILITY_VOCAB.fields.index(field)
    vocab_size = len(DEFAULT_MOBILITY_VOCAB.token_to_id[field])
    field_table = transition_table.get(field, {})
    chosen_tokens = tokens.clone()
    rejected_tokens = tokens.clone()
    valid_mask = torch.zeros(tokens.shape[0], device=tokens.device, dtype=torch.bool)
    all_ids = set(range(vocab_size))
    for row_idx, source_id in enumerate(tokens[:, field_idx].long().detach().cpu().tolist()):
        positives = field_table.get(int(source_id), [])
        if not positives:
            continue
        pos_ids = [item[0] for item in positives]
        weights = torch.tensor([item[1] for item in positives], device=tokens.device, dtype=torch.float32).clamp_min(1e-8)
        choice = torch.multinomial(weights / weights.sum(), 1).item()
        chosen_id = int(pos_ids[int(choice)])
        rejected_pool = sorted(all_ids.difference({int(source_id)}).difference(set(pos_ids)))
        if not rejected_pool:
            continue
        rejected_id = int(rejected_pool[int(torch.randint(0, len(rejected_pool), (1,), device=tokens.device).item())])
        chosen_tokens[row_idx, field_idx] = chosen_id
        rejected_tokens[row_idx, field_idx] = rejected_id
        valid_mask[row_idx] = True
    if not bool(valid_mask.any()):
        return None, None, valid_mask
    return chosen_tokens, rejected_tokens, valid_mask


def parse_counterfactual_fields(raw_fields: Iterable[str]) -> List[str]:
    fields = []
    for field in raw_fields:
        if field.lower() in {"none", "off", "false"}:
            continue
        if field not in DEFAULT_MOBILITY_VOCAB.fields:
            raise ValueError(f"Unknown counterfactual field: {field}")
        fields.append(field)
    return fields
