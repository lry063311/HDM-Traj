from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import torch

from langdirecttraj.eval.instruction_geometry import contract_labels_from_tensors
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.train.control_losses import (
    _bilinear_cell_ids_weights,
    _local_prediction_to_lonlat,
    _od_group_index,
    load_global_transition_prior,
)


def load_selection_prior(path: str | Path | None, device: torch.device) -> dict | None:
    if path is None or str(path) == "":
        return None
    return load_global_transition_prior(path, device=device)


def _od_cell_candidate_scores(
    candidates: torch.Tensor,
    reference_lonlat: torch.Tensor,
    route_scales: torch.Tensor,
    prior: dict,
) -> torch.Tensor:
    """Lower is better. Uses only aggregate OD-bucket cell statistics."""
    if candidates.ndim != 4:
        candidates = candidates[:, None]
    batch, num_modes, seq_len, dim = candidates.shape
    flat = candidates.reshape(batch * num_modes, seq_len, dim).float()
    ref_flat = reference_lonlat[:, None].expand(-1, num_modes, -1, -1).reshape(
        batch * num_modes,
        reference_lonlat.shape[1],
        reference_lonlat.shape[2],
    ).float()
    scale_flat = route_scales.view(-1)[:, None].expand(-1, num_modes).reshape(-1).float()

    lonlat = _local_prediction_to_lonlat(flat, ref_flat, scale_flat)
    ids, weights, hard_ids = _bilinear_cell_ids_weights(lonlat, prior)
    group_idx = _od_group_index(ref_flat, prior).to(device=candidates.device)
    cell_group_nll = prior["od_cell_prior"]["cell_group_nll"].to(device=candidates.device, dtype=candidates.dtype)
    group_idx = group_idx.clamp(0, cell_group_nll.shape[0] - 1)
    nll = cell_group_nll[group_idx[:, None, None], ids]
    expected_nll = (nll * weights.to(device=candidates.device, dtype=candidates.dtype)).sum(dim=-1)

    changed_or_endpoint = torch.ones_like(hard_ids, dtype=torch.bool)
    if hard_ids.shape[1] > 2:
        changed_or_endpoint[:, 1:-1] = (
            (hard_ids[:, 1:-1] != hard_ids[:, :-2])
            | (hard_ids[:, 1:-1] != hard_ids[:, 2:])
        )
    mask = changed_or_endpoint.to(dtype=candidates.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    group_default = prior["od_cell_prior"]["group_default_nll"].to(device=candidates.device, dtype=candidates.dtype)
    normalizer = group_default[group_idx].clamp_min(1e-6)
    scores = (expected_nll * mask).sum(dim=1) / denom / normalizer
    return scores.view(batch, num_modes)


def _transition_candidate_scores(
    candidates: torch.Tensor,
    reference_lonlat: torch.Tensor,
    route_scales: torch.Tensor,
    prior: dict,
) -> torch.Tensor:
    """Lower is better. Scores full candidate grid transitions against Table1 prior."""
    if candidates.ndim != 4:
        candidates = candidates[:, None]
    batch, num_modes, seq_len, dim = candidates.shape
    flat = candidates.reshape(batch * num_modes, seq_len, dim).float()
    ref_flat = reference_lonlat[:, None].expand(-1, num_modes, -1, -1).reshape(
        batch * num_modes,
        reference_lonlat.shape[1],
        reference_lonlat.shape[2],
    ).float()
    scale_flat = route_scales.view(-1)[:, None].expand(-1, num_modes).reshape(-1).float()

    lonlat = _local_prediction_to_lonlat(flat, ref_flat, scale_flat)
    _, _, hard_ids = _bilinear_cell_ids_weights(lonlat, prior)
    if hard_ids.shape[1] < 2:
        return candidates.new_zeros(batch, num_modes)

    src = hard_ids[:, :-1]
    dst = hard_ids[:, 1:]
    changed = src != dst
    pair_ids = src.clamp(0, int(prior["num_cells"]) - 1) * int(prior["num_cells"]) + dst.clamp(
        0,
        int(prior["num_cells"]) - 1,
    )
    transition_nll = prior["transition_nll"].reshape(-1).to(
        device=candidates.device,
        dtype=candidates.dtype,
    )
    nll = transition_nll[pair_ids]
    mask = changed.to(dtype=candidates.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    default_nll = float(prior.get("default_nll", 1.0))
    scores = (nll * mask).sum(dim=1) / denom / max(default_nll, 1e-6)
    return scores.view(batch, num_modes)


def _length_prior_candidate_scores(
    candidates: torch.Tensor,
    reference_lonlat: torch.Tensor,
    route_scales: torch.Tensor,
    prior: dict,
) -> torch.Tensor:
    """Lower is better. OD-bucket detour-ratio score from training statistics."""
    if "od_length_prior" not in prior:
        if candidates.ndim != 4:
            candidates = candidates[:, None]
        return candidates.new_zeros(candidates.shape[:2])
    if candidates.ndim != 4:
        candidates = candidates[:, None]
    batch, num_modes, seq_len, dim = candidates.shape
    flat = candidates.reshape(batch * num_modes, seq_len, dim).float()
    scale_flat = route_scales.view(-1, 1).expand(-1, num_modes).reshape(-1).float()
    length_m = torch.linalg.norm(flat[:, 1:] - flat[:, :-1], dim=-1).sum(dim=1) * scale_flat
    ratio = length_m / scale_flat.clamp_min(100.0)
    ratio = ratio.clamp_min(1e-3).view(batch, num_modes)

    group_idx = _od_group_index(reference_lonlat.float(), prior).to(device=candidates.device)
    length_prior = prior["od_length_prior"]
    p10 = length_prior["p10"].to(device=candidates.device, dtype=candidates.dtype)
    p50 = length_prior["p50"].to(device=candidates.device, dtype=candidates.dtype)
    p90 = length_prior["p90"].to(device=candidates.device, dtype=candidates.dtype)
    group_idx = group_idx.clamp(0, p50.shape[0] - 1)
    target = p50[group_idx].view(batch, 1).clamp_min(1e-3)
    spread = (p90[group_idx] - p10[group_idx]).view(batch, 1).clamp_min(0.15)
    scores = (ratio.to(dtype=candidates.dtype) - target).abs() / spread
    return scores


def _direction_valid_mask(
    candidates: torch.Tensor,
    reference: torch.Tensor | None,
    route_scales: torch.Tensor | None,
    base_idx: torch.Tensor,
    dataset: str,
) -> torch.Tensor:
    direction_valid = torch.ones(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
    if reference is None or route_scales is None:
        return direction_valid
    flat_pred = candidates.reshape(candidates.shape[0] * candidates.shape[1], candidates.shape[2], candidates.shape[3])
    flat_ref = reference[:, None].expand(-1, candidates.shape[1], -1, -1).reshape(
        candidates.shape[0] * candidates.shape[1],
        reference.shape[1],
        reference.shape[2],
    )
    flat_scales = route_scales.view(-1, 1).expand(-1, candidates.shape[1]).reshape(-1)
    labels = contract_labels_from_tensors(
        flat_pred.float(),
        flat_ref.float(),
        flat_scales.float(),
        dataset=dataset,
    )
    direction_ids = torch.tensor(
        [
            {"straight": 0, "left_turn": 1, "right_turn": 2, "u_turn": 3, "multi_turn": 4}.get(
                str(item.get("direction", "unknown")),
                -1,
            )
            for item in labels
        ],
        device=candidates.device,
        dtype=torch.long,
    ).view(candidates.shape[0], candidates.shape[1])
    row_idx = torch.arange(candidates.shape[0], device=candidates.device)
    base_direction = direction_ids[row_idx, base_idx]
    return direction_ids == base_direction.unsqueeze(1)


def _target_direction_valid_mask(
    candidates: torch.Tensor,
    reference: torch.Tensor,
    route_scales: torch.Tensor,
    mobility_tokens: torch.Tensor,
    dataset: str,
) -> torch.Tensor:
    if mobility_tokens.ndim != 2:
        return torch.ones(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
    direction_field_idx = 1
    if candidates.shape[1] <= 0 or mobility_tokens.shape[0] != candidates.shape[0]:
        return torch.ones(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
    flat = candidates.reshape(candidates.shape[0] * candidates.shape[1], candidates.shape[2], candidates.shape[3])
    ref_flat = reference[:, None].expand(-1, candidates.shape[1], -1, -1).reshape_as(flat)
    scale_flat = route_scales[:, None].expand(-1, candidates.shape[1]).reshape(-1)
    labels = contract_labels_from_tensors(flat.float(), ref_flat.float(), scale_flat.float(), dataset=dataset)
    direction_to_id = DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]
    direction_ids = torch.tensor(
        [
            direction_to_id.get(
                str(item.get("direction", "unknown")),
                -1,
            )
            for item in labels
        ],
        device=candidates.device,
        dtype=torch.long,
    ).view(candidates.shape[0], candidates.shape[1])
    target_direction = mobility_tokens[:, direction_field_idx].to(device=candidates.device, dtype=torch.long)
    return direction_ids == target_direction.unsqueeze(1)


def _target_control_valid_mask(
    candidates: torch.Tensor,
    reference: torch.Tensor,
    route_scales: torch.Tensor,
    mobility_tokens: torch.Tensor,
    dataset: str,
    max_mismatches: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Instruction-token feasibility mask derived from generated geometry only.

    Unlike token-oracle diagnostics, this does not inspect the GT trajectory. It
    converts each candidate back to mobility contract labels and checks whether
    direction/curvature/progress/reference-adherence agree with the instruction
    tokens. The mismatch count is useful for guarded route-prior selection.
    """
    if mobility_tokens.ndim != 2 or mobility_tokens.shape[0] != candidates.shape[0]:
        valid = torch.ones(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
        return valid, torch.zeros_like(valid, dtype=torch.long)
    fields = ["direction", "curvature", "progress", "reference_adherence"]
    field_to_idx = {field: idx for idx, field in enumerate(DEFAULT_MOBILITY_VOCAB.fields)}
    flat = candidates.reshape(candidates.shape[0] * candidates.shape[1], candidates.shape[2], candidates.shape[3])
    ref_flat = reference[:, None].expand(-1, candidates.shape[1], -1, -1).reshape_as(flat)
    scale_flat = route_scales[:, None].expand(-1, candidates.shape[1]).reshape(-1)
    labels = contract_labels_from_tensors(flat.float(), ref_flat.float(), scale_flat.float(), dataset=dataset)
    mismatch = torch.zeros(candidates.shape[0], candidates.shape[1], dtype=torch.long, device=candidates.device)
    for field in fields:
        token_to_id = DEFAULT_MOBILITY_VOCAB.token_to_id[field]
        pred_ids = torch.tensor(
            [
                token_to_id.get(str(item.get(field, "unknown")), -1)
                for item in labels
            ],
            device=candidates.device,
            dtype=torch.long,
        ).view(candidates.shape[0], candidates.shape[1])
        target_ids = mobility_tokens[:, field_to_idx[field]].to(device=candidates.device, dtype=torch.long)
        valid_target = target_ids >= 0
        mismatch = mismatch + ((pred_ids != target_ids.unsqueeze(1)) & valid_target.unsqueeze(1)).to(torch.long)
    valid = mismatch <= int(max_mismatches)
    return valid, mismatch


def select_prediction_with_policy(
    pred_out,
    reference: torch.Tensor | None = None,
    reference_lonlat: torch.Tensor | None = None,
    route_scales: torch.Tensor | None = None,
    prior: dict | None = None,
    policy: str = "reranker",
    route_weight: float = 1.0,
    token_guard_margin: float = 0.0,
    token_absolute_threshold: float = float("inf"),
    min_route_improvement: float = 0.0,
    logit_guard_margin: float = float("inf"),
    preserve_direction: bool = False,
    preserve_target_direction: bool = False,
    mobility_tokens: torch.Tensor | None = None,
    dataset: str = "porto",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if not isinstance(pred_out, tuple):
        pred = pred_out
        idx = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
        return pred, idx, {}
    pred, aux = pred_out
    if pred.ndim != 4:
        idx = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
        return pred, idx, {}

    logits = aux.get("candidate_logits") if isinstance(aux, dict) else None
    if logits is None:
        logits = pred.new_zeros(pred.shape[0], pred.shape[1])
    else:
        logits = logits[:, : pred.shape[1]]

    if policy.startswith("fixed_mode_"):
        try:
            mode_idx = int(policy.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Unsupported fixed-mode selection policy: {policy}") from exc
        mode_idx = max(0, min(int(pred.shape[1]) - 1, mode_idx))
        idx = torch.full((pred.shape[0],), mode_idx, dtype=torch.long, device=pred.device)
        return pred[torch.arange(pred.shape[0], device=pred.device), idx], idx, {
            "fixed_mode_index": float(mode_idx),
        }

    if policy == "reranker" or prior is None or reference_lonlat is None or route_scales is None:
        idx = logits.argmax(dim=1)
        return pred[torch.arange(pred.shape[0], device=pred.device), idx], idx, {}
    if policy in {"od_cell_prior", "reranker_plus_od_cell_prior", "reranker_guarded_od_cell_prior"}:
        if "od_cell_prior" not in prior:
            raise ValueError("OD-cell route-aware selection requires a prior with od_cell_prior.")
        route_scores = _od_cell_candidate_scores(pred, reference_lonlat, route_scales, prior)
    elif policy in {
        "transition_prior",
        "realism_guarded_transition_prior",
        "reranker_guarded_transition_prior",
        "target_direction_transition_prior",
        "target_control_transition_prior",
        "reranker_guarded_target_control_transition_prior",
    }:
        route_scores = _transition_candidate_scores(pred, reference_lonlat, route_scales, prior)
    else:
        raise ValueError(f"Unsupported selection policy: {policy}")

    if policy == "realism_guarded_transition_prior":
        base_idx = logits.argmax(dim=1)
        row_idx = torch.arange(pred.shape[0], device=pred.device)
        length_scores = _length_prior_candidate_scores(pred, reference_lonlat, route_scales, prior)
        combined_scores = route_scores + float(route_weight) * length_scores
        base_score = combined_scores[row_idx, base_idx]
        score_gain = base_score.unsqueeze(1) - combined_scores
        base_logits = logits[row_idx, base_idx]
        logit_valid = logits >= base_logits.unsqueeze(1) - float(logit_guard_margin)
        control_valid = torch.ones_like(route_scores, dtype=torch.bool)
        mismatch = torch.zeros_like(route_scores, dtype=torch.long)
        if reference is not None and route_scales is not None and mobility_tokens is not None:
            control_valid, mismatch = _target_control_valid_mask(
                pred,
                reference,
                route_scales,
                mobility_tokens,
                dataset,
                max_mismatches=int(max(0, round(float(token_guard_margin)))),
            )
        valid = control_valid & logit_valid & (score_gain >= float(min_route_improvement))
        valid[row_idx, base_idx] = True
        masked_score = combined_scores.masked_fill(~valid, float("inf"))
        idx = masked_score.argmin(dim=1)
        changed = idx != base_idx
        stats = {
            "realism_guarded_changed_rate": float(changed.float().mean().detach().item()),
            "realism_guarded_score_gain_mean": float((base_score - combined_scores[row_idx, idx]).mean().detach().item()),
            "realism_route_score_selected_mean": float(route_scores[row_idx, idx].mean().detach().item()),
            "realism_length_score_selected_mean": float(length_scores[row_idx, idx].mean().detach().item()),
            "control_feasible_rate": float(control_valid.any(dim=1).float().mean().detach().item()),
            "control_mismatch_selected_mean": float(mismatch[row_idx, idx].float().mean().detach().item()),
        }
        return pred[row_idx, idx], idx, stats
    if policy in {"od_cell_prior", "transition_prior"}:
        idx = route_scores.argmin(dim=1)
    elif policy == "reranker_plus_od_cell_prior":
        centered = route_scores - route_scores.mean(dim=1, keepdim=True)
        scale = route_scores.std(dim=1, keepdim=True).clamp_min(1e-6)
        route_z = centered / scale
        idx = (logits - float(route_weight) * route_z.to(dtype=logits.dtype)).argmax(dim=1)
    elif policy == "target_direction_transition_prior":
        direction_valid = torch.ones_like(route_scores, dtype=torch.bool)
        if reference is not None and route_scales is not None and mobility_tokens is not None:
            direction_valid = _target_direction_valid_mask(
                pred,
                reference,
                route_scales,
                mobility_tokens,
                dataset,
            )
        has_direction_candidate = direction_valid.any(dim=1)
        masked_route = route_scores.masked_fill(~direction_valid, float("inf"))
        direction_idx = masked_route.argmin(dim=1)
        fallback_idx = logits.argmax(dim=1)
        idx = torch.where(has_direction_candidate, direction_idx, fallback_idx)
    elif policy == "target_control_transition_prior":
        control_valid = torch.ones_like(route_scores, dtype=torch.bool)
        mismatch = torch.zeros_like(route_scores, dtype=torch.long)
        if reference is not None and route_scales is not None and mobility_tokens is not None:
            control_valid, mismatch = _target_control_valid_mask(
                pred,
                reference,
                route_scales,
                mobility_tokens,
                dataset,
                max_mismatches=int(max(0, round(float(token_guard_margin)))),
            )
        has_control_candidate = control_valid.any(dim=1)
        masked_route = route_scores.masked_fill(~control_valid, float("inf"))
        control_idx = masked_route.argmin(dim=1)
        fallback_idx = logits.argmax(dim=1)
        idx = torch.where(has_control_candidate, control_idx, fallback_idx)
        stats = {
            "control_feasible_rate": float(has_control_candidate.float().mean().detach().item()),
            "control_feasible_count_mean": float(control_valid.float().sum(dim=1).mean().detach().item()),
            "control_mismatch_selected_mean": float(
                mismatch[torch.arange(pred.shape[0], device=pred.device), idx].float().mean().detach().item()
            ),
        }
        return pred[torch.arange(pred.shape[0], device=pred.device), idx], idx, stats
    elif policy == "reranker_guarded_target_control_transition_prior":
        base_idx = logits.argmax(dim=1)
        row_idx = torch.arange(pred.shape[0], device=pred.device)
        base_route = route_scores[row_idx, base_idx]
        route_gain = base_route.unsqueeze(1) - route_scores
        base_logits = logits[row_idx, base_idx]
        logit_valid = logits >= base_logits.unsqueeze(1) - float(logit_guard_margin)
        control_valid = torch.ones_like(route_scores, dtype=torch.bool)
        mismatch = torch.zeros_like(route_scores, dtype=torch.long)
        if reference is not None and route_scales is not None and mobility_tokens is not None:
            control_valid, mismatch = _target_control_valid_mask(
                pred,
                reference,
                route_scales,
                mobility_tokens,
                dataset,
                max_mismatches=int(max(0, round(float(token_guard_margin)))),
            )
        valid = (
            control_valid
            & logit_valid
            & (route_gain >= float(min_route_improvement))
        )
        valid[row_idx, base_idx] = True
        masked_route = route_scores.masked_fill(~valid, float("inf"))
        idx = masked_route.argmin(dim=1)
        changed = idx != base_idx
        stats = {
            "control_feasible_rate": float(control_valid.any(dim=1).float().mean().detach().item()),
            "control_feasible_count_mean": float(control_valid.float().sum(dim=1).mean().detach().item()),
            "control_mismatch_selected_mean": float(mismatch[row_idx, idx].float().mean().detach().item()),
            "guarded_changed_rate": float(changed.float().mean().detach().item()),
            "guarded_route_gain_mean": float((base_route - route_scores[row_idx, idx]).mean().detach().item()),
        }
        return pred[row_idx, idx], idx, stats
    elif policy in {"reranker_guarded_od_cell_prior", "reranker_guarded_transition_prior"}:
        base_idx = logits.argmax(dim=1)
        row_idx = torch.arange(pred.shape[0], device=pred.device)
        base_route = route_scores[row_idx, base_idx]
        route_gain = base_route.unsqueeze(1) - route_scores
        base_logits = logits[row_idx, base_idx]
        logit_valid = logits >= base_logits.unsqueeze(1) - float(logit_guard_margin)
        direction_valid = torch.ones_like(route_scores, dtype=torch.bool)
        if bool(preserve_direction) and reference is not None and route_scales is not None:
            direction_valid = _direction_valid_mask(pred, reference, route_scales, base_idx, dataset)
        if (
            bool(preserve_target_direction)
            and reference is not None
            and route_scales is not None
            and mobility_tokens is not None
        ):
            direction_valid = direction_valid & _target_direction_valid_mask(
                pred,
                reference,
                route_scales,
                mobility_tokens,
                dataset,
            )

        token_features = aux.get("reranker_token_features") if isinstance(aux, dict) else None
        if token_features is None:
            token_score = pred.new_zeros(route_scores.shape)
            token_valid = torch.zeros_like(route_scores, dtype=torch.bool)
            token_valid[row_idx, base_idx] = True
        else:
            token_features = token_features[:, : pred.shape[1]].to(dtype=route_scores.dtype)
            token_score = token_features.mean(dim=-1)
            base_token_features = token_features[row_idx, base_idx]
            token_valid = (
                token_features
                <= base_token_features.unsqueeze(1) + float(token_guard_margin)
            ).all(dim=-1)
            if math.isfinite(float(token_absolute_threshold)):
                token_valid = token_valid & (
                    token_features <= float(token_absolute_threshold)
                ).all(dim=-1)
        base_token = token_score[row_idx, base_idx]

        valid = (
            (route_gain >= float(min_route_improvement))
            & token_valid
            & logit_valid
            & direction_valid
        )
        valid[row_idx, base_idx] = True
        masked_route = route_scores.masked_fill(~valid, float("inf"))
        idx = masked_route.argmin(dim=1)
    stats = {
        "route_score_selected_mean": float(route_scores[torch.arange(pred.shape[0], device=pred.device), idx].mean().detach().item()),
        "route_score_best_mean": float(route_scores.min(dim=1).values.mean().detach().item()),
        "route_score_reranker_mean": float(route_scores[torch.arange(pred.shape[0], device=pred.device), logits.argmax(dim=1)].mean().detach().item()),
    }
    return pred[torch.arange(pred.shape[0], device=pred.device), idx], idx, stats
