from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from langdirecttraj.data.instruction_dataset import collate_instruction_batch
from langdirecttraj.counterfactual_policy import independent_collateral_field_names
from langdirecttraj.eval.eval_anchor_decoder import build_model, load_checkpoint, prepare_eval_batch, summarize
from langdirecttraj.eval.eval_counterfactual_controllability import (
    build_pair_calibrator,
    group_counterfactual_pairs,
    read_records,
    tensor_record,
)
from langdirecttraj.eval.instruction_geometry import (
    GeometryTokenCalibrator,
    geometry_values,
    tensor_dict_to_rows,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.train.control_losses import (
    _route_transition_alignment_per_sample,
    load_counterfactual_transition_table,
)
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _to_device_batch(record: Dict, target_key: str, device: torch.device) -> Dict:
    item = tensor_record(record, target_key=target_key)
    batch = collate_instruction_batch([item])
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _trip_duration_tensor(batch: Dict, device: torch.device) -> torch.Tensor:
    durations = []
    for attrs in batch.get("trip_attributes", []):
        if not isinstance(attrs, dict):
            attrs = {}
        durations.append(float(attrs.get("trip_duration_min", 0.0) or 0.0))
    if not durations:
        durations = [0.0]
    return torch.tensor(durations, device=device, dtype=torch.float32)


def _candidate_errors_m(candidates: torch.Tensor, target: torch.Tensor, route_scales: torch.Tensor) -> torch.Tensor:
    error = torch.linalg.norm(candidates - target[:, None], dim=-1)
    error = error * route_scales.view(-1, 1, 1).to(device=candidates.device, dtype=candidates.dtype)
    return error.mean(dim=-1)


def _aux_matrix(aux: Dict, key: str, k: int, default: float = 0.0) -> np.ndarray:
    value = aux.get(key) if isinstance(aux, dict) else None
    if not torch.is_tensor(value) or value.ndim < 2:
        return np.full((k,), float(default), dtype=np.float64)
    return value[0, :k].detach().cpu().float().numpy().astype(np.float64)


def _aux_pair_feature(aux: Dict, feature_idx: int, k: int, default: float = float("nan")) -> np.ndarray:
    value = aux.get("counterfactual_pair_features") if isinstance(aux, dict) else None
    if not torch.is_tensor(value) or value.ndim != 3 or value.shape[-1] <= feature_idx:
        return np.full((k,), float(default), dtype=np.float64)
    return value[0, :k, feature_idx].detach().cpu().float().numpy().astype(np.float64)


def _aux_scalar(aux: Dict, key: str, default: float = float("nan")) -> float:
    value = aux.get(key) if isinstance(aux, dict) else None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().cpu().float().reshape(-1)[0].item())
    return float(default)


def _generate_candidates(
    model,
    qwen_provider,
    record: Dict,
    attr_dim: int,
    target_key: str,
    device: torch.device,
    counterfactual_adapter_scale: float = 0.0,
    base_selected_trajectory: torch.Tensor | None = None,
    base_mobility_tokens: torch.Tensor | None = None,
    base_selected_index: torch.Tensor | None = None,
) -> Dict:
    batch = _to_device_batch(record, target_key, device)
    target, reference, route_scales = prepare_eval_batch(batch, model.decoder.target_len)
    attr_cond = make_attr_cond(batch, attr_dim, device)
    trip_duration_min = _trip_duration_tensor(batch, device)
    semantic_plan_tokens = extract_qwen_semantic_tokens(qwen_provider, batch["instruction"], device)
    with torch.no_grad():
        counterfactual_enabled = bool(record.get("counterfactual", {}).get("enabled"))
        counterfactual_mode = counterfactual_enabled and float(counterfactual_adapter_scale) != 0.0
        eval_selector_active = (
            counterfactual_enabled
            and float(getattr(model.decoder, "counterfactual_eval_selector_weight", 0.0)) != 0.0
        )
        cf_field_ids = None
        if counterfactual_mode or eval_selector_active:
            field_name = str(record.get("counterfactual", {}).get("field", ""))
            if field_name in DEFAULT_MOBILITY_VOCAB.fields:
                cf_field_ids = torch.full(
                    (batch["mobility_tokens"].shape[0],),
                    DEFAULT_MOBILITY_VOCAB.fields.index(field_name),
                    device=device,
                    dtype=torch.long,
                )
        pred_out = model(
            reference,
            batch["mobility_tokens"],
            attr_cond,
            semantic_plan_tokens=semantic_plan_tokens,
            counterfactual_mode=counterfactual_mode,
            counterfactual_adapter_scale=float(counterfactual_adapter_scale),
            counterfactual_field_ids=cf_field_ids,
            base_selected_trajectory=base_selected_trajectory,
            base_mobility_tokens=base_mobility_tokens,
            route_scales=route_scales[:, 0],
            base_selected_index=base_selected_index,
            trip_duration_min=trip_duration_min,
            return_aux=True,
        )
    if isinstance(pred_out, tuple):
        pred, aux = pred_out
    else:
        pred, aux = pred_out, {}
    if pred.ndim == 3:
        pred = pred[:, None]
    logits = aux.get("candidate_logits") if isinstance(aux, dict) else None
    if logits is None:
        logits = pred.new_zeros((pred.shape[0], pred.shape[1]))
    ade = _candidate_errors_m(pred, target, route_scales[:, 0])
    flat_pred = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2], pred.shape[3])
    flat_ref = reference[:, None].expand_as(pred).reshape_as(flat_pred)
    flat_scales = route_scales[:, 0].repeat_interleave(pred.shape[1])
    geom_rows = tensor_dict_to_rows(geometry_values(flat_pred.float(), flat_ref.float(), flat_scales.float()))
    flat_target = target[:, None].expand_as(pred).reshape_as(flat_pred)
    route_alignment = _route_transition_alignment_per_sample(
        flat_pred.float(),
        flat_target.float(),
        coarse_points=32,
        grid_size=12,
        beta=0.08,
    ).view(pred.shape[0], pred.shape[1])
    return {
        "candidates": pred,
        "logits": logits,
        "ade": ade,
        "route_alignment": route_alignment,
        "geom_rows": geom_rows,
        "labels": record.get("labels", {}),
        "instruction": str(record.get("instruction", "")),
        "mobility_tokens": batch["mobility_tokens"],
        "aux": aux if isinstance(aux, dict) else {},
    }


def _instruction_score(calibrator: GeometryTokenCalibrator, labels: Dict[str, str], geom: Dict[str, float]) -> float:
    values: List[float] = []
    for field_name in DEFAULT_MOBILITY_VOCAB.fields:
        token = str(labels.get(field_name, "unknown"))
        if token not in calibrator.stats.get(field_name, {}):
            continue
        values.append(calibrator.score_token(field_name, token, geom))
    return float(np.mean(values)) if values else float("nan")


def _collateral_drift(
    calibrator: GeometryTokenCalibrator,
    changed_field: str,
    original_labels: Dict[str, str],
    base_geom: Dict[str, float],
    cf_geom: Dict[str, float],
    collateral_mode: str = "independent",
) -> float:
    values: List[float] = []
    if str(collateral_mode) == "raw":
        field_names = [
            field
            for field in DEFAULT_MOBILITY_VOCAB.fields
            if field != changed_field
        ]
    else:
        field_names = independent_collateral_field_names(changed_field)
    for field_name in field_names:
        token = str(original_labels.get(field_name, "unknown"))
        if token not in calibrator.stats.get(field_name, {}):
            continue
        values.append(calibrator.normalized_metric_delta(field_name, token, base_geom, cf_geom))
    return float(np.mean(values)) if values else float("nan")


def _safe_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def evaluate(
    model,
    qwen_provider,
    pairs: List[Tuple[Dict, Dict]],
    attr_dim: int,
    target_key: str,
    calibrator: GeometryTokenCalibrator,
    device: torch.device,
    ade_temperature_m: float,
    hybrid_ade_weight: float,
    instruction_weight: float,
    target_effect_weight: float,
    collateral_weight: float,
    collateral_mode: str = "independent",
    success_margin_z: float = 0.0,
    hybrid_collateral_margin: float = 0.08,
    hybrid_priority: str = "weighted",
    route_feasible_margin: float = 0.05,
    counterfactual_adapter_scale: float = 0.0,
    baseline_mode: str = "same_mode",
) -> Dict:
    rows: List[Dict] = []
    pair_selection_gaps: List[Dict] = []
    selected_indices: List[int] = []
    oracle_indices: List[int] = []
    hybrid_indices: List[int] = []
    mode_values = defaultdict(lambda: defaultdict(list))
    num_candidates = int(getattr(model.decoder, "num_candidates", 1))

    for pair_idx, (original, counterfactual) in enumerate(pairs):
        cf_meta = counterfactual.get("counterfactual", {})
        field_name = str(cf_meta.get("field", "unknown"))
        if field_name not in DEFAULT_MOBILITY_VOCAB.fields:
            continue
        source_token = str(cf_meta.get("from", original.get("labels", {}).get(field_name, "unknown")))
        target_token = str(cf_meta.get("to", counterfactual.get("labels", {}).get(field_name, "unknown")))
        if target_token not in calibrator.stats.get(field_name, {}):
            continue

        base = _generate_candidates(
            model,
            qwen_provider,
            original,
            attr_dim,
            target_key,
            device,
            counterfactual_adapter_scale=counterfactual_adapter_scale,
        )
        base_k = int(min(base["candidates"].shape[1], num_candidates))
        base_selected_idx_tensor = torch.argmax(base["logits"][:, :base_k], dim=1)
        base_selected_trajectory = base["candidates"][
            torch.arange(base["candidates"].shape[0], device=base["candidates"].device),
            base_selected_idx_tensor,
        ]
        cf = _generate_candidates(
            model,
            qwen_provider,
            counterfactual,
            attr_dim,
            target_key,
            device,
            counterfactual_adapter_scale=counterfactual_adapter_scale,
            base_selected_trajectory=base_selected_trajectory,
            base_mobility_tokens=base["mobility_tokens"],
            base_selected_index=base_selected_idx_tensor,
        )
        k = int(min(base["candidates"].shape[1], cf["candidates"].shape[1], num_candidates))
        if k <= 0:
            continue

        cf_ade = cf["ade"][0, :k].detach().cpu().float().numpy()
        cf_route_alignment = cf["route_alignment"][0, :k].detach().cpu().float().numpy()
        cf_logits = cf["logits"][0, :k].detach().cpu().float().numpy()
        cf_aux = cf.get("aux", {})
        cf_delta = _aux_matrix(cf_aux, "counterfactual_scorer_delta", k)
        cf_policy_logits = _aux_matrix(cf_aux, "counterfactual_policy_logits", k)
        hybrid_delta = _aux_matrix(cf_aux, "hybrid_policy_delta", k)
        hybrid_raw_delta = _aux_matrix(cf_aux, "hybrid_policy_raw_delta", k)
        eval_selector_logits = _aux_matrix(cf_aux, "counterfactual_eval_selector_logits", k, default=float("nan"))
        eval_selector_cost = _aux_matrix(cf_aux, "counterfactual_eval_selector_cost", k, default=float("nan"))
        eval_selector_feasible = _aux_matrix(cf_aux, "counterfactual_eval_selector_feasible", k, default=float("nan"))
        cf_logits_np = cf_logits.astype(np.float64)
        cf_base_logits = cf_logits_np - cf_delta
        logits_without_hybrid_delta = cf_logits_np - hybrid_delta
        logits_without_hybrid_and_cf_delta = cf_logits_np - hybrid_delta - cf_delta
        pair_target_effect_proxy = _aux_pair_feature(cf_aux, 0, k)
        pair_target_score_proxy = _aux_pair_feature(cf_aux, 1, k)
        pair_collateral_proxy = _aux_pair_feature(cf_aux, 2, k)
        pair_mode_switch = _aux_pair_feature(cf_aux, 3, k)
        cf_delta_alpha = _aux_scalar(cf_aux, "counterfactual_scorer_alpha")
        cf_policy_gate = _aux_scalar(cf_aux, "counterfactual_policy_gate")
        hybrid_alpha = _aux_scalar(cf_aux, "hybrid_policy_alpha")
        selected_idx = int(np.argmax(cf_logits))
        base_selected_idx = int(base_selected_idx_tensor[0].detach().cpu().item())
        oracle_idx = int(np.argmin(cf_ade))
        best_ade = float(np.min(cf_ade))
        best_route_alignment = float(np.min(cf_route_alignment))
        hybrid_costs: List[float] = []
        hybrid_feasible: List[bool] = []
        route_feasible_flags: List[bool] = []

        for mode_idx in range(k):
            base_geom = (
                base["geom_rows"][base_selected_idx]
                if baseline_mode == "base_selected"
                else base["geom_rows"][mode_idx]
            )
            cf_geom = cf["geom_rows"][mode_idx]
            base_target_score = calibrator.score_token(field_name, target_token, base_geom)
            cf_target_score = calibrator.score_token(field_name, target_token, cf_geom)
            target_effect = float(base_target_score - cf_target_score)
            collateral = _collateral_drift(
                calibrator,
                field_name,
                original.get("labels", {}),
                base_geom,
                cf_geom,
                collateral_mode=collateral_mode,
            )
            if not np.isfinite(collateral):
                collateral = 0.0
            instruction_score = _instruction_score(calibrator, counterfactual.get("labels", {}), cf_geom)
            relative_ade = float(cf_ade[mode_idx] - best_ade)
            feasible = target_effect > float(success_margin_z) and collateral <= float(hybrid_collateral_margin)
            route_feasible = float(cf_route_alignment[mode_idx]) <= (
                best_route_alignment + max(float(route_feasible_margin), 0.0)
            )
            same_mode_feasible = bool(route_feasible and feasible)
            hybrid_cost = (
                float(hybrid_ade_weight) * relative_ade / max(float(ade_temperature_m), 1e-6)
                + float(instruction_weight) * float(cf_target_score)
                - float(target_effect_weight) * target_effect
                + float(collateral_weight) * collateral
            )
            hybrid_costs.append(float(hybrid_cost))
            hybrid_feasible.append(bool(feasible))
            route_feasible_flags.append(bool(route_feasible))
            rows.append(
                {
                    "pair_index": pair_idx,
                    "sample_id": int(original.get("sample_id", -1)),
                    "counterfactual_id": int(counterfactual.get("sample_id", counterfactual.get("counterfactual_id", -1))),
                    "field": field_name,
                    "transition": f"{field_name}:{source_token}->{target_token}",
                    "mode": mode_idx,
                    "selected": mode_idx == selected_idx,
                    "oracle_ade": mode_idx == oracle_idx,
                    "hybrid_best": False,
                    "ade_m": float(cf_ade[mode_idx]),
                    "relative_ade_m": relative_ade,
                    "route_alignment_loss": float(cf_route_alignment[mode_idx]),
                    "relative_route_alignment_loss": float(cf_route_alignment[mode_idx] - best_route_alignment),
                    "route_feasible": bool(route_feasible),
                    "instruction_score": instruction_score,
                    "target_token_score": float(cf_target_score),
                    "target_effect_z": target_effect,
                    "collateral_drift_z": collateral,
                    "control_feasible": bool(feasible),
                    "hybrid_feasible": bool(feasible),
                    "same_mode_feasible": bool(same_mode_feasible),
                    "logit": float(cf_logits[mode_idx]),
                    "base_logit_without_cf_delta": float(cf_base_logits[mode_idx]),
                    "logit_without_hybrid_policy_delta": float(logits_without_hybrid_delta[mode_idx]),
                    "logit_without_hybrid_and_cf_delta": float(logits_without_hybrid_and_cf_delta[mode_idx]),
                    "counterfactual_scorer_delta": float(cf_delta[mode_idx]),
                    "counterfactual_scorer_alpha": float(cf_delta_alpha),
                    "counterfactual_policy_logit": float(cf_policy_logits[mode_idx]),
                    "counterfactual_policy_gate": float(cf_policy_gate),
                    "hybrid_policy_delta": float(hybrid_delta[mode_idx]),
                    "hybrid_policy_raw_delta": float(hybrid_raw_delta[mode_idx]),
                    "hybrid_policy_alpha": float(hybrid_alpha),
                    "counterfactual_eval_selector_logit": float(eval_selector_logits[mode_idx]),
                    "counterfactual_eval_selector_cost": float(eval_selector_cost[mode_idx]),
                    "counterfactual_eval_selector_feasible": float(eval_selector_feasible[mode_idx]),
                    "pair_target_effect_proxy": float(pair_target_effect_proxy[mode_idx]),
                    "pair_target_score_proxy": float(pair_target_score_proxy[mode_idx]),
                    "pair_collateral_proxy": float(pair_collateral_proxy[mode_idx]),
                    "pair_mode_switch": float(pair_mode_switch[mode_idx]),
                    "hybrid_cost": float(hybrid_cost),
                }
            )
            bucket = mode_values[mode_idx]
            bucket["ade_m"].append(float(cf_ade[mode_idx]))
            bucket["relative_ade_m"].append(relative_ade)
            bucket["route_alignment_loss"].append(float(cf_route_alignment[mode_idx]))
            bucket["relative_route_alignment_loss"].append(float(cf_route_alignment[mode_idx] - best_route_alignment))
            bucket["route_feasible"].append(float(route_feasible))
            bucket["instruction_score"].append(instruction_score)
            bucket["target_token_score"].append(float(cf_target_score))
            bucket["target_effect_z"].append(target_effect)
            bucket["collateral_drift_z"].append(collateral)
            bucket["control_feasible"].append(float(feasible))
            bucket["hybrid_feasible"].append(float(feasible))
            bucket["same_mode_feasible"].append(float(same_mode_feasible))
            bucket["logit"].append(float(cf_logits[mode_idx]))
            bucket["base_logit_without_cf_delta"].append(float(cf_base_logits[mode_idx]))
            bucket["logit_without_hybrid_policy_delta"].append(float(logits_without_hybrid_delta[mode_idx]))
            bucket["logit_without_hybrid_and_cf_delta"].append(float(logits_without_hybrid_and_cf_delta[mode_idx]))
            bucket["counterfactual_scorer_delta"].append(float(cf_delta[mode_idx]))
            bucket["counterfactual_scorer_alpha"].append(float(cf_delta_alpha))
            bucket["counterfactual_policy_logit"].append(float(cf_policy_logits[mode_idx]))
            bucket["counterfactual_policy_gate"].append(float(cf_policy_gate))
            bucket["hybrid_policy_delta"].append(float(hybrid_delta[mode_idx]))
            bucket["hybrid_policy_raw_delta"].append(float(hybrid_raw_delta[mode_idx]))
            bucket["hybrid_policy_alpha"].append(float(hybrid_alpha))
            bucket["counterfactual_eval_selector_logit"].append(float(eval_selector_logits[mode_idx]))
            bucket["counterfactual_eval_selector_cost"].append(float(eval_selector_cost[mode_idx]))
            bucket["counterfactual_eval_selector_feasible"].append(float(eval_selector_feasible[mode_idx]))
            bucket["pair_target_effect_proxy"].append(float(pair_target_effect_proxy[mode_idx]))
            bucket["pair_target_score_proxy"].append(float(pair_target_score_proxy[mode_idx]))
            bucket["pair_collateral_proxy"].append(float(pair_collateral_proxy[mode_idx]))
            bucket["pair_mode_switch"].append(float(pair_mode_switch[mode_idx]))
            bucket["hybrid_cost"].append(float(hybrid_cost))

        cost_arr = np.asarray(hybrid_costs, dtype=np.float64)
        feasible_arr = np.asarray(hybrid_feasible, dtype=bool)
        route_feasible_arr = np.asarray(route_feasible_flags, dtype=bool)
        if str(hybrid_priority) == "feasible_first" and bool(feasible_arr.any()):
            hybrid_idx = int(np.argmin(np.where(feasible_arr, cost_arr, np.inf)))
        else:
            hybrid_idx = int(np.argmin(cost_arr))
        for row in rows[-k:]:
            if int(row["mode"]) == hybrid_idx:
                row["hybrid_best"] = True
        selected_indices.append(selected_idx)
        oracle_indices.append(oracle_idx)
        hybrid_indices.append(hybrid_idx)
        route_idx = int(np.argmin(cf_route_alignment))
        route_control_idx = int(
            np.argmin(np.where(route_feasible_arr & feasible_arr, cost_arr, np.inf))
            if bool((route_feasible_arr & feasible_arr).any())
            else route_idx
        )
        if np.isfinite(eval_selector_logits).any():
            eval_selector_idx = int(np.nanargmax(eval_selector_logits))
        elif np.isfinite(eval_selector_cost).any():
            eval_selector_idx = int(np.nanargmin(eval_selector_cost))
        else:
            eval_selector_idx = -1
        selected_logit = float(cf_logits_np[selected_idx])
        pair_gap = {
            "pair_index": int(pair_idx),
            "selected_idx": int(selected_idx),
            "hybrid_idx": int(hybrid_idx),
            "route_control_idx": int(route_control_idx),
            "eval_selector_idx": int(eval_selector_idx),
            "hybrid_minus_selected_final_logit": float(cf_logits_np[hybrid_idx] - selected_logit),
            "route_control_minus_selected_final_logit": float(cf_logits_np[route_control_idx] - selected_logit),
            "hybrid_minus_selected_base_logit": float(cf_base_logits[hybrid_idx] - cf_base_logits[selected_idx]),
            "route_control_minus_selected_base_logit": float(
                cf_base_logits[route_control_idx] - cf_base_logits[selected_idx]
            ),
            "hybrid_minus_selected_scorer_delta": float(cf_delta[hybrid_idx] - cf_delta[selected_idx]),
            "route_control_minus_selected_scorer_delta": float(cf_delta[route_control_idx] - cf_delta[selected_idx]),
            "hybrid_minus_selected_hybrid_delta": float(hybrid_delta[hybrid_idx] - hybrid_delta[selected_idx]),
            "route_control_minus_selected_hybrid_delta": float(
                hybrid_delta[route_control_idx] - hybrid_delta[selected_idx]
            ),
            "hybrid_minus_selected_policy_logit": float(cf_policy_logits[hybrid_idx] - cf_policy_logits[selected_idx]),
            "route_control_minus_selected_policy_logit": float(
                cf_policy_logits[route_control_idx] - cf_policy_logits[selected_idx]
            ),
        }
        if eval_selector_idx >= 0:
            pair_gap.update(
                {
                    "eval_selector_minus_selected_final_logit": float(
                        cf_logits_np[eval_selector_idx] - selected_logit
                    ),
                    "eval_selector_minus_selected_eval_logit": float(
                        eval_selector_logits[eval_selector_idx] - eval_selector_logits[selected_idx]
                    ),
                    "eval_selector_minus_selected_eval_cost": float(
                        eval_selector_cost[eval_selector_idx] - eval_selector_cost[selected_idx]
                    ),
                    "eval_selector_selected_feasible": float(eval_selector_feasible[eval_selector_idx]),
                }
            )
        pair_selection_gaps.append(pair_gap)
        for row in rows[-k:]:
            mode_idx = int(row["mode"])
            row["route_best"] = mode_idx == route_idx
            row["route_control_best"] = mode_idx == route_control_idx

    selected_arr = np.asarray(selected_indices, dtype=np.int64)
    oracle_arr = np.asarray(oracle_indices, dtype=np.int64)
    hybrid_arr = np.asarray(hybrid_indices, dtype=np.int64)
    selected_rows = [row for row in rows if bool(row.get("selected"))]
    oracle_rows = [row for row in rows if bool(row.get("oracle_ade"))]
    hybrid_rows = [row for row in rows if bool(row.get("hybrid_best"))]
    route_rows = [row for row in rows if bool(row.get("route_best"))]
    route_control_rows = [row for row in rows if bool(row.get("route_control_best"))]
    mode_summary = {}
    for mode_idx in range(num_candidates):
        values = mode_values[mode_idx]
        mode_summary[str(mode_idx)] = {
            key: _safe_mean(vals)
            for key, vals in values.items()
        }
        mode_summary[str(mode_idx)]["num_rows"] = int(len(values.get("ade_m", [])))
    gap_summary: Dict[str, float] = {}
    for key in (
        "hybrid_minus_selected_final_logit",
        "route_control_minus_selected_final_logit",
        "eval_selector_minus_selected_final_logit",
        "hybrid_minus_selected_base_logit",
        "route_control_minus_selected_base_logit",
        "hybrid_minus_selected_scorer_delta",
        "route_control_minus_selected_scorer_delta",
        "hybrid_minus_selected_hybrid_delta",
        "route_control_minus_selected_hybrid_delta",
        "hybrid_minus_selected_policy_logit",
        "route_control_minus_selected_policy_logit",
        "eval_selector_minus_selected_eval_logit",
        "eval_selector_minus_selected_eval_cost",
        "eval_selector_selected_feasible",
    ):
        gap_summary[key] = _safe_mean([float(item.get(key, float("nan"))) for item in pair_selection_gaps])

    return {
        "num_pairs": int(len(selected_indices)),
        "num_rows": int(len(rows)),
        "num_candidates": int(num_candidates),
        "selected_matches_oracle_ade": float(np.mean(selected_arr == oracle_arr)) if selected_arr.size else float("nan"),
        "selected_matches_hybrid_best": float(np.mean(selected_arr == hybrid_arr)) if selected_arr.size else float("nan"),
        "hybrid_matches_oracle_ade": float(np.mean(hybrid_arr == oracle_arr)) if hybrid_arr.size else float("nan"),
        "selected_index_counts": _index_counts(selected_indices, num_candidates),
        "oracle_ade_index_counts": _index_counts(oracle_indices, num_candidates),
        "hybrid_index_counts": _index_counts(hybrid_indices, num_candidates),
        "route_feasible_rate": _mean_from_rows(rows, "route_feasible"),
        "control_feasible_rate": _mean_from_rows(rows, "control_feasible"),
        "same_mode_feasible_rate": _mean_from_rows(rows, "same_mode_feasible"),
        "selected_route_feasible_rate": _mean_from_rows(selected_rows, "route_feasible"),
        "selected_control_feasible_rate": _mean_from_rows(selected_rows, "control_feasible"),
        "selected_same_mode_feasible_rate": _mean_from_rows(selected_rows, "same_mode_feasible"),
        "oracle_route_feasible_rate": _mean_from_rows(oracle_rows, "route_feasible"),
        "hybrid_route_feasible_rate": _mean_from_rows(hybrid_rows, "route_feasible"),
        "route_best_control_feasible_rate": _mean_from_rows(route_rows, "control_feasible"),
        "route_control_best_mean_ade_m": _mean_from_rows(route_control_rows, "ade_m"),
        "route_control_best_mean_route_alignment_loss": _mean_from_rows(route_control_rows, "route_alignment_loss"),
        "route_control_best_mean_target_effect_z": _mean_from_rows(route_control_rows, "target_effect_z"),
        "route_control_best_mean_collateral_drift_z": _mean_from_rows(route_control_rows, "collateral_drift_z"),
        "selection_gap_summary": gap_summary,
        "mode_summary": mode_summary,
        "rows": rows,
        "pair_selection_gaps": pair_selection_gaps,
    }


def _index_counts(indices: List[int], num_candidates: int) -> Dict[str, int]:
    counts = {str(idx): 0 for idx in range(num_candidates)}
    for idx in indices:
        counts[str(int(idx))] += 1
    return counts


def _group_rows_by_pair(rows: List[Dict]) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["pair_index"])].append(row)
    return dict(sorted(grouped.items()))


def _mean_from_rows(rows: List[Dict], key: str) -> float:
    return _safe_mean([float(row.get(key, float("nan"))) for row in rows])


def sweep_hybrid_weights(
    rows: List[Dict],
    num_candidates: int,
    ade_temperature_values: List[float],
    ade_weights: List[float],
    instruction_weights: List[float],
    target_effect_weights: List[float],
    collateral_weights: List[float],
    hybrid_priority: str = "weighted",
) -> List[Dict]:
    grouped = _group_rows_by_pair(rows)
    results: List[Dict] = []
    for ade_temperature_m in ade_temperature_values:
        for ade_weight in ade_weights:
            for instruction_weight in instruction_weights:
                for target_effect_weight in target_effect_weights:
                    for collateral_weight in collateral_weights:
                        selected_rows: List[Dict] = []
                        oracle_rows: List[Dict] = []
                        hybrid_rows: List[Dict] = []
                        selected_indices: List[int] = []
                        oracle_indices: List[int] = []
                        hybrid_indices: List[int] = []

                        for pair_rows in grouped.values():
                            if not pair_rows:
                                continue
                            selected = next((row for row in pair_rows if bool(row.get("selected"))), pair_rows[0])
                            oracle = next((row for row in pair_rows if bool(row.get("oracle_ade"))), pair_rows[0])
                            costs = []
                            feasible = []
                            for row in pair_rows:
                                relative_ade = float(row.get("relative_ade_m", 0.0))
                                target_token_score = float(row.get("target_token_score", row.get("instruction_score", 0.0)))
                                target_effect = float(row.get("target_effect_z", 0.0))
                                collateral = float(row.get("collateral_drift_z", 0.0))
                                cost = (
                                    float(ade_weight) * relative_ade / max(float(ade_temperature_m), 1e-6)
                                    + float(instruction_weight) * target_token_score
                                    - float(target_effect_weight) * target_effect
                                    + float(collateral_weight) * collateral
                                )
                                costs.append(cost)
                                feasible.append(bool(row.get("hybrid_feasible", False)))
                            cost_arr = np.asarray(costs, dtype=np.float64)
                            feasible_arr = np.asarray(feasible, dtype=bool)
                            if str(hybrid_priority) == "feasible_first" and bool(feasible_arr.any()):
                                hybrid = pair_rows[int(np.argmin(np.where(feasible_arr, cost_arr, np.inf)))]
                            else:
                                hybrid = pair_rows[int(np.argmin(cost_arr))]

                            selected_rows.append(selected)
                            oracle_rows.append(oracle)
                            hybrid_rows.append(hybrid)
                            selected_indices.append(int(selected["mode"]))
                            oracle_indices.append(int(oracle["mode"]))
                            hybrid_indices.append(int(hybrid["mode"]))

                        selected_arr = np.asarray(selected_indices, dtype=np.int64)
                        oracle_arr = np.asarray(oracle_indices, dtype=np.int64)
                        hybrid_arr = np.asarray(hybrid_indices, dtype=np.int64)
                        result = {
                            "ade_temperature_m": float(ade_temperature_m),
                            "hybrid_ade_weight": float(ade_weight),
                            "instruction_weight": float(instruction_weight),
                            "target_effect_weight": float(target_effect_weight),
                            "collateral_weight": float(collateral_weight),
                            "num_pairs": int(len(hybrid_rows)),
                            "selected_matches_hybrid_best": (
                                float(np.mean(selected_arr == hybrid_arr)) if hybrid_arr.size else float("nan")
                            ),
                            "hybrid_matches_oracle_ade": (
                                float(np.mean(hybrid_arr == oracle_arr)) if hybrid_arr.size else float("nan")
                            ),
                            "hybrid_index_counts": _index_counts(hybrid_indices, num_candidates),
                            "hybrid_mean_ade_m": _mean_from_rows(hybrid_rows, "ade_m"),
                            "hybrid_mean_relative_ade_m": _mean_from_rows(hybrid_rows, "relative_ade_m"),
                            "hybrid_mean_instruction_score": _mean_from_rows(hybrid_rows, "instruction_score"),
                            "hybrid_mean_target_token_score": _mean_from_rows(hybrid_rows, "target_token_score"),
                            "hybrid_mean_target_effect_z": _mean_from_rows(hybrid_rows, "target_effect_z"),
                            "hybrid_mean_collateral_drift_z": _mean_from_rows(hybrid_rows, "collateral_drift_z"),
                            "hybrid_feasible_rate": _mean_from_rows(hybrid_rows, "hybrid_feasible"),
                            "selected_mean_ade_m": _mean_from_rows(selected_rows, "ade_m"),
                            "selected_mean_instruction_score": _mean_from_rows(selected_rows, "instruction_score"),
                            "selected_mean_target_token_score": _mean_from_rows(selected_rows, "target_token_score"),
                            "selected_mean_target_effect_z": _mean_from_rows(selected_rows, "target_effect_z"),
                            "selected_mean_collateral_drift_z": _mean_from_rows(selected_rows, "collateral_drift_z"),
                            "selected_feasible_rate": _mean_from_rows(selected_rows, "hybrid_feasible"),
                        }
                        result["hybrid_minus_selected_ade_m"] = (
                            result["hybrid_mean_ade_m"] - result["selected_mean_ade_m"]
                        )
                        result["hybrid_minus_selected_instruction_score"] = (
                            result["hybrid_mean_instruction_score"] - result["selected_mean_instruction_score"]
                        )
                        result["hybrid_minus_selected_target_effect_z"] = (
                            result["hybrid_mean_target_effect_z"] - result["selected_mean_target_effect_z"]
                        )
                        result["hybrid_minus_selected_collateral_drift_z"] = (
                            result["hybrid_mean_collateral_drift_z"] - result["selected_mean_collateral_drift_z"]
                        )
                        results.append(result)
    results.sort(
        key=lambda item: (
            -float(item["hybrid_mean_target_effect_z"]),
            float(item["hybrid_mean_relative_ade_m"]),
            float(item["hybrid_mean_collateral_drift_z"]),
        )
    )
    return results


def _float_list(raw: List[float] | None, fallback: float) -> List[float]:
    values = raw if raw else [fallback]
    return [float(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual-aware per-mode diagnostic for candidate selection.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "counterfactual_mode_diagnostics")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--fields", nargs="+", default=["direction", "progress"])
    parser.add_argument("--ade-temperature-m", type=float, default=20.0)
    parser.add_argument("--hybrid-ade-weight", type=float, default=0.0)
    parser.add_argument("--instruction-weight", type=float, default=0.5)
    parser.add_argument("--target-effect-weight", type=float, default=1.0)
    parser.add_argument("--collateral-weight", type=float, default=0.5)
    parser.add_argument("--hybrid-collateral-mode", choices=["independent", "raw"], default="independent")
    parser.add_argument("--success-margin-z", type=float, default=0.0)
    parser.add_argument("--hybrid-collateral-margin", type=float, default=0.08)
    parser.add_argument("--hybrid-priority", choices=["weighted", "feasible_first"], default="weighted")
    parser.add_argument(
        "--route-feasible-margin",
        type=float,
        default=0.05,
        help=(
            "Per-pair route-alignment slack for generator-side diagnostics. "
            "A mode is route-feasible when its transition-alignment loss is within "
            "this margin of the best route-aligned candidate."
        ),
    )
    parser.add_argument("--sweep-ade-temperatures-m", nargs="*", type=float, default=None)
    parser.add_argument("--sweep-ade-weights", nargs="*", type=float, default=None)
    parser.add_argument("--sweep-instruction-weights", nargs="*", type=float, default=None)
    parser.add_argument("--sweep-target-effect-weights", nargs="*", type=float, default=None)
    parser.add_argument("--sweep-collateral-weights", nargs="*", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for CLI symmetry; pairs are evaluated one at a time.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--token-calibration-path", type=Path, default=None)
    parser.add_argument("--counterfactual-adapter-scale", type=float, default=0.0)
    parser.add_argument("--counterfactual-reranker-baseline", choices=["same_mode", "base_selected"], default="same_mode")
    parser.add_argument("--counterfactual-eval-selector-weight", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-instruction-weight", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-target-effect-weight", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-collateral-weight", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-success-margin", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-collateral-margin", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-disable-feasible-first", action="store_true")
    parser.add_argument(
        "--selector-noncollapse-mode-override",
        choices=["disabled", "additive"],
        default=None,
        help=(
            "Optional evaluation-only override for decoder.selector_noncollapse_mode. "
            "Use disabled when diagnosing pure reranker logits without additive "
            "target/collateral/base-preservation heuristics."
        ),
    )
    parser.add_argument(
        "--counterfactual-policy-blend-mode-override",
        choices=["blend", "override"],
        default=None,
        help=(
            "Optional evaluation-only override for decoder.counterfactual_policy_blend_mode. "
            "Use override to test whether policy logits can select control-family candidates "
            "without base-logit domination."
        ),
    )
    parser.add_argument("--transition-diagnostics-path", type=Path, default=None)
    parser.add_argument("--transition-min-success-rate", type=float, default=0.5)
    parser.add_argument("--transition-min-effect-z", type=float, default=0.0)
    parser.add_argument("--transition-min-token-match-rate", type=float, default=0.0)
    parser.add_argument("--transition-min-pairs", type=int, default=2)
    parser.add_argument("--transition-max-collateral-z", type=float, default=0.0)
    parser.add_argument(
        "--transition-allowlist",
        default="",
        help=(
            "Optional comma-separated transition keys. When set, diagnostics "
            "only evaluate pairs whose transition key is in this allowlist."
        ),
    )
    return parser.parse_args()


def _apply_counterfactual_eval_selector_overrides(model, args: argparse.Namespace) -> Dict[str, object]:
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return {}
    overrides: Dict[str, object] = {}
    float_overrides = {
        "counterfactual_eval_selector_weight": args.counterfactual_eval_selector_weight,
        "counterfactual_eval_selector_instruction_weight": args.counterfactual_eval_selector_instruction_weight,
        "counterfactual_eval_selector_target_effect_weight": args.counterfactual_eval_selector_target_effect_weight,
        "counterfactual_eval_selector_collateral_weight": args.counterfactual_eval_selector_collateral_weight,
        "counterfactual_eval_selector_success_margin": args.counterfactual_eval_selector_success_margin,
        "counterfactual_eval_selector_collateral_margin": args.counterfactual_eval_selector_collateral_margin,
    }
    for name, value in float_overrides.items():
        if value is None:
            continue
        setattr(decoder, name, float(value))
        overrides[name] = float(value)
    if bool(args.counterfactual_eval_selector_disable_feasible_first):
        decoder.counterfactual_eval_selector_feasible_first = False
        overrides["counterfactual_eval_selector_feasible_first"] = False
    if args.selector_noncollapse_mode_override is not None:
        decoder.selector_noncollapse_mode = str(args.selector_noncollapse_mode_override)
        overrides["selector_noncollapse_mode"] = str(args.selector_noncollapse_mode_override)
    if args.counterfactual_policy_blend_mode_override is not None:
        decoder.counterfactual_policy_blend_mode = str(args.counterfactual_policy_blend_mode_override)
        overrides["counterfactual_policy_blend_mode"] = str(args.counterfactual_policy_blend_mode_override)
    return overrides


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint, device, token_calibration_path=args.token_calibration_path)
    eval_selector_overrides = _apply_counterfactual_eval_selector_overrides(model, args)
    saved_args = checkpoint.get("args", {})
    attr_dim = int(saved_args.get("attr_dim", 6))
    jsonl_path = args.input or (
        REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_counterfactual_instruction_samples.filtered.jsonl"
    )
    records = read_records(jsonl_path, max_records=args.max_records)
    transition_table = load_counterfactual_transition_table(
        args.transition_diagnostics_path,
        min_success_rate=args.transition_min_success_rate,
        min_effect_z=args.transition_min_effect_z,
        min_pairs=args.transition_min_pairs,
        min_token_match_rate=args.transition_min_token_match_rate,
        max_collateral_z=args.transition_max_collateral_z,
        allow_transitions=args.transition_allowlist,
    )
    pairs, transition_filter_stats = group_counterfactual_pairs(
        records,
        fields=set(args.fields),
        max_pairs=args.max_pairs,
        transition_table=transition_table,
        pair_sampling="balanced_field",
    )
    if not pairs:
        raise ValueError(f"No counterfactual pairs found in {jsonl_path} for fields={args.fields}")
    if args.token_calibration_path:
        calibrator = GeometryTokenCalibrator.from_json(str(args.token_calibration_path))
    else:
        calibrator = build_pair_calibrator(pairs, args.target_key, model.decoder.target_len, device)

    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device) if args.qwen_planner_checkpoint else None
    metrics = evaluate(
        model,
        qwen_provider,
        pairs,
        attr_dim,
        args.target_key,
        calibrator,
        device,
        ade_temperature_m=args.ade_temperature_m,
        hybrid_ade_weight=args.hybrid_ade_weight,
        instruction_weight=args.instruction_weight,
        target_effect_weight=args.target_effect_weight,
        collateral_weight=args.collateral_weight,
        collateral_mode=args.hybrid_collateral_mode,
        success_margin_z=args.success_margin_z,
        hybrid_collateral_margin=args.hybrid_collateral_margin,
        hybrid_priority=args.hybrid_priority,
        route_feasible_margin=args.route_feasible_margin,
        counterfactual_adapter_scale=args.counterfactual_adapter_scale,
        baseline_mode=args.counterfactual_reranker_baseline,
    )
    rows = metrics.pop("rows")
    pair_selection_gaps = metrics.pop("pair_selection_gaps", [])
    sweep = sweep_hybrid_weights(
        rows,
        num_candidates=int(metrics.get("num_candidates", getattr(model.decoder, "num_candidates", 1))),
        ade_temperature_values=_float_list(args.sweep_ade_temperatures_m, args.ade_temperature_m),
        ade_weights=_float_list(args.sweep_ade_weights, args.hybrid_ade_weight),
        instruction_weights=_float_list(args.sweep_instruction_weights, args.instruction_weight),
        target_effect_weights=_float_list(args.sweep_target_effect_weights, args.target_effect_weight),
        collateral_weights=_float_list(args.sweep_collateral_weights, args.collateral_weight),
        hybrid_priority=args.hybrid_priority,
    )
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "input": str(jsonl_path),
            "target_key": args.target_key,
            "fields": list(args.fields),
            "counterfactual_adapter_scale": float(args.counterfactual_adapter_scale),
            "counterfactual_reranker_baseline": args.counterfactual_reranker_baseline,
            "counterfactual_eval_selector": {
                "weight": float(getattr(model.decoder, "counterfactual_eval_selector_weight", 0.0)),
                "instruction_weight": float(getattr(model.decoder, "counterfactual_eval_selector_instruction_weight", 0.5)),
                "target_effect_weight": float(getattr(model.decoder, "counterfactual_eval_selector_target_effect_weight", 1.0)),
                "collateral_weight": float(getattr(model.decoder, "counterfactual_eval_selector_collateral_weight", 0.5)),
                "feasible_first": bool(getattr(model.decoder, "counterfactual_eval_selector_feasible_first", True)),
                "success_margin": float(getattr(model.decoder, "counterfactual_eval_selector_success_margin", 0.0)),
                "collateral_margin": float(getattr(model.decoder, "counterfactual_eval_selector_collateral_margin", 0.08)),
                "selector_noncollapse_mode": str(getattr(model.decoder, "selector_noncollapse_mode", "unknown")),
                "counterfactual_policy_blend_mode": str(
                    getattr(model.decoder, "counterfactual_policy_blend_mode", "unknown")
                ),
                "cli_overrides": eval_selector_overrides,
            },
            "transition_filter": {
                "diagnostics_path": str(args.transition_diagnostics_path) if args.transition_diagnostics_path else None,
                "enabled": bool(transition_table),
                "min_success_rate": float(args.transition_min_success_rate),
                "min_effect_z": float(args.transition_min_effect_z),
                "min_token_match_rate": float(args.transition_min_token_match_rate),
                "min_pairs": int(args.transition_min_pairs),
                "max_collateral_z": float(args.transition_max_collateral_z),
                "allowlist": str(args.transition_allowlist),
                "num_fields": int(len(transition_table)),
                "stats": transition_filter_stats,
            },
            "qwen_planner_checkpoint": str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None,
            "hybrid_cost": {
                "ade_temperature_m": args.ade_temperature_m,
                "hybrid_ade_weight": args.hybrid_ade_weight,
                "instruction_weight": args.instruction_weight,
                "target_effect_weight": args.target_effect_weight,
                "collateral_weight": args.collateral_weight,
                "collateral_mode": args.hybrid_collateral_mode,
                "success_margin_z": args.success_margin_z,
                "hybrid_collateral_margin": args.hybrid_collateral_margin,
                "hybrid_priority": args.hybrid_priority,
                "route_feasible_margin": args.route_feasible_margin,
            },
            "hybrid_weight_sweep": sweep,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    rows_path = args.output_dir / "mode_rows.jsonl"
    gaps_path = args.output_dir / "selection_gaps.jsonl"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with gaps_path.open("w", encoding="utf-8") as f:
        for row in pair_selection_gaps:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                **metrics,
                "metrics_path": str(metrics_path),
                "rows_path": str(rows_path),
                "selection_gaps_path": str(gaps_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
