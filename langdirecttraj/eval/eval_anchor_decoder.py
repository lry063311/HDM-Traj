from __future__ import annotations

import argparse
import json
import os
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from langdirecttraj.data.instruction_dataset import (
    InstructionTrajectoryDataset,
    collate_instruction_batch,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.train.train_anchor_decoder import (
    AnchorDecoderModel,
    _local_meter_pair,
    _resample_single,
    build_qwen_provider,
    extract_qwen_semantic_bundle,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    move_batch,
    qwen_semantic_bundle_for_batch,
    road_condition_kwargs,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the minimal anchor residual decoder.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "anchor_decoder_eval")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--target-key", default="trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--plot-samples", type=int, default=6)
    parser.add_argument("--sensitivity-field", default="reference_adherence")
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--token-calibration-path", type=Path, default=None)
    parser.add_argument("--qwen-plan-cache", type=Path, default=None)
    parser.add_argument("--road-corridor-cache", type=Path, default=None)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    # Checkpoints saved on Linux may contain pathlib.PosixPath objects in
    # saved args.  Windows cannot instantiate PosixPath during unpickling, so
    # map it to WindowsPath only for the duration of torch.load.
    original_posix_path = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)
    finally:
        pathlib.PosixPath = original_posix_path


def _load_model_state_shape_safe(model: torch.nn.Module, checkpoint: Dict) -> None:
    source_state = checkpoint.get("model_state", checkpoint)
    target_state = model.state_dict()
    loadable_state = {}
    skipped = []
    for key, value in source_state.items():
        if key not in target_state:
            skipped.append({"key": key, "source": list(value.shape), "target": None})
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped.append(
                {
                    "key": key,
                    "source": list(value.shape),
                    "target": list(target_state[key].shape),
                }
            )
            continue
        loadable_state[key] = value

    missing, unexpected = model.load_state_dict(loadable_state, strict=False)
    if missing or unexpected or skipped:
        print(
            json.dumps(
                {
                    "checkpoint_load_note": (
                        "Loaded shape-compatible checkpoint tensors for calibration-compatible "
                        "evaluation; skipped tensors require current-shape initialization."
                    ),
                    "loaded_keys": len(loadable_state),
                    "skipped_shape_or_missing_keys": len(skipped),
                    "skipped_keys": skipped[:20],
                    "missing_keys": len(missing),
                    "unexpected_keys": len(unexpected),
                },
                ensure_ascii=False,
            )
        )


def _arg_value(saved_args: Dict, key: str, default):
    return saved_args.get(key, default)


def build_model(
    checkpoint: Dict,
    device: torch.device,
    token_calibration_path: str | Path | None = None,
    arg_overrides: Dict[str, object] | None = None,
) -> AnchorDecoderModel:
    raw_args = checkpoint.get("args", {})
    saved_args = dict(raw_args) if isinstance(raw_args, dict) else vars(raw_args)
    if arg_overrides:
        for key, value in arg_overrides.items():
            if value is not None:
                saved_args[key] = value
    calibration_path = token_calibration_path or _arg_value(saved_args, "token_calibration_path", None)
    model = AnchorDecoderModel(
        attr_dim=int(_arg_value(saved_args, "attr_dim", 6)),
        cond_dim=int(_arg_value(saved_args, "cond_dim", 128)),
        target_len=int(_arg_value(saved_args, "target_len", 200)),
        fusion_mode=str(_arg_value(saved_args, "fusion_mode", "gated")),
        hidden_dim=int(_arg_value(saved_args, "hidden_dim", 128)),
        tangent_scale=float(_arg_value(saved_args, "tangent_scale", 0.35)),
        normal_scale=float(_arg_value(saved_args, "normal_scale", 0.20)),
        dropout=0.0,
        decoder_type=str(_arg_value(saved_args, "decoder", checkpoint.get("model_type", "anchor_residual"))),
        dit_depth=int(_arg_value(saved_args, "dit_depth", 4)),
        dit_num_heads=int(_arg_value(saved_args, "dit_num_heads", 4)),
        dit_mlp_ratio=float(_arg_value(saved_args, "dit_mlp_ratio", 4.0)),
        num_candidates=int(_arg_value(saved_args, "num_candidates", 1)),
        qwen_plan_dim=int(_arg_value(saved_args, "qwen_plan_dim", 0)),
        qwen_plan_scale=float(_arg_value(saved_args, "qwen_plan_scale", 1.0)),
        disable_qwen_plan_conditioning=bool(
            _arg_value(saved_args, "disable_qwen_plan_conditioning", False)
        ),
        disable_structured_field_conditioning=bool(
            _arg_value(saved_args, "disable_structured_field_conditioning", False)
        ),
        semantic_control_enabled=bool(_arg_value(saved_args, "semantic_control_enabled", False)),
        field_control_fields=_arg_value(saved_args, "field_control_fields", "direction,curvature,reference_adherence"),
        field_control_latent_dim=int(_arg_value(saved_args, "field_control_latent_dim", 128)),
        field_control_num_act_classes=int(_arg_value(saved_args, "field_control_num_act_classes", 8)),
        local_edit_window_predictor=str(_arg_value(saved_args, "local_edit_window_predictor", "field_head_v1")),
        base_preserving_selector_enabled=bool(_arg_value(saved_args, "base_preserving_selector_enabled", False)),
        selector_noncollapse_mode=str(_arg_value(saved_args, "selector_noncollapse_mode", "additive")),
        pairwise_feasible_preference_enabled=bool(
            _arg_value(saved_args, "pairwise_feasible_preference_enabled", False)
        ),
        token_calibration_path=calibration_path,
        reranker_type=str(_arg_value(saved_args, "reranker_type", "concat")),
        mode_control_type=str(_arg_value(saved_args, "mode_control_type", "none")),
        mode_control_strength=float(_arg_value(saved_args, "mode_control_strength", 1.0)),
        mode_role_budget_strength=float(_arg_value(saved_args, "mode_role_budget_strength", 1.0)),
        mode_role_config_path=_arg_value(saved_args, "mode_role_config_path", None),
        mode_role_prior_logit_weight=float(_arg_value(saved_args, "mode_role_prior_logit_weight", 0.0)),
        mode_role_prior_field_weights=_arg_value(saved_args, "mode_role_prior_field_weights", ""),
        mode_role_prior_scope=str(_arg_value(saved_args, "mode_role_prior_scope", "all")),
        route_skeleton_type=str(_arg_value(saved_args, "route_skeleton_type", "none")),
        route_skeleton_points=int(_arg_value(saved_args, "route_skeleton_points", 16)),
        route_skeleton_strength=float(_arg_value(saved_args, "route_skeleton_strength", 1.0)),
        route_skeleton_tangent_scale=float(_arg_value(saved_args, "route_skeleton_tangent_scale", 0.45)),
        route_skeleton_normal_scale=float(_arg_value(saved_args, "route_skeleton_normal_scale", 0.35)),
        route_skeleton_condition_source=str(_arg_value(saved_args, "route_skeleton_condition_source", "fused")),
        route_proposal_logit_weight=float(_arg_value(saved_args, "route_proposal_logit_weight", 0.0)),
        route_control_budget_ratio=float(_arg_value(saved_args, "route_control_budget_ratio", 0.0)),
        route_control_budget_mode=str(_arg_value(saved_args, "route_control_budget_mode", "rms")),
        route_control_budget_min=float(_arg_value(saved_args, "route_control_budget_min", 0.0)),
        route_control_budget_max=float(_arg_value(saved_args, "route_control_budget_max", 0.0)),
        route_control_selection_adapter_weight=float(
            _arg_value(saved_args, "route_control_selection_adapter_weight", 0.0)
        ),
        route_control_selection_adapter_clip=float(
            _arg_value(saved_args, "route_control_selection_adapter_clip", 2.0)
        ),
        route_prototype_bank_path=_arg_value(saved_args, "route_prototype_bank_path", None),
        route_prototype_prior_strength=float(_arg_value(saved_args, "route_prototype_prior_strength", 0.0)),
        route_prototype_prior_clip=float(_arg_value(saved_args, "route_prototype_prior_clip", 1.25)),
        route_prototype_prior_mix=str(_arg_value(saved_args, "route_prototype_prior_mix", "fixed")),
        route_prototype_prior_gate=str(_arg_value(saved_args, "route_prototype_prior_gate", "route_skeleton")),
        route_token_planner_type=str(_arg_value(saved_args, "route_token_planner_type", "none")),
        route_token_plan_points=int(_arg_value(saved_args, "route_token_plan_points", 16)),
        route_token_grid_size=int(_arg_value(saved_args, "route_token_grid_size", 12)),
        route_token_grid_extent=float(_arg_value(saved_args, "route_token_grid_extent", 2.0)),
        route_token_condition_weight=float(_arg_value(saved_args, "route_token_condition_weight", 1.0)),
        route_token_anchor_weight=float(_arg_value(saved_args, "route_token_anchor_weight", 0.0)),
        route_token_condition_path=str(_arg_value(saved_args, "route_token_condition_path", "shared")),
        route_token_planner_condition_source=str(_arg_value(saved_args, "route_token_planner_condition_source", "fused")),
        direction_control_type=str(_arg_value(saved_args, "direction_control_type", "none")),
        direction_control_strength=float(_arg_value(saved_args, "direction_control_strength", 0.0)),
        direction_control_tangent_weight=float(_arg_value(saved_args, "direction_control_tangent_weight", 1.0)),
        direction_control_apply=str(_arg_value(saved_args, "direction_control_apply", "all")),
        curvature_control_type=str(_arg_value(saved_args, "curvature_control_type", "none")),
        curvature_control_strength=float(_arg_value(saved_args, "curvature_control_strength", 0.0)),
        curvature_control_apply=str(_arg_value(saved_args, "curvature_control_apply", "all")),
        curvature_control_projection=str(_arg_value(saved_args, "curvature_control_projection", "none")),
        curvature_control_projection_strength=float(
            _arg_value(saved_args, "curvature_control_projection_strength", 1.0)
        ),
        counterfactual_direction_residual_type=str(
            _arg_value(saved_args, "counterfactual_direction_residual_type", "shared")
        ),
        counterfactual_curvature_residual_type=str(
            _arg_value(saved_args, "counterfactual_curvature_residual_type", "shared")
        ),
        counterfactual_reference_residual_type=str(
            _arg_value(saved_args, "counterfactual_reference_residual_type", "shared")
        ),
        reference_control_type=str(_arg_value(saved_args, "reference_control_type", "none")),
        reference_control_strength=float(_arg_value(saved_args, "reference_control_strength", 0.0)),
        reference_control_apply=str(_arg_value(saved_args, "reference_control_apply", "all")),
        reference_positive_candidate_strength=float(
            _arg_value(saved_args, "reference_positive_candidate_strength", 0.0)
        ),
        reference_positive_candidate_scope=str(
            _arg_value(saved_args, "reference_positive_candidate_scope", "all")
        ),
        reference_distance_candidate_strength=float(
            _arg_value(saved_args, "reference_distance_candidate_strength", 0.0)
        ),
        reference_distance_candidate_scope=str(
            _arg_value(saved_args, "reference_distance_candidate_scope", "none")
        ),
        reference_distance_candidate_mode=str(
            _arg_value(saved_args, "reference_distance_candidate_mode", "global_normal")
        ),
        counterfactual_residual_strength=float(_arg_value(saved_args, "counterfactual_residual_strength", 1.0)),
        counterfactual_route_frame_budget_mode=str(
            _arg_value(saved_args, "counterfactual_route_frame_budget_mode", "none")
        ),
        counterfactual_route_frame_tangent_budget=float(
            _arg_value(saved_args, "counterfactual_route_frame_tangent_budget", 0.08)
        ),
        counterfactual_route_frame_normal_budget=float(
            _arg_value(saved_args, "counterfactual_route_frame_normal_budget", 0.08)
        ),
        counterfactual_route_frame_fields=_arg_value(
            saved_args,
            "counterfactual_route_frame_fields",
            None,
        ),
        counterfactual_local_proposal_type=str(
            _arg_value(saved_args, "counterfactual_local_proposal_type", "none")
        ),
        counterfactual_local_proposal_strength=float(
            _arg_value(saved_args, "counterfactual_local_proposal_strength", 0.0)
        ),
        language_control_residual_type=str(_arg_value(saved_args, "language_control_residual_type", "direct")),
        language_control_budget_ratio=float(_arg_value(saved_args, "language_control_budget_ratio", 0.35)),
        language_control_soft_gate_min=float(_arg_value(saved_args, "language_control_soft_gate_min", 0.25)),
        language_control_soft_gate_max=float(_arg_value(saved_args, "language_control_soft_gate_max", 1.20)),
        language_control_field_ratios=_arg_value(saved_args, "language_control_field_ratios", None),
        language_control_gate_fields=_arg_value(saved_args, "language_control_gate_fields", None),
        language_control_gate_scope=str(_arg_value(saved_args, "language_control_gate_scope", "all")),
        counterfactual_eval_selector_weight=float(_arg_value(saved_args, "counterfactual_eval_selector_weight", 0.0)),
        counterfactual_eval_selector_instruction_weight=float(
            _arg_value(saved_args, "counterfactual_eval_selector_instruction_weight", 0.5)
        ),
        counterfactual_eval_selector_target_effect_weight=float(
            _arg_value(saved_args, "counterfactual_eval_selector_target_effect_weight", 1.0)
        ),
        counterfactual_eval_selector_collateral_weight=float(
            _arg_value(saved_args, "counterfactual_eval_selector_collateral_weight", 0.5)
        ),
        counterfactual_eval_selector_mode_switch_weight=float(
            _arg_value(saved_args, "counterfactual_eval_selector_mode_switch_weight", 0.0)
        ),
        counterfactual_eval_selector_reference_mode_switch_weight=float(
            _arg_value(saved_args, "counterfactual_eval_selector_reference_mode_switch_weight", 0.0)
        ),
        counterfactual_eval_selector_reference_reduction_weight=float(
            _arg_value(saved_args, "counterfactual_eval_selector_reference_reduction_weight", 0.0)
        ),
        counterfactual_eval_selector_reference_score_weight=float(
            _arg_value(saved_args, "counterfactual_eval_selector_reference_score_weight", 0.0)
        ),
        counterfactual_eval_selector_feasible_first=bool(
            _arg_value(
                saved_args,
                "counterfactual_eval_selector_feasible_first",
                not bool(_arg_value(saved_args, "counterfactual_eval_selector_disable_feasible_first", False)),
            )
        ),
        counterfactual_eval_selector_success_margin=float(
            _arg_value(saved_args, "counterfactual_eval_selector_success_margin", 0.0)
        ),
        counterfactual_eval_selector_collateral_margin=float(
            _arg_value(saved_args, "counterfactual_eval_selector_collateral_margin", 0.08)
        ),
        counterfactual_eval_selector_collateral_mode=str(
            _arg_value(saved_args, "counterfactual_eval_selector_collateral_mode", "independent")
        ),
        counterfactual_policy_blend_mode=str(_arg_value(saved_args, "counterfactual_policy_blend_mode", "blend")),
        counterfactual_metric_policy_weight=float(_arg_value(saved_args, "counterfactual_metric_policy_weight", 0.0)),
        counterfactual_metric_policy_feasible_first=bool(
            _arg_value(saved_args, "counterfactual_metric_policy_feasible_first", True)
        ),
        counterfactual_metric_policy_success_margin=float(
            _arg_value(saved_args, "counterfactual_metric_policy_success_margin", 0.0)
        ),
        counterfactual_metric_policy_collateral_margin=float(
            _arg_value(saved_args, "counterfactual_metric_policy_collateral_margin", 0.08)
        ),
        counterfactual_metric_policy_raw_collateral_margin=float(
            _arg_value(saved_args, "counterfactual_metric_policy_raw_collateral_margin", 0.15)
        ),
        road_corridor_feature_dim=int(_arg_value(saved_args, "road_corridor_feature_dim", 8)),
        road_corridor_hidden_dim=_arg_value(saved_args, "road_corridor_hidden_dim", None),
        road_corridor_num_layers=int(_arg_value(saved_args, "road_corridor_num_layers", 2)),
        road_corridor_num_heads=_arg_value(saved_args, "road_corridor_num_heads", None),
        road_condition_weight=float(_arg_value(saved_args, "road_condition_weight", 0.0)),
        road_candidate_logit_weight=float(_arg_value(saved_args, "road_candidate_logit_weight", 0.0)),
        road_candidate_logit_mode=str(_arg_value(saved_args, "road_candidate_logit_mode", "add")),
        road_candidate_anchor_weight=float(_arg_value(saved_args, "road_candidate_anchor_weight", 0.0)),
        road_candidate_residual_scale=float(_arg_value(saved_args, "road_candidate_residual_scale", 1.0)),
        disable_road_conditioning=bool(_arg_value(saved_args, "disable_road_conditioning", False)),
        route_memory_prior_condition_weight=float(
            _arg_value(saved_args, "route_memory_prior_condition_weight", 0.0)
        ),
        route_memory_prior_condition_path=str(
            _arg_value(saved_args, "route_memory_prior_condition_path", "separate")
        ),
        route_memory_prior_points=int(_arg_value(saved_args, "route_memory_prior_points", 16)),
        route_memory_prior_num_layers=int(_arg_value(saved_args, "route_memory_prior_num_layers", 1)),
        route_memory_prior_num_heads=(
            int(_arg_value(saved_args, "route_memory_prior_num_heads", 0)) or None
        ),
        route_memory_proposal_enabled=bool(_arg_value(saved_args, "route_memory_proposal_enabled", False)),
        route_memory_proposal_points=int(_arg_value(saved_args, "route_memory_proposal_points", 16)),
        route_memory_proposal_component_dim=int(_arg_value(saved_args, "route_memory_proposal_component_dim", 4)),
        route_memory_proposal_num_layers=int(_arg_value(saved_args, "route_memory_proposal_num_layers", 1)),
        route_memory_proposal_num_heads=(
            int(_arg_value(saved_args, "route_memory_proposal_num_heads", 0)) or None
        ),
        route_memory_proposal_residual_scale=float(
            _arg_value(saved_args, "route_memory_proposal_residual_scale", 0.05)
        ),
        route_memory_proposal_base_mode=str(
            _arg_value(saved_args, "route_memory_proposal_base_mode", "mix")
        ),
        route_memory_proposal_logit_weight=float(
            _arg_value(saved_args, "route_memory_proposal_logit_weight", 0.0)
        ),
        route_memory_proposal_logit_mode=str(
            _arg_value(saved_args, "route_memory_proposal_logit_mode", "add")
        ),
    )
    _load_model_state_shape_safe(model, checkpoint)
    model.to(device)
    model.eval()
    return model


def prepare_eval_batch(batch: Dict, target_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = []
    references = []
    route_scales = []
    for i in range(batch["traj"].shape[0]):
        traj_len = int(batch["traj_mask"][i].sum().item())
        ref_len = int(batch["reference_mask"][i].sum().item())
        traj_i = batch["traj"][i, : max(traj_len, 1)]
        ref_i = batch["reference"][i, : max(ref_len, 1)]
        if ref_i.shape[0] < 2:
            ref_i = traj_i

        traj_norm, ref_norm, route_scale = _local_meter_pair(traj_i, ref_i)
        targets.append(_resample_single(traj_norm, target_len))
        references.append(_resample_single(ref_norm, target_len))
        route_scales.append(route_scale)

    return (
        torch.stack(targets, dim=0),
        torch.stack(references, dim=0),
        torch.stack(route_scales, dim=0).view(-1, 1),
    )


def point_segment_distance(points: torch.Tensor, segments_start: torch.Tensor, segments_end: torch.Tensor) -> torch.Tensor:
    points_exp = points[:, None, :]
    start = segments_start[None, :, :]
    end = segments_end[None, :, :]
    seg = end - start
    denom = (seg * seg).sum(dim=-1).clamp_min(1e-8)
    t = ((points_exp - start) * seg).sum(dim=-1) / denom
    t = t.clamp(0.0, 1.0)
    proj = start + t.unsqueeze(-1) * seg
    return torch.linalg.norm(points_exp - proj, dim=-1).min(dim=1).values


def reference_distance_m(pred: torch.Tensor, reference: torch.Tensor, route_scales: torch.Tensor) -> torch.Tensor:
    values = []
    for i in range(pred.shape[0]):
        dists = point_segment_distance(pred[i], reference[i, :-1], reference[i, 1:])
        values.append(dists.mean() * route_scales[i, 0])
    return torch.stack(values, dim=0)


def select_prediction(pred_out):
    if not isinstance(pred_out, tuple):
        return pred_out
    pred, aux = pred_out
    if pred.ndim != 4:
        return pred
    logits = aux.get("candidate_logits") if isinstance(aux, dict) else None
    if logits is None:
        index = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
    else:
        index = logits.argmax(dim=1)
    return pred[torch.arange(pred.shape[0], device=pred.device), index]


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def evaluate_model(
    model: AnchorDecoderModel,
    loader: DataLoader,
    device: torch.device,
    attr_dim: int,
    sensitivity_field: str,
    qwen_provider=None,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    ade_values: List[float] = []
    fde_values: List[float] = []
    ref_values: List[float] = []
    sensitivity_values: Dict[str, List[float]] = {}
    sensitivity_changed_values: Dict[str, List[float]] = {}
    plot_records: List[Dict] = []

    field_index = DEFAULT_MOBILITY_VOCAB.fields.index(sensitivity_field)
    field_tokens = DEFAULT_MOBILITY_VOCAB.token_to_id[sensitivity_field]

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            target, reference, route_scales = prepare_eval_batch(batch, model.decoder.target_len)
            attr_cond = make_attr_cond(batch, attr_dim, device)
            semantic_plan_tokens = qwen_semantic_bundle_for_batch(qwen_provider, batch, device)
            pred = select_prediction(
                model(
                    reference,
                    batch["mobility_tokens"],
                    attr_cond,
                    semantic_plan_tokens=semantic_plan_tokens,
                    **road_condition_kwargs(batch),
                    return_aux=True,
                )
            )

            point_error_m = torch.linalg.norm(pred - target, dim=-1) * route_scales
            ade = point_error_m.mean(dim=1)
            fde = point_error_m[:, -1]
            ref_dist = reference_distance_m(pred, reference, route_scales)

            ade_values.extend(ade.detach().cpu().tolist())
            fde_values.extend(fde.detach().cpu().tolist())
            ref_values.extend(ref_dist.detach().cpu().tolist())

            base_pred = pred
            for token_name, token_id in field_tokens.items():
                counter_tokens = batch["mobility_tokens"].clone()
                counter_tokens[:, field_index] = int(token_id)
                counter_pred = select_prediction(
                    model(
                        reference,
                        counter_tokens,
                        attr_cond,
                        semantic_plan_tokens=semantic_plan_tokens,
                        **road_condition_kwargs(batch),
                        return_aux=True,
                    )
                )
                delta = torch.linalg.norm(counter_pred - base_pred, dim=-1).mean(dim=1) * route_scales[:, 0]
                sensitivity_values.setdefault(token_name, []).extend(delta.detach().cpu().tolist())
                changed = batch["mobility_tokens"][:, field_index] != int(token_id)
                if bool(changed.any()):
                    sensitivity_changed_values.setdefault(token_name, []).extend(delta[changed].detach().cpu().tolist())

            if len(plot_records) < 32:
                remaining = 32 - len(plot_records)
                for i in range(min(remaining, pred.shape[0])):
                    plot_records.append(
                        {
                            "sample_id": int(batch["sample_id"][i].item()),
                            "instruction": batch["instruction"][i],
                            "labels": batch["labels"][i],
                            "target": target[i].detach().cpu().numpy(),
                            "reference": reference[i].detach().cpu().numpy(),
                            "prediction": pred[i].detach().cpu().numpy(),
                            "route_scale": float(route_scales[i, 0].item()),
                        }
                    )

    metrics = {
        "ade_m": summarize(ade_values),
        "fde_m": summarize(fde_values),
        "mean_reference_distance_m": summarize(ref_values),
        "sensitivity_field": sensitivity_field,
        "counterfactual_sensitivity_m": {
            token_name: summarize(values) for token_name, values in sorted(sensitivity_values.items())
        },
        "counterfactual_sensitivity_changed_only_m": {
            token_name: summarize(values) for token_name, values in sorted(sensitivity_changed_values.items())
        },
        "num_evaluated": len(ade_values),
    }
    return metrics, plot_records, []


def plot_predictions(records: List[Dict], output_path: Path, max_samples: int) -> None:
    if max_samples <= 0 or not records:
        return
    import matplotlib.pyplot as plt

    selected = records[:max_samples]
    cols = min(3, len(selected))
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for ax, record in zip(axes.ravel(), selected):
        scale = record["route_scale"]
        target = record["target"] * scale
        reference = record["reference"] * scale
        pred = record["prediction"] * scale
        ax.plot(reference[:, 0], reference[:, 1], color="#8a8a8a", linewidth=1.5, label="reference")
        ax.plot(target[:, 0], target[:, 1], color="#2474b5", linewidth=1.8, label="target")
        ax.plot(pred[:, 0], pred[:, 1], color="#d64b3c", linewidth=1.8, label="prediction")
        ax.scatter(target[0, 0], target[0, 1], color="#1b9e77", s=16)
        ax.scatter(target[-1, 0], target[-1, 1], color="#222222", s=16)
        ax.set_aspect("equal", adjustable="datalim")
        title = f"id={record['sample_id']} | {record['labels'].get('direction', '')}"
        ax.set_title(title, fontsize=9)
        ax.axis("on")
        ax.grid(True, alpha=0.25)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint, device, token_calibration_path=args.token_calibration_path)
    saved_args = checkpoint.get("args", {})
    attr_dim = int(_arg_value(saved_args, "attr_dim", 6))
    if args.qwen_plan_cache is not None:
        args.qwen_planner_checkpoint = None
    elif args.qwen_planner_checkpoint is None:
        saved_qwen_checkpoint = _arg_value(saved_args, "qwen_planner_checkpoint", None)
        if saved_qwen_checkpoint:
            args.qwen_planner_checkpoint = Path(saved_qwen_checkpoint)
    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device)

    if args.sensitivity_field not in DEFAULT_MOBILITY_VOCAB.fields:
        raise ValueError(f"Unknown sensitivity field: {args.sensitivity_field}")

    jsonl_path = args.input or (REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_instruction_samples.jsonl")
    dataset = InstructionTrajectoryDataset(
        jsonl_path,
        max_samples=args.max_samples,
        target_key=args.target_key,
        qwen_plan_cache_path=args.qwen_plan_cache,
        road_corridor_cache_path=args.road_corridor_cache,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_instruction_batch,
    )

    metrics, plot_records, _ = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        attr_dim=attr_dim,
        sensitivity_field=args.sensitivity_field,
        qwen_provider=qwen_provider,
    )
    metrics.update(
        {
            "dataset": args.dataset,
            "jsonl_path": str(jsonl_path),
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "model_type": str(checkpoint.get("model_type", "unknown")),
            "target_key": args.target_key,
            "qwen_planner_checkpoint": str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None,
            "qwen_semantic_conditioning": qwen_provider is not None,
        }
    )

    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    plot_path = output_dir / "predictions.png"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_predictions(plot_records, plot_path, args.plot_samples)

    print(json.dumps({**metrics, "metrics_path": str(metrics_path), "plot_path": str(plot_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
