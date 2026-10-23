from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from langdirecttraj.counterfactual_policy import independent_collateral_field_names
from langdirecttraj.eval.eval_anchor_decoder import build_model, load_checkpoint
from langdirecttraj.eval.eval_counterfactual_controllability import (
    add_pace_metrics_to_rows,
    apply_counterfactual_eval_selector_overrides,
    build_pair_calibrator,
    collateral_drift_z_by_field,
    generate_candidate_geometries,
    geometry_values,
    group_counterfactual_pairs,
    independent_collateral_fields,
    read_records,
    tensor_record,
    tensor_dict_to_rows,
)
from langdirecttraj.eval.instruction_geometry import GeometryTokenCalibrator
from langdirecttraj.eval.route_selection import load_selection_prior, select_prediction_with_policy
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.numerics import safe_atan2
from langdirecttraj.train.control_losses import load_counterfactual_transition_table
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    parse_field_epochs,
    parse_field_float_map,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether the K-candidate counterfactual pool contains "
            "field-preserving edits before changing scorer/generator losses."
        )
    )
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "counterfactual_candidate_edit_diagnostics")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--pair-sampling", choices=["sequential", "balanced_field"], default="balanced_field")
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["distance", "direction", "curvature", "progress", "reference_adherence"],
    )
    parser.add_argument("--success-margin-z", type=float, default=0.0)
    parser.add_argument("--independent-collateral-margin-z", type=float, default=0.12)
    parser.add_argument(
        "--collateral-field-mode",
        choices=["independent", "curvature_raw", "raw"],
        default="independent",
        help=(
            "Which non-target fields define the candidate feasibility guardrail. "
            "'curvature_raw' uses raw non-target drift for curvature edits and "
            "the independent policy for other fields."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--token-calibration-path", type=Path, default=None)
    parser.add_argument("--counterfactual-adapter-scale", type=float, default=0.0)
    parser.add_argument(
        "--counterfactual-residual-strength",
        type=float,
        default=None,
        help=(
            "Evaluation-only override for decoder.counterfactual_residual_strength. "
            "Keeps candidate diagnostics aligned with Table 4 sweeps on older checkpoints."
        ),
    )
    parser.add_argument(
        "--reference-positive-candidate-strength-override",
        type=float,
        default=None,
        help="Evaluation-only override for reference-adherence positive candidate strength.",
    )
    parser.add_argument(
        "--reference-positive-candidate-scope-override",
        choices=["all", "none", "counterfactual_reference_adherence"],
        default=None,
        help="Evaluation-only override for when reference-positive candidates are injected.",
    )
    parser.add_argument("--transition-diagnostics-path", type=Path, default=None)
    parser.add_argument("--transition-min-success-rate", type=float, default=0.6)
    parser.add_argument("--transition-min-effect-z", type=float, default=0.05)
    parser.add_argument("--transition-min-token-match-rate", type=float, default=0.5)
    parser.add_argument("--transition-min-pairs", type=int, default=2)
    parser.add_argument("--transition-field-min-success-rates", default="")
    parser.add_argument("--transition-field-min-effect-z", default="")
    parser.add_argument("--transition-field-min-token-match-rates", default="")
    parser.add_argument("--transition-field-min-pairs", default="")
    parser.add_argument("--transition-max-collateral-z", type=float, default=0.0)
    parser.add_argument("--transition-field-max-collateral-z", default="")
    parser.add_argument("--transition-allowlist", default="")
    parser.add_argument(
        "--selection-policy",
        choices=[
            "reranker",
            "od_cell_prior",
            "reranker_plus_od_cell_prior",
            "reranker_guarded_od_cell_prior",
            "transition_prior",
            "reranker_guarded_transition_prior",
            "target_direction_transition_prior",
            "fixed_mode_0",
            "fixed_mode_1",
            "fixed_mode_2",
            "fixed_mode_3",
            "fixed_mode_4",
            "fixed_mode_5",
        ],
        default="reranker",
    )
    parser.add_argument("--selection-prior-path", type=Path, default=None)
    parser.add_argument("--selection-route-weight", type=float, default=1.0)
    parser.add_argument("--selection-token-guard-margin", type=float, default=0.0)
    parser.add_argument("--selection-token-absolute-threshold", type=float, default=float("inf"))
    parser.add_argument("--selection-min-route-improvement", type=float, default=0.0)
    parser.add_argument("--selection-logit-guard-margin", type=float, default=float("inf"))
    parser.add_argument("--selection-preserve-direction", action="store_true")
    parser.add_argument("--selection-preserve-target-direction", action="store_true")
    parser.add_argument("--counterfactual-eval-selector-weight", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-instruction-weight", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-target-effect-weight", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-collateral-weight", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-mode-switch-weight", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-reference-mode-switch-weight", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-reference-reduction-weight", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-reference-score-weight", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-success-margin", type=float, default=None)
    parser.add_argument("--counterfactual-eval-selector-collateral-margin", type=float, default=None)
    parser.add_argument(
        "--counterfactual-eval-selector-collateral-mode",
        choices=["independent", "curvature_raw", "raw"],
        default=None,
    )
    parser.add_argument("--counterfactual-eval-selector-disable-feasible-first", action="store_true")
    parser.add_argument(
        "--counterfactual-direction-residual-type-override",
        choices=[
            "",
            "shared",
            "transition_contract_pulse",
            "transition_arc_skeleton",
            "transition_arc_multiturn_burst",
            "transition_hybrid_skeleton_reflect",
            "transition_pivot_skeleton",
            "transition_reflect_skeleton",
            "transition_skeleton_reversal",
            "transition_skeleton_reversal_scurve",
            "transition_skeleton",
            "transition_template",
            "transition_template_balanced",
            "transition_template_contract",
            "transition_template_mean_projected",
            "transition_template_projected",
        ],
        default="",
        help=(
            "Evaluation-only override for direction counterfactual residual variants. "
            "Empty keeps the checkpoint configuration."
        ),
    )
    parser.add_argument(
        "--counterfactual-direction-residual-bank",
        nargs="+",
        choices=[
            "shared",
            "transition_contract_pulse",
            "transition_arc_skeleton",
            "transition_arc_multiturn_burst",
            "transition_hybrid_skeleton_reflect",
            "transition_pivot_skeleton",
            "transition_reflect_skeleton",
            "transition_skeleton_reversal",
            "transition_skeleton_reversal_scurve",
            "transition_skeleton",
            "transition_template",
            "transition_template_balanced",
            "transition_template_contract",
            "transition_template_mean_projected",
            "transition_template_projected",
        ],
        default=None,
        help=(
            "Eval-only direction candidate bank. For direction counterfactual pairs, "
            "generate candidates once per residual type and merge the pool before "
            "computing oracle target/raw diagnostics. Non-direction pairs keep the "
            "single residual override."
        ),
    )
    parser.add_argument(
        "--counterfactual-curvature-residual-bank",
        nargs="+",
        choices=[
            "shared",
            "local_turn_density",
            "base_preserving_local_turn_density",
            "base_selected_projected_normal_only",
            "base_selected_projected_normal_selector_compose",
            "direction_neutral_turn_density",
            "direction_preserving_curvature_burst",
            "signed_balanced_winding_basis",
            "midlocal_winding_pulse",
        ],
        default=None,
        help=(
            "Eval-only curvature candidate bank. For curvature counterfactual pairs, "
            "generate candidates once per residual type and merge the pool before "
            "computing oracle target/raw diagnostics. Non-curvature pairs keep the "
            "single residual override."
        ),
    )
    parser.add_argument(
        "--guided-refine-candidates",
        action="store_true",
        help=(
            "Eval-only SOTA-oriented candidate refinement inspired by OmniControl/GMD/UPGLD: "
            "run a few differentiable coordinate steps on generated candidates to improve the "
            "target field while penalizing collateral and off-manifold movement."
        ),
    )
    parser.add_argument("--guided-refine-steps", type=int, default=6)
    parser.add_argument("--guided-refine-lr", type=float, default=0.015)
    parser.add_argument(
        "--guided-refine-mode",
        choices=["coordinate", "route_frame_mask", "route_frame_target_mask", "route_frame_transition_mask"],
        default="coordinate",
        help=(
            "coordinate applies unconstrained coordinate guidance. route_frame_mask projects "
            "updates into a local tangent/normal frame and gates them to the trajectory middle. "
            "route_frame_target_mask further uses a target-aware local mask and projects the "
            "target step away from guardrail gradients. route_frame_transition_mask further "
            "uses transition-specific local masks for hard cases like straight->multi_turn "
            "and smooth->winding."
        ),
    )
    parser.add_argument(
        "--guided-refine-scales",
        default="0.5,1.0,2.0",
        help="Comma-separated guidance scale multipliers; each scale becomes a proposal family.",
    )
    parser.add_argument("--guided-refine-target-weight", type=float, default=1.0)
    parser.add_argument("--guided-refine-collateral-weight", type=float, default=0.75)
    parser.add_argument("--guided-refine-trust-weight", type=float, default=0.35)
    parser.add_argument("--guided-refine-reference-weight", type=float, default=0.10)
    parser.add_argument("--guided-refine-smoothness-weight", type=float, default=0.05)
    parser.add_argument("--guided-refine-endpoint-weight", type=float, default=50.0)
    parser.add_argument("--guided-refine-tangent-ratio", type=float, default=0.15)
    parser.add_argument("--guided-refine-step-clip", type=float, default=0.035)
    parser.add_argument("--guided-refine-max-delta", type=float, default=0.18)
    parser.add_argument(
        "--guided-refine-fields",
        nargs="+",
        default=["direction", "curvature", "reference_adherence"],
        help="Counterfactual fields eligible for guided refinement.",
    )
    parser.add_argument(
        "--counterfactual-curvature-residual-type-override",
        choices=[
            "",
            "shared",
            "local_turn_density",
            "base_preserving_local_turn_density",
            "base_selected_projected_normal_only",
            "base_selected_projected_normal_selector_compose",
            "direction_neutral_turn_density",
            "direction_preserving_curvature_burst",
            "signed_balanced_winding_basis",
            "midlocal_winding_pulse",
        ],
        default="",
        help=(
            "Evaluation-only override for testing curvature counterfactual residual variants "
            "on an existing checkpoint. Empty keeps the checkpoint configuration."
        ),
    )
    parser.add_argument(
        "--counterfactual-reference-residual-type-override",
        choices=["", "shared", "normal_only", "normal_mean_shift", "normal_hybrid_shift", "normal_only_projected"],
        default="",
        help=(
            "Evaluation-only override for reference-adherence counterfactual residual variants. "
            "Empty keeps the checkpoint configuration."
        ),
    )
    parser.add_argument(
        "--counterfactual-route-frame-budget-mode",
        choices=["none", "clamp", "soft_tanh"],
        default=None,
        help="Evaluation-only override for route-frame counterfactual residual budgeting.",
    )
    parser.add_argument("--counterfactual-route-frame-tangent-budget", type=float, default=None)
    parser.add_argument("--counterfactual-route-frame-normal-budget", type=float, default=None)
    parser.add_argument(
        "--counterfactual-route-frame-fields",
        nargs="+",
        default=None,
        help="Fields whose counterfactual residuals should be route-frame budgeted.",
    )
    parser.add_argument("--save-pair-details", action="store_true")
    return parser.parse_args()


def _safe_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _safe_p90(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, 90))


def _summarize(values: List[float]) -> Dict[str, float]:
    return {"mean": _safe_mean(values), "p90": _safe_p90(values)}


def _aux_candidate_tensor(aux: Dict, key: str, num_candidates: int) -> np.ndarray | None:
    value = aux.get(key) if isinstance(aux, dict) else None
    if value is None:
        return None
    if torch.is_tensor(value):
        arr = value.detach().float().cpu().numpy()
    else:
        arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return None
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim >= 1 and arr.shape[0] != num_candidates and arr.shape[0] == num_candidates:
        return arr
    if arr.ndim >= 1 and arr.shape[0] == num_candidates:
        return arr
    if arr.ndim >= 1 and arr.shape[0] % max(num_candidates, 1) == 0 and arr.shape[0] != num_candidates:
        try:
            arr = arr.reshape(num_candidates, *arr.shape[1:])
        except ValueError:
            return None
    if arr.ndim >= 1 and arr.shape[0] == num_candidates:
        return arr
    return None


def _candidate_component_summary(
    aux: Dict,
    num_candidates: int,
    key: str,
) -> List[Dict[str, float]]:
    arr = _aux_candidate_tensor(aux, key, num_candidates)
    if arr is None:
        return [{"norm": float("nan"), "endpoint_norm": float("nan")} for _ in range(num_candidates)]
    flat = arr.reshape(num_candidates, -1)
    norms = np.linalg.norm(flat, axis=1)
    if arr.ndim >= 2:
        endpoint = arr[:, -1].reshape(num_candidates, -1)
    else:
        endpoint = arr.reshape(num_candidates, -1)
    endpoint_norms = np.linalg.norm(endpoint, axis=1)
    return [
        {
            "norm": float(norms[idx]),
            "endpoint_norm": float(endpoint_norms[idx]),
        }
        for idx in range(num_candidates)
    ]


def _candidate_scalar_summary(
    aux: Dict,
    num_candidates: int,
    key: str,
) -> List[float]:
    arr = _aux_candidate_tensor(aux, key, num_candidates)
    if arr is None:
        return [float("nan") for _ in range(num_candidates)]
    flat = arr.reshape(num_candidates, -1)
    values = flat[:, 0]
    return [float(v) for v in values.tolist()]


def _build_candidate_control_diagnostics(aux: Dict, num_candidates: int) -> List[Dict[str, float]]:
    direction = _candidate_component_summary(aux, num_candidates, "direction_control_delta")
    curvature = _candidate_component_summary(aux, num_candidates, "curvature_control_delta")
    reference = _candidate_component_summary(aux, num_candidates, "reference_control_delta")
    counterfactual = _candidate_component_summary(aux, num_candidates, "counterfactual_residual_delta")
    language_raw = _candidate_component_summary(aux, num_candidates, "language_control_delta_raw")
    language_final = _candidate_component_summary(aux, num_candidates, "language_control_delta")
    raw_residual = _candidate_component_summary(aux, num_candidates, "raw_residual")
    residual = _candidate_component_summary(aux, num_candidates, "residual")
    budget_scale = _candidate_scalar_summary(aux, num_candidates, "language_control_budget_scale")
    control_ratio = _candidate_scalar_summary(aux, num_candidates, "language_control_ratio")
    raw_norm = _candidate_scalar_summary(aux, num_candidates, "language_control_raw_norm")
    final_norm = _candidate_scalar_summary(aux, num_candidates, "language_control_final_norm")
    base_norm = _candidate_scalar_summary(aux, num_candidates, "language_control_base_norm")
    rows: List[Dict[str, float]] = []
    for idx in range(num_candidates):
        rows.append(
            {
                "mode": int(idx),
                "direction_control_norm": direction[idx]["norm"],
                "direction_control_endpoint_norm": direction[idx]["endpoint_norm"],
                "curvature_control_norm": curvature[idx]["norm"],
                "curvature_control_endpoint_norm": curvature[idx]["endpoint_norm"],
                "reference_control_norm": reference[idx]["norm"],
                "reference_control_endpoint_norm": reference[idx]["endpoint_norm"],
                "counterfactual_residual_norm": counterfactual[idx]["norm"],
                "counterfactual_residual_endpoint_norm": counterfactual[idx]["endpoint_norm"],
                "language_control_raw_norm_mode": language_raw[idx]["norm"],
                "language_control_raw_endpoint_norm": language_raw[idx]["endpoint_norm"],
                "language_control_final_norm_mode": language_final[idx]["norm"],
                "language_control_final_endpoint_norm": language_final[idx]["endpoint_norm"],
                "raw_residual_norm_mode": raw_residual[idx]["norm"],
                "residual_norm_mode": residual[idx]["norm"],
                "language_control_budget_scale": budget_scale[idx],
                "language_control_ratio": control_ratio[idx],
                "language_control_raw_norm_global": raw_norm[idx],
                "language_control_final_norm_global": final_norm[idx],
                "language_control_base_norm_global": base_norm[idx],
            }
        )
    return rows


def _candidate_scores(
    calibrator: GeometryTokenCalibrator,
    original: Dict,
    counterfactual: Dict,
    base_geom: Dict[str, float],
    cf_geoms: List[Dict[str, float]],
    success_margin_z: float,
    independent_margin_z: float,
    collateral_field_mode: str,
) -> List[Dict[str, float | int | bool]]:
    cf_meta = counterfactual.get("counterfactual", {})
    field_name = str(cf_meta.get("field", "unknown"))
    source_token = str(cf_meta.get("from", original.get("labels", {}).get(field_name, "unknown")))
    target_token = str(cf_meta.get("to", counterfactual.get("labels", {}).get(field_name, "unknown")))
    base_target_score = calibrator.score_token(field_name, target_token, base_geom)
    rows: List[Dict[str, float | int | bool]] = []
    for mode_idx, cf_geom in enumerate(cf_geoms):
        cf_target_score = calibrator.score_token(field_name, target_token, cf_geom)
        target_effect = float(base_target_score - cf_target_score)
        drift_by_field = collateral_drift_z_by_field(
            calibrator,
            field_name,
            original.get("labels", {}),
            base_geom,
            cf_geom,
        )
        raw_collateral = float(np.mean(list(drift_by_field.values()))) if drift_by_field else float("nan")
        independent_by_field = independent_collateral_fields(field_name, drift_by_field)
        independent_collateral = (
            float(np.mean(list(independent_by_field.values()))) if independent_by_field else float("nan")
        )
        if str(collateral_field_mode) == "raw":
            guardrail_by_field = drift_by_field
            guardrail_source = "raw"
        elif str(collateral_field_mode) == "curvature_raw":
            if field_name == "curvature":
                guardrail_by_field = drift_by_field
                guardrail_source = "curvature_raw"
            elif independent_by_field:
                guardrail_by_field = independent_by_field
                guardrail_source = "independent"
            else:
                guardrail_by_field = drift_by_field
                guardrail_source = "raw_fallback"
        elif str(collateral_field_mode) == "independent":
            if independent_by_field:
                guardrail_by_field = independent_by_field
                guardrail_source = "independent"
            else:
                guardrail_by_field = drift_by_field
                guardrail_source = "raw_fallback"
        else:
            raise ValueError(f"Unsupported collateral_field_mode={collateral_field_mode}")
        guardrail_collateral = (
            float(np.mean(list(guardrail_by_field.values()))) if guardrail_by_field else float("nan")
        )
        (
            metric_name,
            base_to_target_delta,
            observed_delta,
            signed_metric_delta,
            signed_metric_delta_z,
        ) = calibrator.transition_metric_delta(field_name, source_token, target_token, base_geom, cf_geom)
        independent_available = bool(independent_by_field)
        guardrail_ok = bool(
            np.isfinite(guardrail_collateral) and guardrail_collateral <= float(independent_margin_z)
        )
        target_ok = bool(target_effect > float(success_margin_z))
        rows.append(
            {
                "mode": int(mode_idx),
                "target_effect_z": target_effect,
                "target_score_z": float(cf_target_score),
                "raw_collateral_z": raw_collateral,
                "independent_collateral_z": independent_collateral,
                "guardrail_collateral_z": guardrail_collateral,
                "guardrail_collateral_source": guardrail_source,
                "direction_drift_z": float(drift_by_field.get("direction", float("nan"))),
                "distance_drift_z": float(drift_by_field.get("distance", float("nan"))),
                "curvature_drift_z": float(drift_by_field.get("curvature", float("nan"))),
                "progress_drift_z": float(drift_by_field.get("progress", float("nan"))),
                "reference_adherence_drift_z": float(
                    drift_by_field.get("reference_adherence", float("nan"))
                ),
                "target_ok": target_ok,
                "independent_available": independent_available,
                "guardrail_ok": guardrail_ok,
                "feasible": bool(target_ok and guardrail_ok),
                "transition_metric": metric_name,
                "base_to_target_delta": float(base_to_target_delta),
                "observed_delta": float(observed_delta),
                "signed_metric_delta": float(signed_metric_delta),
                "signed_metric_delta_z": float(signed_metric_delta_z),
            }
        )
    return rows


def _parse_guided_scales(raw: str) -> List[float]:
    values: List[float] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if math.isfinite(value) and value > 0:
            values.append(value)
    return values or [1.0]


def _torch_positive(value: torch.Tensor | float) -> torch.Tensor:
    if torch.is_tensor(value):
        return torch.clamp(value, min=0.0)
    return torch.tensor(max(float(value), 0.0), dtype=torch.float32)


def _soft_max(values: List[torch.Tensor], temperature: float = 0.04) -> torch.Tensor:
    stacked = torch.stack(values, dim=0)
    return temperature * torch.logsumexp(stacked / max(float(temperature), 1e-6), dim=0)


def _torch_contract_direction_score(points: torch.Tensor, token: str) -> torch.Tensor:
    """Differentiable approximation of the official contract direction score."""

    coarse_points = 16
    coarse = F.interpolate(points.transpose(1, 2), size=coarse_points, mode="linear", align_corners=True).transpose(1, 2)
    diffs = coarse[:, 1:] - coarse[:, :-1]
    headings = safe_atan2(diffs[..., 1], diffs[..., 0])
    delta = headings[:, 1:] - headings[:, :-1]
    delta = safe_atan2(torch.sin(delta), torch.cos(delta))
    abs_delta = delta.abs()
    turn_threshold = math.radians(35.0)
    turn_temp = math.radians(10.0)
    soft_strong = torch.sigmoid((abs_delta - turn_threshold) / max(turn_temp, 1e-6))
    displacement = torch.linalg.norm(points[:, -1] - points[:, 0], dim=-1)
    length = torch.linalg.norm(points[:, 1:] - points[:, :-1], dim=-1).sum(dim=1).clamp_min(1e-6)
    displacement_ratio = displacement / length
    net = delta.sum(dim=1)
    abs_net = net.abs()
    total_abs = (abs_delta * soft_strong).sum(dim=1)
    turn_count = soft_strong.sum(dim=1)

    full_diffs = points[:, 1:] - points[:, :-1]
    full_headings = safe_atan2(full_diffs[..., 1], full_diffs[..., 0])
    full_delta = full_headings[:, 1:] - full_headings[:, :-1]
    full_delta = safe_atan2(torch.sin(full_delta), torch.cos(full_delta))
    signed_turn = full_delta.sum(dim=1)
    first = full_diffs[:, 0]
    first_norm = torch.linalg.norm(first, dim=-1, keepdim=True).clamp_min(1e-6)
    ex = first / first_norm
    ey = torch.stack([-ex[:, 1], ex[:, 0]], dim=-1)
    rel_end = points[:, -1] - points[:, 0]
    local_dy = (rel_end * ey).sum(dim=-1)
    threshold = math.radians(15.0)
    strong_count = (full_delta.abs() > threshold).sum(dim=1).to(dtype=points.dtype).clamp_min(1.0)
    left_ratio = torch.sigmoid((full_delta - threshold) / max(math.radians(5.0), 1e-6)).sum(dim=1) / strong_count
    right_ratio = torch.sigmoid((-full_delta - threshold) / max(math.radians(5.0), 1e-6)).sum(dim=1) / strong_count

    deg35 = math.radians(35.0)
    deg135 = math.radians(135.0)
    deg120 = math.radians(120.0)
    eps = 1e-6
    u_heading_strength = torch.minimum(abs_net / max(deg135, eps), _torch_positive(0.55 - displacement_ratio) / 0.55)
    u_loop_strength = torch.minimum(_torch_positive(0.25 - displacement_ratio) / 0.25, total_abs / max(deg120, eps))
    u_strength = torch.maximum(u_heading_strength, u_loop_strength)
    multi_strength = torch.maximum(turn_count / 3.0, total_abs / max(deg120, eps))
    left_strength = _soft_max(
        [
            _torch_positive(net) / max(deg35, eps),
            _torch_positive(signed_turn) / max(deg35, eps),
            0.35 * _torch_positive(local_dy - 0.02) / 0.12,
            left_ratio,
        ]
    )
    right_strength = _soft_max(
        [
            _torch_positive(-net) / max(deg35, eps),
            _torch_positive(-signed_turn) / max(deg35, eps),
            0.35 * _torch_positive(-local_dy - 0.02) / 0.12,
            right_ratio,
        ]
    )
    left_clear = left_strength.clamp(0.0, 1.0)
    right_clear = right_strength.clamp(0.0, 1.0)
    directional_strength = torch.maximum(left_strength, right_strength)
    scores = {
        "straight": (
            abs_net / deg35
            + 0.4 * total_abs / max(deg120, eps)
            + _torch_positive(turn_count - 1.0)
            + 0.4 * torch.maximum(left_strength, right_strength)
        ),
        "left_turn": (
            _torch_positive(1.0 - left_strength)
            + 0.5 * _torch_positive(multi_strength - 1.0) * (1.0 - 0.75 * left_clear)
            + 0.5 * _torch_positive(u_strength - 1.0)
            - 0.08 * left_strength
        ),
        "right_turn": (
            _torch_positive(1.0 - right_strength)
            + 0.5 * _torch_positive(multi_strength - 1.0) * (1.0 - 0.75 * right_clear)
            + 0.5 * _torch_positive(u_strength - 1.0)
            - 0.08 * right_strength
        ),
        "u_turn": (
            _torch_positive(1.0 - u_strength)
            + 0.35 * _torch_positive(multi_strength - 1.0)
            - 0.08 * u_strength
        ),
        "multi_turn": (
            _torch_positive(1.0 - multi_strength)
            + 0.35 * _torch_positive(u_strength - 1.0)
            + 0.25 * multi_strength * directional_strength.clamp(max=1.0)
            + 0.75 * _torch_positive(directional_strength - 1.0)
            - 0.08 * multi_strength
        ),
    }
    return scores.get(str(token), scores["straight"])


def _torch_field_score(
    calibrator: GeometryTokenCalibrator,
    field_name: str,
    token: str,
    points: torch.Tensor,
    reference: torch.Tensor,
    route_scales: torch.Tensor,
    duration_min: torch.Tensor,
) -> torch.Tensor:
    if field_name == "direction":
        return _torch_contract_direction_score(points, token)
    geom = geometry_values(points, reference.expand(points.shape[0], -1, -1), route_scales.expand(points.shape[0]))
    stat = calibrator.stats[field_name][token]
    if stat.metric == "avg_speed_kmh":
        length_km = geom["length_m"].clamp_min(0.0) / 1000.0
        value = torch.clamp(length_km / (duration_min.expand(points.shape[0]).clamp_min(1e-6) / 60.0), 0.0, 120.0)
    else:
        value = geom.get(stat.metric)
        if value is None:
            value = points.new_zeros(points.shape[0])
    return (value - float(stat.center)).abs() / max(float(stat.scale), 1e-6)


def _guided_refine_loss(
    refined: torch.Tensor,
    seed: torch.Tensor,
    base_geom: Dict[str, float],
    original: Dict,
    counterfactual: Dict,
    calibrator: GeometryTokenCalibrator,
    reference: torch.Tensor,
    route_scales: torch.Tensor,
    duration_min: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss, stats, _ = _guided_refine_terms(
        refined,
        seed,
        base_geom,
        original,
        counterfactual,
        calibrator,
        reference,
        route_scales,
        duration_min,
        args,
    )
    return loss, stats


def _guided_refine_terms(
    refined: torch.Tensor,
    seed: torch.Tensor,
    base_geom: Dict[str, float],
    original: Dict,
    counterfactual: Dict,
    calibrator: GeometryTokenCalibrator,
    reference: torch.Tensor,
    route_scales: torch.Tensor,
    duration_min: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    cf_meta = counterfactual.get("counterfactual", {})
    field_name = str(cf_meta.get("field", "unknown"))
    target_token = str(cf_meta.get("to", counterfactual.get("labels", {}).get(field_name, "unknown")))
    target_score = _torch_field_score(calibrator, field_name, target_token, refined, reference, route_scales, duration_min)
    target_loss = target_score.mean()

    collateral_terms: List[torch.Tensor] = []
    if str(args.collateral_field_mode) == "raw":
        collateral_fields = [field for field in DEFAULT_MOBILITY_VOCAB.fields if field != field_name]
    elif str(args.collateral_field_mode) == "curvature_raw" and field_name == "curvature":
        collateral_fields = [field for field in DEFAULT_MOBILITY_VOCAB.fields if field != field_name]
    else:
        collateral_fields = independent_collateral_field_names(field_name, fields=DEFAULT_MOBILITY_VOCAB.fields)
    for collateral_field in collateral_fields:
        token = str(original.get("labels", {}).get(collateral_field, "unknown"))
        if token not in DEFAULT_MOBILITY_VOCAB.token_to_id.get(collateral_field, {}):
            continue
        score = _torch_field_score(calibrator, collateral_field, token, refined, reference, route_scales, duration_min)
        base_score = float(calibrator.score_token(collateral_field, token, base_geom))
        collateral_terms.append((score - base_score).abs())
    collateral = torch.stack(collateral_terms, dim=0).mean() if collateral_terms else refined.new_tensor(0.0)

    trust = (refined - seed).pow(2).mean()
    reference_loss = torch.linalg.norm(refined - reference.expand_as(refined), dim=-1).mean()
    if refined.shape[1] > 2:
        accel = refined[:, 2:] - 2.0 * refined[:, 1:-1] + refined[:, :-2]
        smooth = accel.pow(2).sum(dim=-1).mean()
    else:
        smooth = refined.new_tensor(0.0)
    endpoint = (
        (refined[:, 0] - seed[:, 0]).pow(2).sum(dim=-1)
        + (refined[:, -1] - seed[:, -1]).pow(2).sum(dim=-1)
    ).mean()
    loss = (
        float(args.guided_refine_target_weight) * target_loss
        + float(args.guided_refine_collateral_weight) * collateral
        + float(args.guided_refine_trust_weight) * trust
        + float(args.guided_refine_reference_weight) * reference_loss
        + float(args.guided_refine_smoothness_weight) * smooth
        + float(args.guided_refine_endpoint_weight) * endpoint
    )
    stats = {
        "target_loss": float(target_loss.detach().item()),
        "collateral": float(collateral.detach().item()),
        "trust": float(trust.detach().item()),
        "reference": float(reference_loss.detach().item()),
        "smooth": float(smooth.detach().item()),
        "endpoint": float(endpoint.detach().item()),
    }
    terms = {
        "target_loss": target_loss,
        "collateral": collateral,
        "trust": trust,
        "reference": reference_loss,
        "smooth": smooth,
        "endpoint": endpoint,
    }
    return loss, stats, terms


def _record_duration_min(record: Dict, candidates: torch.Tensor) -> torch.Tensor:
    attrs = record.get("trip_attributes", {})
    if not isinstance(attrs, dict):
        attrs = {}
    return torch.tensor(
        [float(attrs.get("trip_duration_min", 0.0) or 0.0)],
        device=candidates.device,
        dtype=candidates.dtype,
    ).clamp_min(1e-6)


def _candidate_geoms_from_tensor(candidates: torch.Tensor, reference: torch.Tensor, route_scales: torch.Tensor, record: Dict) -> List[Dict[str, float]]:
    geoms: List[Dict[str, float]] = []
    attrs = record.get("trip_attributes", {})
    trip_attrs = [attrs if isinstance(attrs, dict) else {}]
    for mode_idx in range(candidates.shape[1]):
        mode_pred = candidates[:, mode_idx]
        geom = tensor_dict_to_rows(geometry_values(mode_pred.float(), reference.float(), route_scales.float()))[0]
        geom = add_pace_metrics_to_rows([geom], trip_attrs)[0]
        geoms.append(geom)
    return geoms


def _route_frame_basis(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"Expected [K, T, 2] points, got {tuple(points.shape)}")
    diffs = torch.zeros_like(points)
    if points.shape[1] == 1:
        diffs[..., 0] = 1.0
    elif points.shape[1] == 2:
        segment = points[:, 1:] - points[:, :-1]
        diffs[:, 0] = segment[:, 0]
        diffs[:, 1] = segment[:, 0]
    else:
        diffs[:, 0] = points[:, 1] - points[:, 0]
        diffs[:, -1] = points[:, -1] - points[:, -2]
        diffs[:, 1:-1] = points[:, 2:] - points[:, :-2]

    endpoint_vec = points[:, -1] - points[:, 0]
    endpoint_norm = torch.linalg.norm(endpoint_vec, dim=-1, keepdim=True)
    fallback = endpoint_vec / endpoint_norm.clamp_min(1e-8)
    fallback = torch.where(
        endpoint_norm > 1e-8,
        fallback,
        torch.tensor([1.0, 0.0], device=points.device, dtype=points.dtype).view(1, 2).expand(points.shape[0], 2),
    )
    tangent_norm = torch.linalg.norm(diffs, dim=-1, keepdim=True)
    tangent = torch.where(tangent_norm > 1e-8, diffs / tangent_norm.clamp_min(1e-8), fallback[:, None, :])
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
    return tangent, normal


def _middle_refine_mask(points: torch.Tensor) -> torch.Tensor:
    if points.ndim != 3:
        raise ValueError(f"Expected [K, T, C] points, got {tuple(points.shape)}")
    time = torch.linspace(0.0, 1.0, points.shape[1], device=points.device, dtype=points.dtype)
    mask = torch.sin(time * math.pi).pow(2)
    if mask.numel() > 0:
        mask[0] = 0.0
        mask[-1] = 0.0
    return mask.view(1, points.shape[1], 1)


def _gaussian_time_mask(points: torch.Tensor, centers: List[float], sigma: float) -> torch.Tensor:
    time = torch.linspace(0.0, 1.0, points.shape[1], device=points.device, dtype=points.dtype).view(1, points.shape[1], 1)
    sigma = max(float(sigma), 1e-3)
    values = [
        torch.exp(-0.5 * ((time - float(center)) / sigma).pow(2))
        for center in centers
    ]
    mask = torch.stack(values, dim=0).amax(dim=0)
    mask[:, 0] = 0.0
    mask[:, -1] = 0.0
    return mask


def _time_window_mask(
    points: torch.Tensor,
    start: float,
    end: float,
    softness: float = 0.03,
) -> torch.Tensor:
    time = torch.linspace(0.0, 1.0, points.shape[1], device=points.device, dtype=points.dtype).view(1, points.shape[1], 1)
    softness = max(float(softness), 1e-3)
    left = torch.sigmoid((time - float(start)) / softness)
    right = torch.sigmoid((float(end) - time) / softness)
    mask = left * right
    mask[:, 0] = 0.0
    mask[:, -1] = 0.0
    return mask


def _local_turn_energy(points: torch.Tensor) -> torch.Tensor:
    energy = points.new_zeros(points.shape[0], points.shape[1], 1)
    if points.shape[1] < 3:
        return energy
    seg = points[:, 1:] - points[:, :-1]
    heading = safe_atan2(seg[..., 1], seg[..., 0])
    turn = heading[:, 1:] - heading[:, :-1]
    turn = safe_atan2(torch.sin(turn), torch.cos(turn)).abs()
    energy[:, 1:-1, 0] = turn
    return energy


def _normalize_mask_energy(energy: torch.Tensor) -> torch.Tensor:
    flat = energy.amax(dim=1, keepdim=True).clamp_min(1e-6)
    return (energy / flat).clamp(0.0, 1.0)


def _merge_masks(*masks: torch.Tensor) -> torch.Tensor:
    merged = masks[0]
    for mask in masks[1:]:
        merged = torch.maximum(merged, mask)
    return merged


def _target_refine_mask(points: torch.Tensor, field_name: str, target_token: str) -> torch.Tensor:
    base_mask = _middle_refine_mask(points).expand(points.shape[0], -1, -1)
    turn_energy = _normalize_mask_energy(_local_turn_energy(points))
    if field_name == "direction":
        if target_token == "multi_turn":
            focus_mask = _gaussian_time_mask(points, centers=[0.32, 0.5, 0.68], sigma=0.11)
            mask = torch.maximum(base_mask, focus_mask.expand(points.shape[0], -1, -1))
        elif target_token in {"left_turn", "right_turn", "u_turn"}:
            focus_mask = _gaussian_time_mask(points, centers=[0.5], sigma=0.16)
            mask = torch.maximum(base_mask, focus_mask.expand(points.shape[0], -1, -1))
        else:
            mask = base_mask
        return mask * (0.30 + 0.70 * turn_energy)
    if field_name == "curvature":
        if target_token == "winding":
            focus_mask = _gaussian_time_mask(points, centers=[0.28, 0.5, 0.72], sigma=0.10)
            mask = torch.maximum(base_mask, focus_mask.expand(points.shape[0], -1, -1))
        else:
            mask = base_mask
        return mask * (0.35 + 0.65 * turn_energy)
    if field_name == "reference_adherence":
        focus_mask = _gaussian_time_mask(points, centers=[0.2, 0.5, 0.8], sigma=0.14)
        return torch.maximum(base_mask, focus_mask.expand(points.shape[0], -1, -1))
    return base_mask


def _transition_refine_mask(
    points: torch.Tensor,
    field_name: str,
    source_token: str,
    target_token: str,
) -> torch.Tensor:
    base_mask = _target_refine_mask(points, field_name, target_token)
    turn_energy = _normalize_mask_energy(_local_turn_energy(points))
    create_energy = 1.0 - turn_energy

    if field_name == "direction" and target_token == "multi_turn":
        if source_token == "straight":
            burst = _gaussian_time_mask(points, centers=[0.20, 0.36, 0.52, 0.68, 0.84], sigma=0.055)
            span = _time_window_mask(points, 0.12, 0.92, softness=0.035)
            mask = _merge_masks(base_mask, burst.expand(points.shape[0], -1, -1), span.expand(points.shape[0], -1, -1))
            modulation = 0.55 + 0.25 * burst + 0.20 * create_energy
            return mask * modulation
        if source_token in {"left_turn", "right_turn", "u_turn"}:
            burst = _gaussian_time_mask(points, centers=[0.48, 0.64, 0.80], sigma=0.07)
            span = _time_window_mask(points, 0.38, 0.94, softness=0.04)
            mask = _merge_masks(base_mask, burst.expand(points.shape[0], -1, -1), span.expand(points.shape[0], -1, -1))
            modulation = 0.45 + 0.30 * burst + 0.15 * turn_energy + 0.10 * create_energy
            return mask * modulation
        burst = _gaussian_time_mask(points, centers=[0.28, 0.46, 0.64, 0.82], sigma=0.065)
        mask = _merge_masks(base_mask, burst.expand(points.shape[0], -1, -1))
        modulation = 0.50 + 0.30 * burst + 0.20 * create_energy
        return mask * modulation

    if field_name == "curvature" and target_token == "winding":
        if source_token == "smooth":
            burst = _gaussian_time_mask(points, centers=[0.18, 0.30, 0.42, 0.54, 0.66, 0.78, 0.90], sigma=0.045)
            span = _time_window_mask(points, 0.10, 0.94, softness=0.03)
            mask = _merge_masks(base_mask, burst.expand(points.shape[0], -1, -1), span.expand(points.shape[0], -1, -1))
            modulation = 0.58 + 0.22 * burst + 0.20 * create_energy
            return mask * modulation
        burst = _gaussian_time_mask(points, centers=[0.24, 0.40, 0.56, 0.72, 0.88], sigma=0.05)
        mask = _merge_masks(base_mask, burst.expand(points.shape[0], -1, -1))
        modulation = 0.52 + 0.28 * burst + 0.20 * create_energy
        return mask * modulation

    return base_mask


def _project_step_away_from_guard(step: torch.Tensor, guard_grad: torch.Tensor) -> torch.Tensor:
    step_flat = step.reshape(step.shape[0], -1)
    guard_flat = guard_grad.reshape(guard_grad.shape[0], -1)
    guard_norm_sq = guard_flat.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8)
    aligned = (step_flat * guard_flat).sum(dim=1, keepdim=True)
    positive = aligned > 0
    projected = step_flat - positive.to(dtype=step.dtype) * (aligned / guard_norm_sq) * guard_flat
    return projected.view_as(step)


def _project_route_frame_update(
    update: torch.Tensor,
    tangent: torch.Tensor,
    normal: torch.Tensor,
    middle_mask: torch.Tensor,
    tangent_ratio: float,
) -> torch.Tensor:
    tangent_component = (update * tangent).sum(dim=-1, keepdim=True)
    normal_component = (update * normal).sum(dim=-1, keepdim=True)
    return middle_mask * (
        float(tangent_ratio) * tangent_component * tangent
        + normal_component * normal
    )


def _guided_refine_family_outputs(
    cf_candidates: Dict,
    original: Dict,
    counterfactual: Dict,
    base_geom: Dict[str, float],
    calibrator: GeometryTokenCalibrator,
    args: argparse.Namespace,
) -> List[Tuple[str, Dict]]:
    if not bool(args.guided_refine_candidates):
        return []
    field_name = str(counterfactual.get("counterfactual", {}).get("field", "unknown"))
    if field_name not in set(args.guided_refine_fields or []):
        return []
    if not cf_candidates.get("geoms"):
        return []
    seed = cf_candidates["candidates"].detach()
    if seed.ndim != 4 or seed.shape[0] != 1:
        return []
    reference = cf_candidates["reference"].detach()
    route_scales = cf_candidates["route_scales"].detach().view(1)
    duration_min = _record_duration_min(counterfactual, seed)
    scales = _parse_guided_scales(str(args.guided_refine_scales))
    outputs: List[Tuple[str, Dict]] = []
    for scale in scales:
        refined = seed.reshape(seed.shape[1], seed.shape[2], seed.shape[3]).clone()
        seed_flat = refined.detach().clone()
        guided_mode = str(args.guided_refine_mode)
        route_frame_mode = guided_mode in {"route_frame_mask", "route_frame_target_mask", "route_frame_transition_mask"}
        if route_frame_mode:
            tangent, normal = _route_frame_basis(seed_flat)
            if guided_mode in {"route_frame_target_mask", "route_frame_transition_mask"}:
                field_name = str(counterfactual.get("counterfactual", {}).get("field", "unknown"))
                source_token = str(
                    counterfactual.get("counterfactual", {}).get(
                        "from",
                        original.get("labels", {}).get(field_name, "unknown"),
                    )
                )
                target_token = str(
                    counterfactual.get("counterfactual", {}).get(
                        "to",
                        counterfactual.get("labels", {}).get(field_name, "unknown"),
                    )
                )
                if guided_mode == "route_frame_transition_mask":
                    middle_mask = _transition_refine_mask(seed_flat, field_name, source_token, target_token)
                else:
                    middle_mask = _target_refine_mask(seed_flat, field_name, target_token)
            else:
                middle_mask = _middle_refine_mask(seed_flat).expand(seed_flat.shape[0], -1, -1)
        else:
            tangent = normal = middle_mask = None
        for _ in range(max(int(args.guided_refine_steps), 0)):
            refined = refined.detach().requires_grad_(True)
            loss, _, terms = _guided_refine_terms(
                refined,
                seed_flat,
                base_geom,
                original,
                counterfactual,
                calibrator,
                reference,
                route_scales,
                duration_min,
                args,
            )
            if guided_mode in {"route_frame_target_mask", "route_frame_transition_mask"}:
                target_grad = torch.autograd.grad(terms["target_loss"], refined, retain_graph=True)[0]
                guard_loss = (
                    float(args.guided_refine_collateral_weight) * terms["collateral"]
                    + float(args.guided_refine_trust_weight) * terms["trust"]
                    + float(args.guided_refine_reference_weight) * terms["reference"]
                    + float(args.guided_refine_smoothness_weight) * terms["smooth"]
                    + float(args.guided_refine_endpoint_weight) * terms["endpoint"]
                )
                guard_grad = torch.autograd.grad(guard_loss, refined)[0]
                grad = None
            else:
                grad = torch.autograd.grad(loss, refined)[0]
            with torch.no_grad():
                if guided_mode in {"route_frame_target_mask", "route_frame_transition_mask"}:
                    step = _project_step_away_from_guard(
                        -float(args.guided_refine_lr) * float(scale) * target_grad,
                        guard_grad,
                    )
                else:
                    step = -float(args.guided_refine_lr) * float(scale) * grad
                if route_frame_mode:
                    step = _project_route_frame_update(
                        step,
                        tangent,
                        normal,
                        middle_mask,
                        float(args.guided_refine_tangent_ratio),
                    )
                step_norm = torch.linalg.norm(step, dim=-1, keepdim=True).clamp_min(1e-8)
                max_step = float(args.guided_refine_step_clip)
                if max_step > 0:
                    step = step * torch.clamp(max_step / step_norm, max=1.0)
                refined = refined + step
                total_delta = refined - seed_flat
                max_delta = float(args.guided_refine_max_delta)
                if route_frame_mode:
                    total_delta = total_delta * middle_mask
                if max_delta > 0:
                    delta_norm = torch.linalg.norm(total_delta, dim=-1, keepdim=True).clamp_min(1e-8)
                    total_delta = total_delta * torch.clamp(max_delta / delta_norm, max=1.0)
                    refined = seed_flat + total_delta
                refined[:, 0] = seed_flat[:, 0]
                refined[:, -1] = seed_flat[:, -1]
        refined_4d = refined.detach().unsqueeze(0)
        logits = cf_candidates["logits"].detach().clone()
        family = {
            "geoms": _candidate_geoms_from_tensor(refined_4d, reference, route_scales, counterfactual),
            "candidates": refined_4d,
            "logits": logits,
            "aux": {},
            "reference": cf_candidates["reference"],
            "reference_lonlat": cf_candidates["reference_lonlat"],
            "route_scales": cf_candidates["route_scales"],
        }
        outputs.append((f"guided_refine_s{scale:g}", family))
    return outputs


def _best_index(rows: List[Dict], key: str, maximize: bool = True, feasible_only: bool = False) -> int | None:
    candidates = [row for row in rows if (not feasible_only or bool(row.get("feasible")))]
    finite = [
        row for row in candidates
        if np.isfinite(float(row.get(key, float("nan"))))
    ]
    if not finite:
        return None
    best = max(finite, key=lambda row: float(row[key])) if maximize else min(finite, key=lambda row: float(row[key]))
    return int(best["mode"])


def _reshape_optional_candidate_aux(aux: Dict, key: str, num_candidates: int) -> np.ndarray | None:
    value = _aux_candidate_tensor(aux, key, num_candidates)
    if value is None:
        return None
    return value.reshape(num_candidates, -1)


def evaluate_candidate_pool(
    model,
    qwen_provider,
    pairs: List[Tuple[Dict, Dict]],
    attr_dim: int,
    target_key: str,
    calibrator: GeometryTokenCalibrator,
    device: torch.device,
    args: argparse.Namespace,
    selection_prior: dict | None,
) -> Dict:
    pair_rows: List[Dict] = []
    by_field = defaultdict(lambda: defaultdict(list))

    for original, counterfactual in pairs:
        cf_meta = counterfactual.get("counterfactual", {})
        field_name = str(cf_meta.get("field", "unknown"))
        target_token = str(cf_meta.get("to", counterfactual.get("labels", {}).get(field_name, "unknown")))
        if field_name not in DEFAULT_MOBILITY_VOCAB.fields or target_token not in calibrator.stats[field_name]:
            continue

        base_candidates = generate_candidate_geometries(
            model,
            qwen_provider,
            original,
            attr_dim,
            target_key,
            device,
            counterfactual_adapter_scale=0.0,
        )
        base_pred_out = (base_candidates["candidates"], {"candidate_logits": base_candidates["logits"]})
        _, base_idx, _ = select_prediction_with_policy(
            base_pred_out,
            reference=base_candidates["reference"],
            reference_lonlat=base_candidates["reference_lonlat"],
            route_scales=base_candidates["route_scales"],
            prior=selection_prior,
            policy=args.selection_policy,
            route_weight=args.selection_route_weight,
            token_guard_margin=args.selection_token_guard_margin,
            token_absolute_threshold=args.selection_token_absolute_threshold,
            min_route_improvement=args.selection_min_route_improvement,
            logit_guard_margin=args.selection_logit_guard_margin,
            preserve_direction=args.selection_preserve_direction,
            preserve_target_direction=args.selection_preserve_target_direction,
            mobility_tokens=tensor_record(original, target_key=target_key)["mobility_tokens"].to(device).view(1, -1),
            dataset=str(original.get("dataset", "porto") or "porto"),
        )
        base_mode = int(base_idx.view(-1)[0].item())
        base_geom = base_candidates["geoms"][base_mode]
        base_selected_trajectory = base_candidates["candidates"][
            torch.arange(base_candidates["candidates"].shape[0], device=device),
            base_idx.to(device=device, dtype=torch.long).view(-1),
        ]
        original_direction_residual_type = getattr(
            model.decoder,
            "counterfactual_direction_residual_type",
            None,
        )
        original_curvature_residual_type = getattr(
            model.decoder,
            "counterfactual_curvature_residual_type",
            None,
        )
        direction_bank = list(args.counterfactual_direction_residual_bank or [])
        curvature_bank = list(args.counterfactual_curvature_residual_bank or [])
        use_direction_bank = bool(direction_bank) and field_name == "direction"
        use_curvature_bank = bool(curvature_bank) and field_name == "curvature"
        family_outputs: List[Tuple[str, Dict]] = []
        family_specs: List[Tuple[str, str, str]] = []
        if use_direction_bank:
            family_specs.extend(
                [("direction", str(residual_type), str(original_curvature_residual_type or "")) for residual_type in direction_bank]
            )
        elif use_curvature_bank:
            family_specs.extend(
                [("curvature", str(original_direction_residual_type or ""), str(residual_type)) for residual_type in curvature_bank]
            )
        else:
            family_specs.append(
                (
                    "single",
                    str(original_direction_residual_type or ""),
                    str(original_curvature_residual_type or ""),
                )
            )
        try:
            for family_kind, direction_residual_type, curvature_residual_type in family_specs:
                if use_direction_bank:
                    if not hasattr(model.decoder, "set_counterfactual_direction_residual_type"):
                        raise ValueError(
                            "--counterfactual-direction-residual-bank requires a DiT decoder "
                            "with counterfactual direction residual support."
                        )
                if use_curvature_bank and not hasattr(model.decoder, "counterfactual_curvature_residual_type"):
                    raise ValueError(
                        "--counterfactual-curvature-residual-bank requires a DiT decoder "
                        "with counterfactual curvature residual support."
                    )
                if hasattr(model.decoder, "set_counterfactual_direction_residual_type"):
                    model.decoder.set_counterfactual_direction_residual_type(str(direction_residual_type))
                if hasattr(model.decoder, "counterfactual_curvature_residual_type"):
                    model.decoder.counterfactual_curvature_residual_type = str(curvature_residual_type)
                cf_family = generate_candidate_geometries(
                    model,
                    qwen_provider,
                    counterfactual,
                    attr_dim,
                    target_key,
                    device,
                    counterfactual_adapter_scale=float(args.counterfactual_adapter_scale),
                    base_selected_trajectory=base_selected_trajectory,
                    base_mobility_tokens=tensor_record(original, target_key=target_key)["mobility_tokens"].to(device).view(1, -1),
                    base_selected_index=base_idx.to(device=device, dtype=torch.long).view(-1),
                )
                family_name = (
                    f"direction::{direction_residual_type}"
                    if family_kind == "direction"
                    else (
                        f"curvature::{curvature_residual_type}"
                        if family_kind == "curvature"
                        else f"{direction_residual_type}|{curvature_residual_type}"
                    )
                )
                family_outputs.append((family_name, cf_family))
        finally:
            if use_direction_bank and original_direction_residual_type is not None:
                model.decoder.set_counterfactual_direction_residual_type(str(original_direction_residual_type))
            if use_curvature_bank and original_curvature_residual_type is not None:
                model.decoder.counterfactual_curvature_residual_type = str(original_curvature_residual_type)

        if args.guided_refine_candidates and family_outputs:
            seed_outputs = list(family_outputs)
            for _, seed_family in seed_outputs:
                family_outputs.extend(
                    _guided_refine_family_outputs(
                        seed_family,
                        original,
                        counterfactual,
                        base_geom,
                        calibrator,
                        args,
                    )
                )

        if len(family_outputs) == 1:
            cf_candidates = family_outputs[0][1]
            proposal_family_by_candidate = [family_outputs[0][0] for _ in cf_candidates["geoms"]]
            family_mode_by_candidate = list(range(len(cf_candidates["geoms"])))
        else:
            geoms: List[Dict[str, float]] = []
            logits_parts: List[torch.Tensor] = []
            candidate_parts: List[torch.Tensor] = []
            proposal_family_by_candidate: List[str] = []
            family_mode_by_candidate: List[int] = []
            family_aux: Dict[str, List[torch.Tensor]] = defaultdict(list)
            for family_name, family_candidates in family_outputs:
                num_family_candidates = len(family_candidates["geoms"])
                geoms.extend(family_candidates["geoms"])
                candidate_parts.append(family_candidates["candidates"])
                logits_parts.append(family_candidates["logits"])
                proposal_family_by_candidate.extend([family_name] * num_family_candidates)
                family_mode_by_candidate.extend(list(range(num_family_candidates)))
                aux = family_candidates.get("aux", {})
                if isinstance(aux, dict):
                    for key in (
                        "candidate_logits",
                        "counterfactual_eval_selector_cost",
                        "counterfactual_eval_selector_logits",
                        "counterfactual_eval_selector_feasible",
                        "counterfactual_pair_features",
                        "direction_control_delta",
                        "curvature_control_delta",
                        "reference_control_delta",
                        "counterfactual_residual_delta",
                        "language_control_delta_raw",
                        "language_control_delta",
                        "raw_residual",
                        "residual",
                        "language_control_budget_scale",
                        "language_control_ratio",
                        "language_control_raw_norm",
                        "language_control_final_norm",
                        "language_control_base_norm",
                    ):
                        value = aux.get(key)
                        if torch.is_tensor(value) and value.ndim >= 2:
                            family_aux[key].append(value)
            merged_aux: Dict[str, torch.Tensor] = {}
            for key, parts in family_aux.items():
                if not parts:
                    continue
                try:
                    merged_aux[key] = torch.cat(parts, dim=1)
                except RuntimeError:
                    pass
            cf_candidates = {
                "geoms": geoms,
                "candidates": torch.cat(candidate_parts, dim=1),
                "logits": torch.cat(logits_parts, dim=1),
                "aux": merged_aux,
                "reference": family_outputs[0][1]["reference"],
                "reference_lonlat": family_outputs[0][1]["reference_lonlat"],
                "route_scales": family_outputs[0][1]["route_scales"],
            }
        logits = cf_candidates["logits"].detach().float().cpu().numpy().reshape(-1)
        scorer_idx = int(np.argmax(logits)) if logits.size else 0
        eval_selector_cost = _aux_candidate_tensor(
            cf_candidates.get("aux", {}),
            "counterfactual_eval_selector_cost",
            len(cf_candidates["geoms"]),
        )
        eval_selector_logits = _aux_candidate_tensor(
            cf_candidates.get("aux", {}),
            "counterfactual_eval_selector_logits",
            len(cf_candidates["geoms"]),
        )
        eval_selector_feasible = _aux_candidate_tensor(
            cf_candidates.get("aux", {}),
            "counterfactual_eval_selector_feasible",
            len(cf_candidates["geoms"]),
        )
        eval_selector_cost = None if eval_selector_cost is None else eval_selector_cost.reshape(-1)
        eval_selector_logits = None if eval_selector_logits is None else eval_selector_logits.reshape(-1)
        eval_selector_feasible = None if eval_selector_feasible is None else eval_selector_feasible.reshape(-1)
        candidate_control_diagnostics = _build_candidate_control_diagnostics(
            cf_candidates.get("aux", {}),
            len(cf_candidates["geoms"]),
        )
        candidates = _candidate_scores(
            calibrator,
            original,
            counterfactual,
            base_geom,
            cf_candidates["geoms"],
            success_margin_z=float(args.success_margin_z),
            independent_margin_z=float(args.independent_collateral_margin_z),
            collateral_field_mode=str(args.collateral_field_mode),
        )
        for candidate_idx, candidate_row in enumerate(candidates):
            candidate_row["proposal_family"] = (
                proposal_family_by_candidate[candidate_idx]
                if 0 <= candidate_idx < len(proposal_family_by_candidate)
                else ""
            )
            candidate_row["family_mode"] = (
                int(family_mode_by_candidate[candidate_idx])
                if 0 <= candidate_idx < len(family_mode_by_candidate)
                else int(candidate_idx)
            )
        feasible = [row for row in candidates if bool(row["feasible"])]
        best_target_idx = _best_index(candidates, "target_effect_z", maximize=True)
        best_collateral_idx = _best_index(candidates, "guardrail_collateral_z", maximize=False)
        best_raw_collateral_idx = _best_index(candidates, "raw_collateral_z", maximize=False)
        best_feasible_idx = _best_index(candidates, "target_effect_z", maximize=True, feasible_only=True)
        scorer_row = candidates[scorer_idx] if 0 <= scorer_idx < len(candidates) else None
        best_target_row = candidates[best_target_idx] if best_target_idx is not None else None
        best_collateral_row = candidates[best_collateral_idx] if best_collateral_idx is not None else None
        best_raw_collateral_row = candidates[best_raw_collateral_idx] if best_raw_collateral_idx is not None else None
        best_feasible_row = candidates[best_feasible_idx] if best_feasible_idx is not None else None
        scorer_diag = (
            candidate_control_diagnostics[scorer_idx]
            if 0 <= scorer_idx < len(candidate_control_diagnostics)
            else None
        )
        best_target_diag = (
            candidate_control_diagnostics[best_target_idx]
            if best_target_idx is not None and 0 <= best_target_idx < len(candidate_control_diagnostics)
            else None
        )
        best_feasible_diag = (
            candidate_control_diagnostics[best_feasible_idx]
            if best_feasible_idx is not None and 0 <= best_feasible_idx < len(candidate_control_diagnostics)
            else None
        )

        row = {
            "sample_id": int(original.get("sample_id", -1)),
            "counterfactual_id": int(counterfactual.get("counterfactual_id", counterfactual.get("sample_id", -1))),
            "field": field_name,
            "transition": (
                f"{field_name}:{cf_meta.get('from', original.get('labels', {}).get(field_name, 'unknown'))}"
                f"->{target_token}"
            ),
            "base_mode": base_mode,
            "scorer_mode": scorer_idx,
            "best_target_mode": best_target_idx,
            "best_target_family": (
                str(best_target_row.get("proposal_family", "")) if best_target_row else ""
            ),
            "best_target_family_mode": (
                int(best_target_row.get("family_mode", -1)) if best_target_row else -1
            ),
            "best_collateral_mode": best_collateral_idx,
            "best_raw_collateral_mode": best_raw_collateral_idx,
            "best_feasible_mode": best_feasible_idx,
            "best_feasible_family": (
                str(best_feasible_row.get("proposal_family", "")) if best_feasible_row else ""
            ),
            "best_feasible_family_mode": (
                int(best_feasible_row.get("family_mode", -1)) if best_feasible_row else -1
            ),
            "has_feasible_candidate": bool(feasible),
            "num_candidates": int(len(candidates)),
            "scorer_feasible": bool(scorer_row and scorer_row["feasible"]),
            "scorer_eval_selector_feasible": (
                bool(eval_selector_feasible[scorer_idx] > 0.5)
                if eval_selector_feasible is not None and 0 <= scorer_idx < len(eval_selector_feasible)
                else None
            ),
            "scorer_eval_selector_cost": (
                float(eval_selector_cost[scorer_idx])
                if eval_selector_cost is not None and 0 <= scorer_idx < len(eval_selector_cost)
                else float("nan")
            ),
            "scorer_target_effect_z": float(scorer_row["target_effect_z"]) if scorer_row else float("nan"),
            "scorer_raw_collateral_z": float(scorer_row["raw_collateral_z"]) if scorer_row else float("nan"),
            "scorer_direction_drift_z": float(scorer_row["direction_drift_z"]) if scorer_row else float("nan"),
            "scorer_independent_collateral_z": (
                float(scorer_row["independent_collateral_z"]) if scorer_row else float("nan")
            ),
            "scorer_guardrail_collateral_z": (
                float(scorer_row["guardrail_collateral_z"]) if scorer_row else float("nan")
            ),
            "best_target_effect_z": (
                float(best_target_row["target_effect_z"]) if best_target_row else float("nan")
            ),
            "best_target_independent_collateral_z": (
                float(best_target_row["independent_collateral_z"]) if best_target_row else float("nan")
            ),
            "best_target_guardrail_collateral_z": (
                float(best_target_row["guardrail_collateral_z"]) if best_target_row else float("nan")
            ),
            "best_target_raw_collateral_z": (
                float(best_target_row["raw_collateral_z"]) if best_target_row else float("nan")
            ),
            "best_target_direction_drift_z": (
                float(best_target_row["direction_drift_z"]) if best_target_row else float("nan")
            ),
            "best_collateral_target_effect_z": (
                float(best_collateral_row["target_effect_z"]) if best_collateral_row else float("nan")
            ),
            "best_collateral_guardrail_collateral_z": (
                float(best_collateral_row["guardrail_collateral_z"]) if best_collateral_row else float("nan")
            ),
            "best_raw_collateral_target_effect_z": (
                float(best_raw_collateral_row["target_effect_z"]) if best_raw_collateral_row else float("nan")
            ),
            "best_raw_collateral_z": (
                float(best_raw_collateral_row["raw_collateral_z"]) if best_raw_collateral_row else float("nan")
            ),
            "best_raw_collateral_direction_drift_z": (
                float(best_raw_collateral_row["direction_drift_z"]) if best_raw_collateral_row else float("nan")
            ),
            "best_feasible_target_effect_z": (
                float(best_feasible_row["target_effect_z"]) if best_feasible_row else float("nan")
            ),
            "best_feasible_independent_collateral_z": (
                float(best_feasible_row["independent_collateral_z"]) if best_feasible_row else float("nan")
            ),
            "best_feasible_guardrail_collateral_z": (
                float(best_feasible_row["guardrail_collateral_z"]) if best_feasible_row else float("nan")
            ),
            "best_feasible_raw_collateral_z": (
                float(best_feasible_row["raw_collateral_z"]) if best_feasible_row else float("nan")
            ),
            "best_feasible_direction_drift_z": (
                float(best_feasible_row["direction_drift_z"]) if best_feasible_row else float("nan")
            ),
            "scorer_curvature_control_norm": (
                float(scorer_diag["curvature_control_norm"]) if scorer_diag else float("nan")
            ),
            "scorer_counterfactual_residual_norm": (
                float(scorer_diag["counterfactual_residual_norm"]) if scorer_diag else float("nan")
            ),
            "scorer_language_control_final_norm_mode": (
                float(scorer_diag["language_control_final_norm_mode"]) if scorer_diag else float("nan")
            ),
            "scorer_language_control_final_endpoint_norm": (
                float(scorer_diag["language_control_final_endpoint_norm"]) if scorer_diag else float("nan")
            ),
            "best_target_curvature_control_norm": (
                float(best_target_diag["curvature_control_norm"]) if best_target_diag else float("nan")
            ),
            "best_target_counterfactual_residual_norm": (
                float(best_target_diag["counterfactual_residual_norm"]) if best_target_diag else float("nan")
            ),
            "best_target_language_control_final_norm_mode": (
                float(best_target_diag["language_control_final_norm_mode"]) if best_target_diag else float("nan")
            ),
            "best_target_language_control_final_endpoint_norm": (
                float(best_target_diag["language_control_final_endpoint_norm"]) if best_target_diag else float("nan")
            ),
            "best_feasible_curvature_control_norm": (
                float(best_feasible_diag["curvature_control_norm"]) if best_feasible_diag else float("nan")
            ),
            "best_feasible_counterfactual_residual_norm": (
                float(best_feasible_diag["counterfactual_residual_norm"]) if best_feasible_diag else float("nan")
            ),
            "best_feasible_language_control_final_norm_mode": (
                float(best_feasible_diag["language_control_final_norm_mode"]) if best_feasible_diag else float("nan")
            ),
            "best_feasible_language_control_final_endpoint_norm": (
                float(best_feasible_diag["language_control_final_endpoint_norm"]) if best_feasible_diag else float("nan")
            ),
            "candidate_logits": [float(value) for value in logits.tolist()],
            "counterfactual_eval_selector_cost": (
                [float(value) for value in eval_selector_cost.tolist()]
                if eval_selector_cost is not None
                else None
            ),
            "counterfactual_eval_selector_logits": (
                [float(value) for value in eval_selector_logits.tolist()]
                if eval_selector_logits is not None
                else None
            ),
            "counterfactual_eval_selector_feasible": (
                [float(value) for value in eval_selector_feasible.tolist()]
                if eval_selector_feasible is not None
                else None
            ),
            "counterfactual_pair_features": (
                [
                    [float(item) for item in mode_features]
                    for mode_features in cf_candidates["aux"]
                    .get("counterfactual_pair_features", [])[0]
                    .detach()
                    .cpu()
                    .tolist()
                ]
                if isinstance(cf_candidates.get("aux"), dict)
                and cf_candidates["aux"].get("counterfactual_pair_features") is not None
                and cf_candidates["aux"].get("counterfactual_pair_features").ndim == 3
                else None
            ),
            "candidate_control_diagnostics": candidate_control_diagnostics,
            "candidates": candidates,
        }
        pair_rows.append(row)
        bucket = by_field[field_name]
        bucket["has_feasible_candidate"].append(float(row["has_feasible_candidate"]))
        bucket["scorer_feasible"].append(float(row["scorer_feasible"]))
        bucket["scorer_target_effect_z"].append(float(row["scorer_target_effect_z"]))
        bucket["scorer_raw_collateral_z"].append(float(row["scorer_raw_collateral_z"]))
        bucket["scorer_direction_drift_z"].append(float(row["scorer_direction_drift_z"]))
        bucket["scorer_independent_collateral_z"].append(float(row["scorer_independent_collateral_z"]))
        bucket["scorer_guardrail_collateral_z"].append(float(row["scorer_guardrail_collateral_z"]))
        bucket["best_target_effect_z"].append(float(row["best_target_effect_z"]))
        bucket["best_target_independent_collateral_z"].append(float(row["best_target_independent_collateral_z"]))
        bucket["best_target_guardrail_collateral_z"].append(float(row["best_target_guardrail_collateral_z"]))
        bucket["best_target_raw_collateral_z"].append(float(row["best_target_raw_collateral_z"]))
        bucket["best_target_direction_drift_z"].append(float(row["best_target_direction_drift_z"]))
        bucket["best_collateral_target_effect_z"].append(float(row["best_collateral_target_effect_z"]))
        bucket["best_collateral_guardrail_collateral_z"].append(float(row["best_collateral_guardrail_collateral_z"]))
        bucket["best_raw_collateral_target_effect_z"].append(float(row["best_raw_collateral_target_effect_z"]))
        bucket["best_raw_collateral_z"].append(float(row["best_raw_collateral_z"]))
        bucket["best_raw_collateral_direction_drift_z"].append(float(row["best_raw_collateral_direction_drift_z"]))
        bucket["best_feasible_target_effect_z"].append(float(row["best_feasible_target_effect_z"]))
        bucket["best_feasible_independent_collateral_z"].append(float(row["best_feasible_independent_collateral_z"]))
        bucket["best_feasible_guardrail_collateral_z"].append(float(row["best_feasible_guardrail_collateral_z"]))
        bucket["best_feasible_raw_collateral_z"].append(float(row["best_feasible_raw_collateral_z"]))
        bucket["best_feasible_direction_drift_z"].append(float(row["best_feasible_direction_drift_z"]))
        bucket["scorer_curvature_control_norm"].append(float(row["scorer_curvature_control_norm"]))
        bucket["scorer_counterfactual_residual_norm"].append(float(row["scorer_counterfactual_residual_norm"]))
        bucket["scorer_language_control_final_norm_mode"].append(
            float(row["scorer_language_control_final_norm_mode"])
        )
        bucket["scorer_language_control_final_endpoint_norm"].append(
            float(row["scorer_language_control_final_endpoint_norm"])
        )
        bucket["best_target_curvature_control_norm"].append(float(row["best_target_curvature_control_norm"]))
        bucket["best_target_counterfactual_residual_norm"].append(
            float(row["best_target_counterfactual_residual_norm"])
        )
        bucket["best_target_language_control_final_norm_mode"].append(
            float(row["best_target_language_control_final_norm_mode"])
        )
        bucket["best_target_language_control_final_endpoint_norm"].append(
            float(row["best_target_language_control_final_endpoint_norm"])
        )
        bucket["best_feasible_curvature_control_norm"].append(float(row["best_feasible_curvature_control_norm"]))
        bucket["best_feasible_counterfactual_residual_norm"].append(
            float(row["best_feasible_counterfactual_residual_norm"])
        )
        bucket["best_feasible_language_control_final_norm_mode"].append(
            float(row["best_feasible_language_control_final_norm_mode"])
        )
        bucket["best_feasible_language_control_final_endpoint_norm"].append(
            float(row["best_feasible_language_control_final_endpoint_norm"])
        )

    aggregate = {
        "num_pairs": int(len(pair_rows)),
        "feasible_candidate_rate": _safe_mean([float(row["has_feasible_candidate"]) for row in pair_rows]),
        "scorer_feasible_rate": _safe_mean([float(row["scorer_feasible"]) for row in pair_rows]),
        "scorer_target_effect_z": _summarize([float(row["scorer_target_effect_z"]) for row in pair_rows]),
        "scorer_raw_collateral_z": _summarize([float(row["scorer_raw_collateral_z"]) for row in pair_rows]),
        "scorer_direction_drift_z": _summarize([float(row["scorer_direction_drift_z"]) for row in pair_rows]),
        "scorer_independent_collateral_z": _summarize(
            [float(row["scorer_independent_collateral_z"]) for row in pair_rows]
        ),
        "scorer_guardrail_collateral_z": _summarize(
            [float(row["scorer_guardrail_collateral_z"]) for row in pair_rows]
        ),
        "best_target_effect_z": _summarize([float(row["best_target_effect_z"]) for row in pair_rows]),
        "best_target_independent_collateral_z": _summarize(
            [float(row["best_target_independent_collateral_z"]) for row in pair_rows]
        ),
        "best_target_guardrail_collateral_z": _summarize(
            [float(row["best_target_guardrail_collateral_z"]) for row in pair_rows]
        ),
        "best_target_raw_collateral_z": _summarize(
            [float(row["best_target_raw_collateral_z"]) for row in pair_rows]
        ),
        "best_target_direction_drift_z": _summarize(
            [float(row["best_target_direction_drift_z"]) for row in pair_rows]
        ),
        "best_collateral_target_effect_z": _summarize(
            [float(row["best_collateral_target_effect_z"]) for row in pair_rows]
        ),
        "best_collateral_guardrail_collateral_z": _summarize(
            [float(row["best_collateral_guardrail_collateral_z"]) for row in pair_rows]
        ),
        "best_raw_collateral_target_effect_z": _summarize(
            [float(row["best_raw_collateral_target_effect_z"]) for row in pair_rows]
        ),
        "best_raw_collateral_z": _summarize([float(row["best_raw_collateral_z"]) for row in pair_rows]),
        "best_raw_collateral_direction_drift_z": _summarize(
            [float(row["best_raw_collateral_direction_drift_z"]) for row in pair_rows]
        ),
        "best_feasible_target_effect_z": _summarize(
            [float(row["best_feasible_target_effect_z"]) for row in pair_rows]
        ),
        "best_feasible_independent_collateral_z": _summarize(
            [float(row["best_feasible_independent_collateral_z"]) for row in pair_rows]
        ),
        "best_feasible_guardrail_collateral_z": _summarize(
            [float(row["best_feasible_guardrail_collateral_z"]) for row in pair_rows]
        ),
        "best_feasible_raw_collateral_z": _summarize(
            [float(row["best_feasible_raw_collateral_z"]) for row in pair_rows]
        ),
        "best_feasible_direction_drift_z": _summarize(
            [float(row["best_feasible_direction_drift_z"]) for row in pair_rows]
        ),
        "scorer_curvature_control_norm": _summarize(
            [float(row["scorer_curvature_control_norm"]) for row in pair_rows]
        ),
        "scorer_counterfactual_residual_norm": _summarize(
            [float(row["scorer_counterfactual_residual_norm"]) for row in pair_rows]
        ),
        "scorer_language_control_final_norm_mode": _summarize(
            [float(row["scorer_language_control_final_norm_mode"]) for row in pair_rows]
        ),
        "scorer_language_control_final_endpoint_norm": _summarize(
            [float(row["scorer_language_control_final_endpoint_norm"]) for row in pair_rows]
        ),
        "best_target_curvature_control_norm": _summarize(
            [float(row["best_target_curvature_control_norm"]) for row in pair_rows]
        ),
        "best_target_counterfactual_residual_norm": _summarize(
            [float(row["best_target_counterfactual_residual_norm"]) for row in pair_rows]
        ),
        "best_target_language_control_final_norm_mode": _summarize(
            [float(row["best_target_language_control_final_norm_mode"]) for row in pair_rows]
        ),
        "best_target_language_control_final_endpoint_norm": _summarize(
            [float(row["best_target_language_control_final_endpoint_norm"]) for row in pair_rows]
        ),
        "best_feasible_curvature_control_norm": _summarize(
            [float(row["best_feasible_curvature_control_norm"]) for row in pair_rows]
        ),
        "best_feasible_counterfactual_residual_norm": _summarize(
            [float(row["best_feasible_counterfactual_residual_norm"]) for row in pair_rows]
        ),
        "best_feasible_language_control_final_norm_mode": _summarize(
            [float(row["best_feasible_language_control_final_norm_mode"]) for row in pair_rows]
        ),
        "best_feasible_language_control_final_endpoint_norm": _summarize(
            [float(row["best_feasible_language_control_final_endpoint_norm"]) for row in pair_rows]
        ),
    }
    by_field_summary = {}
    for field_name, bucket in sorted(by_field.items()):
        by_field_summary[field_name] = {
            "num_pairs": int(len(bucket["has_feasible_candidate"])),
            "feasible_candidate_rate": _safe_mean(bucket["has_feasible_candidate"]),
            "scorer_feasible_rate": _safe_mean(bucket["scorer_feasible"]),
            "scorer_target_effect_z": _summarize(bucket["scorer_target_effect_z"]),
            "scorer_raw_collateral_z": _summarize(bucket["scorer_raw_collateral_z"]),
            "scorer_direction_drift_z": _summarize(bucket["scorer_direction_drift_z"]),
            "scorer_independent_collateral_z": _summarize(bucket["scorer_independent_collateral_z"]),
            "scorer_guardrail_collateral_z": _summarize(bucket["scorer_guardrail_collateral_z"]),
            "best_target_effect_z": _summarize(bucket["best_target_effect_z"]),
            "best_target_independent_collateral_z": _summarize(bucket["best_target_independent_collateral_z"]),
            "best_target_guardrail_collateral_z": _summarize(bucket["best_target_guardrail_collateral_z"]),
            "best_target_raw_collateral_z": _summarize(bucket["best_target_raw_collateral_z"]),
            "best_target_direction_drift_z": _summarize(bucket["best_target_direction_drift_z"]),
            "best_collateral_target_effect_z": _summarize(bucket["best_collateral_target_effect_z"]),
            "best_collateral_guardrail_collateral_z": _summarize(bucket["best_collateral_guardrail_collateral_z"]),
            "best_raw_collateral_target_effect_z": _summarize(bucket["best_raw_collateral_target_effect_z"]),
            "best_raw_collateral_z": _summarize(bucket["best_raw_collateral_z"]),
            "best_raw_collateral_direction_drift_z": _summarize(bucket["best_raw_collateral_direction_drift_z"]),
            "best_feasible_target_effect_z": _summarize(bucket["best_feasible_target_effect_z"]),
            "best_feasible_independent_collateral_z": _summarize(bucket["best_feasible_independent_collateral_z"]),
            "best_feasible_guardrail_collateral_z": _summarize(bucket["best_feasible_guardrail_collateral_z"]),
            "best_feasible_raw_collateral_z": _summarize(bucket["best_feasible_raw_collateral_z"]),
            "best_feasible_direction_drift_z": _summarize(bucket["best_feasible_direction_drift_z"]),
            "scorer_curvature_control_norm": _summarize(bucket["scorer_curvature_control_norm"]),
            "scorer_counterfactual_residual_norm": _summarize(bucket["scorer_counterfactual_residual_norm"]),
            "scorer_language_control_final_norm_mode": _summarize(
                bucket["scorer_language_control_final_norm_mode"]
            ),
            "scorer_language_control_final_endpoint_norm": _summarize(
                bucket["scorer_language_control_final_endpoint_norm"]
            ),
            "best_target_curvature_control_norm": _summarize(bucket["best_target_curvature_control_norm"]),
            "best_target_counterfactual_residual_norm": _summarize(
                bucket["best_target_counterfactual_residual_norm"]
            ),
            "best_target_language_control_final_norm_mode": _summarize(
                bucket["best_target_language_control_final_norm_mode"]
            ),
            "best_target_language_control_final_endpoint_norm": _summarize(
                bucket["best_target_language_control_final_endpoint_norm"]
            ),
            "best_feasible_curvature_control_norm": _summarize(bucket["best_feasible_curvature_control_norm"]),
            "best_feasible_counterfactual_residual_norm": _summarize(
                bucket["best_feasible_counterfactual_residual_norm"]
            ),
            "best_feasible_language_control_final_norm_mode": _summarize(
                bucket["best_feasible_language_control_final_norm_mode"]
            ),
            "best_feasible_language_control_final_endpoint_norm": _summarize(
                bucket["best_feasible_language_control_final_endpoint_norm"]
            ),
        }
    return {
        "aggregate": aggregate,
        "by_field": by_field_summary,
        "pairs": pair_rows,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint, device, token_calibration_path=args.token_calibration_path)
    if args.counterfactual_direction_residual_type_override:
        if not hasattr(model.decoder, "set_counterfactual_direction_residual_type"):
            raise ValueError(
                "--counterfactual-direction-residual-type-override requires a DiT decoder "
                "with counterfactual direction residual support."
            )
        model.decoder.set_counterfactual_direction_residual_type(
            str(args.counterfactual_direction_residual_type_override)
        )
    if args.counterfactual_curvature_residual_type_override:
        setattr(
            model.decoder,
            "counterfactual_curvature_residual_type",
            str(args.counterfactual_curvature_residual_type_override),
        )
    if args.counterfactual_reference_residual_type_override:
        if not hasattr(model.decoder, "counterfactual_reference_residual_type"):
            raise ValueError(
                "--counterfactual-reference-residual-type-override requires a DiT decoder "
                "with counterfactual reference residual support."
            )
        model.decoder.counterfactual_reference_residual_type = str(
            args.counterfactual_reference_residual_type_override
        )
    eval_selector_overrides = apply_counterfactual_eval_selector_overrides(model, args)
    if args.counterfactual_residual_strength is not None:
        if not hasattr(model.decoder, "counterfactual_residual_strength"):
            raise ValueError(
                "--counterfactual-residual-strength requires a DiT decoder "
                "with counterfactual residual support."
            )
        model.decoder.counterfactual_residual_strength = float(args.counterfactual_residual_strength)
    if args.counterfactual_route_frame_budget_mode is not None:
        if not hasattr(model.decoder, "counterfactual_route_frame_budget_mode"):
            raise ValueError(
                "--counterfactual-route-frame-budget-mode requires a DiT decoder "
                "with route-frame budget support."
            )
        model.decoder.counterfactual_route_frame_budget_mode = str(args.counterfactual_route_frame_budget_mode)
    if args.counterfactual_route_frame_tangent_budget is not None:
        if not hasattr(model.decoder, "counterfactual_route_frame_tangent_budget"):
            raise ValueError(
                "--counterfactual-route-frame-tangent-budget requires a DiT decoder "
                "with route-frame budget support."
            )
        model.decoder.counterfactual_route_frame_tangent_budget = float(args.counterfactual_route_frame_tangent_budget)
    if args.counterfactual_route_frame_normal_budget is not None:
        if not hasattr(model.decoder, "counterfactual_route_frame_normal_budget"):
            raise ValueError(
                "--counterfactual-route-frame-normal-budget requires a DiT decoder "
                "with route-frame budget support."
            )
        model.decoder.counterfactual_route_frame_normal_budget = float(args.counterfactual_route_frame_normal_budget)
    if args.counterfactual_route_frame_fields is not None:
        parser_fn = getattr(model.decoder, "parse_route_frame_budget_fields", None)
        if parser_fn is None:
            raise ValueError(
                "--counterfactual-route-frame-fields requires a DiT decoder "
                "with route-frame budget support."
            )
        model.decoder.counterfactual_route_frame_fields = parser_fn(args.counterfactual_route_frame_fields)
    if args.reference_positive_candidate_strength_override is not None:
        if not hasattr(model.decoder, "set_reference_positive_candidate_strength"):
            raise ValueError(
                "--reference-positive-candidate-strength-override requires a compatible DiT decoder."
            )
        model.decoder.set_reference_positive_candidate_strength(
            float(args.reference_positive_candidate_strength_override)
        )
    if args.reference_positive_candidate_scope_override is not None:
        if not hasattr(model.decoder, "set_reference_positive_candidate_scope"):
            raise ValueError(
                "--reference-positive-candidate-scope-override requires a compatible DiT decoder."
            )
        model.decoder.set_reference_positive_candidate_scope(str(args.reference_positive_candidate_scope_override))
    saved_args = checkpoint.get("args", {})
    attr_dim = int(saved_args.get("attr_dim", 6))
    if args.qwen_planner_checkpoint is None:
        saved_qwen_checkpoint = saved_args.get("qwen_planner_checkpoint")
        if saved_qwen_checkpoint:
            args.qwen_planner_checkpoint = Path(saved_qwen_checkpoint)
    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device) if args.qwen_planner_checkpoint else None
    selection_prior = load_selection_prior(args.selection_prior_path, device)
    records = read_records(args.input, max_records=args.max_records)
    transition_field_min_success_rates = parse_field_float_map(
        args.transition_field_min_success_rates,
        name="transition-field-min-success-rates",
    )
    transition_field_min_effect_z = parse_field_float_map(
        args.transition_field_min_effect_z,
        name="transition-field-min-effect-z",
        allow_negative=True,
    )
    transition_field_min_token_match_rates = parse_field_float_map(
        args.transition_field_min_token_match_rates,
        name="transition-field-min-token-match-rates",
    )
    transition_field_min_pairs = parse_field_epochs(
        args.transition_field_min_pairs,
        name="transition-field-min-pairs",
    )
    transition_field_max_collateral_z = parse_field_float_map(
        args.transition_field_max_collateral_z,
        name="transition-field-max-collateral-z",
    )
    transition_table = load_counterfactual_transition_table(
        args.transition_diagnostics_path,
        min_success_rate=args.transition_min_success_rate,
        min_effect_z=args.transition_min_effect_z,
        min_pairs=args.transition_min_pairs,
        min_token_match_rate=args.transition_min_token_match_rate,
        max_collateral_z=args.transition_max_collateral_z,
        field_min_success_rate=transition_field_min_success_rates,
        field_min_effect_z=transition_field_min_effect_z,
        field_min_pairs=transition_field_min_pairs,
        field_min_token_match_rate=transition_field_min_token_match_rates,
        field_max_collateral_z=transition_field_max_collateral_z,
        allow_transitions=args.transition_allowlist,
    )
    pairs, transition_filter_stats = group_counterfactual_pairs(
        records,
        fields=set(args.fields),
        max_pairs=args.max_pairs,
        transition_table=transition_table,
        pair_sampling=args.pair_sampling,
    )
    if args.token_calibration_path is not None and args.token_calibration_path.exists():
        calibrator = GeometryTokenCalibrator.from_json(str(args.token_calibration_path))
        calibration_source = str(args.token_calibration_path)
    else:
        calibrator = build_pair_calibrator(pairs, args.target_key, model.decoder.target_len, device)
        calibration_source = "paired target trajectories"
    result = evaluate_candidate_pool(
        model=model,
        qwen_provider=qwen_provider,
        pairs=pairs,
        attr_dim=attr_dim,
        target_key=args.target_key,
        calibrator=calibrator,
        device=device,
        args=args,
        selection_prior=selection_prior,
    )
    metrics = {
        **{key: value for key, value in result.items() if key != "pairs"},
        "dataset": args.dataset,
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "target_key": args.target_key,
        "fields": args.fields,
        "success_margin_z": float(args.success_margin_z),
        "independent_collateral_margin_z": float(args.independent_collateral_margin_z),
        "collateral_field_mode": str(args.collateral_field_mode),
        "counterfactual_adapter_scale": float(args.counterfactual_adapter_scale),
        "reference_positive_candidate_strength_override": (
            None
            if args.reference_positive_candidate_strength_override is None
            else float(args.reference_positive_candidate_strength_override)
        ),
        "reference_positive_candidate_scope_override": args.reference_positive_candidate_scope_override,
        "counterfactual_curvature_residual_type": str(
            getattr(model.decoder, "counterfactual_curvature_residual_type", "unknown")
        ),
        "counterfactual_reference_residual_type": str(
            getattr(model.decoder, "counterfactual_reference_residual_type", "unknown")
        ),
        "counterfactual_direction_residual_type": str(
            getattr(model.decoder, "counterfactual_direction_residual_type", "unknown")
        ),
        "counterfactual_direction_residual_bank": (
            list(args.counterfactual_direction_residual_bank)
            if args.counterfactual_direction_residual_bank
            else None
        ),
        "counterfactual_residual_strength": float(
            getattr(model.decoder, "counterfactual_residual_strength", float("nan"))
        ),
        "selection_policy": args.selection_policy,
        "selection_prior_path": str(args.selection_prior_path) if args.selection_prior_path else None,
        "counterfactual_eval_selector": {
            "weight": float(getattr(model.decoder, "counterfactual_eval_selector_weight", 0.0)),
            "instruction_weight": float(getattr(model.decoder, "counterfactual_eval_selector_instruction_weight", 0.5)),
            "target_effect_weight": float(getattr(model.decoder, "counterfactual_eval_selector_target_effect_weight", 1.0)),
            "collateral_weight": float(getattr(model.decoder, "counterfactual_eval_selector_collateral_weight", 0.5)),
            "reference_reduction_weight": float(
                getattr(model.decoder, "counterfactual_eval_selector_reference_reduction_weight", 0.0)
            ),
            "reference_score_weight": float(
                getattr(model.decoder, "counterfactual_eval_selector_reference_score_weight", 0.0)
            ),
            "feasible_first": bool(getattr(model.decoder, "counterfactual_eval_selector_feasible_first", True)),
            "success_margin": float(getattr(model.decoder, "counterfactual_eval_selector_success_margin", 0.0)),
            "collateral_margin": float(getattr(model.decoder, "counterfactual_eval_selector_collateral_margin", 0.08)),
            "cli_overrides": eval_selector_overrides,
        },
        "transition_filter": {
            "diagnostics_path": str(args.transition_diagnostics_path) if args.transition_diagnostics_path else None,
            "enabled": bool(transition_table),
            "stats": transition_filter_stats,
            "pair_sampling": args.pair_sampling,
        },
        "qwen_planner_checkpoint": str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None,
        "qwen_semantic_conditioning": qwen_provider is not None,
        "calibration_source": calibration_source,
    }
    out_dir = args.output_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics["metrics_path"] = str(metrics_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_pair_details:
        details_path = out_dir / "pair_candidates.jsonl"
        with details_path.open("w", encoding="utf-8") as f:
            for row in result["pairs"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics["pair_details_path"] = str(details_path)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
