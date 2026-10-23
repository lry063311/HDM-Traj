from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from langdirecttraj.data.compact_store import (
    expand_compact_record,
    is_compact_meta_path,
    load_compact_manifest,
    open_compact_arrays,
)
from langdirecttraj.data.instruction_dataset import collate_instruction_batch, validate_official_record_protocol
from langdirecttraj.eval.eval_anchor_decoder import (
    build_model,
    load_checkpoint,
    prepare_eval_batch,
    summarize,
)
from langdirecttraj.eval.export_model_predictions import denormalize_predictions, reference_to_local, resample_np
from langdirecttraj.eval.instruction_geometry import (
    GeometryTokenCalibrator,
    add_pace_metrics_to_rows,
    geometry_values,
    tensor_dict_to_rows,
)
from langdirecttraj.eval.route_selection import (
    _transition_candidate_scores,
    load_selection_prior,
    select_prediction_with_policy,
)
from langdirecttraj.counterfactual_policy import (
    independent_collateral_field_names,
    serializable_collateral_exclusions,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    counterfactual_candidate_pool_guard_aux_diagnostics,
    extract_qwen_semantic_bundle,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    parse_field_epochs,
    parse_field_float_map,
    road_condition_kwargs,
    resolve_device,
)
from langdirecttraj.train.control_losses import load_counterfactual_transition_table


REPO_ROOT = Path(__file__).resolve().parents[2]
_ROAD_CORRIDOR_LOOKUP: Dict[int, Dict[str, np.ndarray | float]] = {}


def log_stage(message: str) -> None:
    print(f"[table4] {message}", flush=True)


def load_route_prior_lookup(path: Path | None, key: str) -> Dict[int, np.ndarray]:
    if path is None:
        return {}
    payload = np.load(path)
    if key not in payload:
        raise ValueError(f"{path} must contain route prior key '{key}'. Available keys: {list(payload.files)}")
    if "sample_id" in payload:
        sample_ids = np.asarray(payload["sample_id"], dtype=np.int64)
    elif "sample_ids" in payload:
        sample_ids = np.asarray(payload["sample_ids"], dtype=np.int64)
    else:
        raise ValueError(f"{path} must contain sample_id or sample_ids for route-prior alignment.")
    routes = np.asarray(payload[key], dtype=np.float32)
    if routes.ndim != 3 or routes.shape[-1] != 2:
        raise ValueError(f"Route prior '{key}' must be [N,T,2], got {routes.shape}")
    if sample_ids.shape[0] != routes.shape[0]:
        raise ValueError(f"Route prior id/route length mismatch: {sample_ids.shape[0]} vs {routes.shape[0]}")
    return {int(sid): routes[i] for i, sid in enumerate(sample_ids.tolist())}


def route_prior_to_local(
    record: Dict,
    route_prior_lookup: Dict[int, np.ndarray],
    target_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if not route_prior_lookup:
        return None
    sample_id = int(record.get("sample_id", -1))
    route = route_prior_lookup.get(sample_id)
    if route is None:
        return None
    reference = np.asarray(record["reference"], dtype=np.float32)
    _, origin, scale, route_scale = reference_to_local(reference)
    route_resampled = resample_np(np.asarray(route, dtype=np.float32), int(target_len))
    route_m = (route_resampled - origin[None, :]) * scale[None, :]
    route_local = route_m / max(float(route_scale), 1e-6)
    return torch.from_numpy(route_local.astype(np.float32)).to(device=device).unsqueeze(0)


def blend_route_prior_base(
    base: torch.Tensor | None,
    route_prior: torch.Tensor,
    weight: float,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_dtype = dtype or (base.dtype if torch.is_tensor(base) else torch.float32)
    route = route_prior.to(device=device, dtype=target_dtype)
    w = max(0.0, min(float(weight), 1.0))
    if base is None or w >= 1.0:
        return route
    base_tensor = base.to(device=device, dtype=target_dtype)
    if base_tensor.ndim == 2:
        base_tensor = base_tensor.unsqueeze(0)
    if route.shape[1] != base_tensor.shape[1]:
        route = torch.nn.functional.interpolate(
            route.transpose(1, 2),
            size=int(base_tensor.shape[1]),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return (1.0 - w) * base_tensor + w * route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Table 4: counterfactual controllability evaluation.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "counterfactual_controllability_eval")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument(
        "--source-num-shards",
        type=int,
        default=1,
        help=(
            "Split the compact counterfactual file by source_sample_id for local full evaluation. "
            "Use with --source-shard-index; all shards together form the full set."
        ),
    )
    parser.add_argument(
        "--source-shard-index",
        type=int,
        default=0,
        help="Shard index in [0, source-num-shards). Keeps original/counterfactual pairs together.",
    )
    parser.add_argument(
        "--pair-sampling",
        choices=["sequential", "balanced_field"],
        default="balanced_field",
        help=(
            "How to choose pairs when --max-pairs is smaller than the available set. "
            "balanced_field is the official local-goal setting because Table 4 must not be dominated "
            "by whichever field appears first in the JSONL."
        ),
    )
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["distance", "direction", "curvature", "progress", "reference_adherence"],
    )
    parser.add_argument(
        "--success-margin-z",
        type=float,
        default=0.0,
        help="A counterfactual succeeds if target-token distance improves by more than this normalized margin.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--disable-qwen", action="store_true", help="Do not inherit or load a saved Qwen planner.")
    parser.add_argument("--token-calibration-path", type=Path, default=None)
    parser.add_argument("--road-corridor-cache", type=Path, default=None)
    parser.add_argument(
        "--route-prior-npz",
        type=Path,
        default=None,
        help=(
            "Optional Stage-2 route-memory output. When provided, the original "
            "sample's selected route skeleton is used as the base trajectory for "
            "same-OD local-edit evaluation."
        ),
    )
    parser.add_argument(
        "--route-prior-key",
        default="generated_trajectories",
        help="Trajectory array key inside --route-prior-npz.",
    )
    parser.add_argument(
        "--route-prior-required",
        action="store_true",
        help="Fail if any evaluated original sample is missing from --route-prior-npz.",
    )
    parser.add_argument(
        "--route-prior-blend-weight",
        type=float,
        default=1.0,
        help=(
            "Blend Stage-2 route prior into the Stage-1 selected base trajectory for Table IV. "
            "1.0 preserves the previous hard-replacement behavior; small values such as "
            "0.05/0.10/0.20 test weak route-prior fusion."
        ),
    )
    parser.add_argument(
        "--counterfactual-adapter-scale",
        type=float,
        default=None,
        help=(
            "Enable decoder counterfactual adapters only for changed-token records. "
            "Defaults to the checkpoint training value when available; pass 0 for an explicit ablation."
        ),
    )
    parser.add_argument(
        "--counterfactual-residual-strength",
        type=float,
        default=None,
        help=(
            "Optional evaluation override for decoder.counterfactual_residual_strength. "
            "Use this to sweep counterfactual residual strength on older checkpoints."
        ),
    )
    parser.add_argument(
        "--counterfactual-route-frame-budget-mode",
        choices=["none", "clamp", "soft_tanh"],
        default=None,
        help=(
            "Optional evaluation override for bounded route-frame counterfactual residuals. "
            "Default keeps the checkpoint configuration."
        ),
    )
    parser.add_argument("--counterfactual-route-frame-tangent-budget", type=float, default=None)
    parser.add_argument("--counterfactual-route-frame-normal-budget", type=float, default=None)
    parser.add_argument(
        "--counterfactual-route-frame-fields",
        default=None,
        help=(
            "Optional comma-separated fields affected by the route-frame budget. "
            "Default keeps the checkpoint configuration."
        ),
    )
    parser.add_argument(
        "--reference-positive-candidate-strength-override",
        type=float,
        default=None,
        help=(
            "Optional evaluation override for explicit close-reference candidate construction. "
            "Use this to test whether reference_adherence=tight failures are candidate-pool "
            "coverage failures or selector failures."
        ),
    )
    parser.add_argument(
        "--reference-positive-candidate-scope-override",
        choices=["all", "none", "counterfactual_reference_adherence"],
        default=None,
        help=(
            "Optional evaluation override for the explicit close-reference candidate scope. "
            "Use counterfactual_reference_adherence to activate it only for reference_adherence "
            "counterfactual edits."
        ),
    )
    parser.add_argument(
        "--reference-distance-candidate-strength-override",
        type=float,
        default=None,
        help="Optional evaluation override for bounded reference-distance candidate proposals.",
    )
    parser.add_argument(
        "--reference-distance-candidate-scope-override",
        choices=["all", "none", "instruction", "counterfactual_reference_adherence"],
        default=None,
        help="Optional evaluation override for bounded reference-distance proposal scope.",
    )
    parser.add_argument(
        "--reference-distance-candidate-mode-override",
        choices=["global_normal", "local_direction_preserving"],
        default=None,
        help="Optional evaluation override for bounded reference-distance proposal shape.",
    )
    parser.add_argument(
        "--counterfactual-local-proposal-type-override",
        choices=["none", "mask_controlled_v1"],
        default=None,
        help="Optional evaluation override for the mask-conditioned local proposal branch.",
    )
    parser.add_argument(
        "--counterfactual-local-proposal-strength-override",
        type=float,
        default=None,
        help="Optional evaluation override for the local proposal branch strength.",
    )
    parser.add_argument(
        "--base-preserving-selector-enabled-override",
        action="store_true",
        help="Enable the base-preserving base/control candidate family split during evaluation.",
    )
    parser.add_argument(
        "--transition-diagnostics-path",
        type=Path,
        default=None,
        help="Optional diagnostics JSON. When set, Table 4 only evaluates transitions that pass the same quality gate as training.",
    )
    parser.add_argument("--transition-min-success-rate", type=float, default=0.6)
    parser.add_argument("--transition-min-effect-z", type=float, default=0.05)
    parser.add_argument("--transition-min-token-match-rate", type=float, default=0.5)
    parser.add_argument("--transition-min-pairs", type=int, default=2)
    parser.add_argument("--transition-field-min-success-rates", default="")
    parser.add_argument("--transition-field-min-effect-z", default="")
    parser.add_argument("--transition-field-min-token-match-rates", default="")
    parser.add_argument("--transition-field-min-pairs", default="")
    parser.add_argument(
        "--transition-max-collateral-z",
        type=float,
        default=0.0,
        help="Optional transition-quality gate: drop pseudo target transitions whose collateral drift is above this z-score.",
    )
    parser.add_argument("--transition-field-max-collateral-z", default="")
    parser.add_argument(
        "--transition-allowlist",
        default="",
        help="Optional comma-separated transition keys. When set, Table 4 evaluates only these transitions.",
    )
    parser.add_argument(
        "--selection-policy",
        choices=[
            "reranker",
            "od_cell_prior",
            "reranker_plus_od_cell_prior",
            "reranker_guarded_od_cell_prior",
            "transition_prior",
            "realism_guarded_transition_prior",
            "reranker_guarded_transition_prior",
            "target_direction_transition_prior",
            "target_control_transition_prior",
            "reranker_guarded_target_control_transition_prior",
            "fixed_mode_0",
            "fixed_mode_1",
            "fixed_mode_2",
            "fixed_mode_3",
            "fixed_mode_4",
            "fixed_mode_5",
            "hybrid_counterfactual",
        ],
        default=None,
    )
    parser.add_argument("--selection-prior-path", type=Path, default=None)
    parser.add_argument("--selection-route-weight", type=float, default=1.0)
    parser.add_argument("--selection-token-guard-margin", type=float, default=0.0)
    parser.add_argument("--selection-token-absolute-threshold", type=float, default=float("inf"))
    parser.add_argument("--selection-min-route-improvement", type=float, default=0.0)
    parser.add_argument("--selection-logit-guard-margin", type=float, default=float("inf"))
    parser.add_argument("--selection-preserve-direction", action="store_true")
    parser.add_argument("--selection-preserve-target-direction", action="store_true")
    parser.add_argument(
        "--qwen-plan-scale-override",
        type=float,
        default=None,
        help="Evaluation-only override for the dense Qwen plan-token conditioning scale.",
    )
    parser.add_argument("--route-proposal-logit-weight-override", type=float, default=None)
    parser.add_argument("--route-control-selection-adapter-weight-override", type=float, default=None)
    parser.add_argument("--route-control-selection-adapter-clip-override", type=float, default=None)
    parser.add_argument("--selection-hybrid-instruction-weight", type=float, default=None)
    parser.add_argument("--selection-hybrid-target-effect-weight", type=float, default=None)
    parser.add_argument("--selection-hybrid-collateral-weight", type=float, default=None)
    parser.add_argument("--selection-hybrid-collateral-margin", type=float, default=None)
    parser.add_argument("--selection-hybrid-raw-collateral-margin", type=float, default=None)
    parser.add_argument("--selection-hybrid-logit-weight", type=float, default=None)
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
        help=(
            "Override selector collateral fields. 'curvature_raw' keeps the official "
            "independent policy for other fields but uses raw non-target drift for curvature edits; "
            "'raw' uses raw non-target drift for every counterfactual field."
        ),
    )
    parser.add_argument("--counterfactual-eval-selector-disable-feasible-first", action="store_true")
    parser.add_argument(
        "--selector-noncollapse-mode-override",
        choices=["disabled", "additive"],
        default=None,
        help=(
            "Optional evaluation-only override for decoder.selector_noncollapse_mode. "
            "Use disabled when diagnosing pure reranker logits without the additive "
            "target/collateral/base-preservation heuristic."
        ),
    )
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
            "Table-4 reranking. Not supported for selection_policy=hybrid_counterfactual."
        ),
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
            "Evaluation-only override for curvature counterfactual residual variants. "
            "Empty keeps the checkpoint configuration."
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
            "Table-4 reranking. Not supported for selection_policy=hybrid_counterfactual."
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
        "--export-failure-cases",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for pair-level Table 4 diagnostics. "
            "Records are sorted by raw collateral drift so local runs can identify the transitions "
            "that corrupt non-target geometry."
        ),
    )
    parser.add_argument(
        "--export-candidate-rows",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for per-candidate Table 4 diagnostics. "
            "This is independent from --export-failure-cases so scorer/proposal "
            "alignment can be audited without exporting pair-level failures."
        ),
    )
    parser.add_argument(
        "--export-visual-cases",
        type=Path,
        default=None,
        help=(
            "Optional directory for Figure-style counterfactual visualization bundles. "
            "Exports selected base/counterfactual trajectories and reference coordinates."
        ),
    )
    parser.add_argument(
        "--max-visual-cases",
        type=int,
        default=12,
        help="Maximum number of visualization bundles to export when --export-visual-cases is set.",
    )
    parser.add_argument(
        "--max-failure-cases",
        type=int,
        default=64,
        help="Maximum number of pair-level diagnostics to export when --export-failure-cases is set.",
    )
    parser.add_argument(
        "--oracle-candidate-diagnostics",
        action="store_true",
        help=(
            "Evaluate all same-mode base/counterfactual candidate pairs and report whether "
            "the pool contains target-effective low-collateral candidates. This does not "
            "change Table 4 selection or metrics."
        ),
    )
    parser.add_argument(
        "--oracle-candidate-independent-margin",
        type=float,
        default=0.06,
        help="Independent-collateral threshold used only by --oracle-candidate-diagnostics.",
    )
    parser.add_argument(
        "--oracle-candidate-raw-margin",
        type=float,
        default=0.15,
        help="Raw-collateral threshold used only by --oracle-candidate-diagnostics.",
    )
    parser.add_argument(
        "--selection-hybrid-priority",
        choices=[
            "weighted",
            "feasible_first",
            "target_feasible_first",
            "target_joint_feasible_first",
            "pareto_feasible_first",
        ],
        default=None,
        help=(
            "Metric-aligned policy for Table 4 hybrid_counterfactual selection. It selects a "
            "same-mode original/counterfactual candidate by route transition score, instruction "
            "target score, target effect, and independent collateral."
        ),
    )
    return parser.parse_args()


def load_road_corridor_lookup(path: Path | None) -> Dict[int, Dict[str, np.ndarray | float]]:
    if path is None:
        return {}
    cache = np.load(path)
    if "sample_id" not in cache:
        raise ValueError(f"Road corridor cache must include sample_id for Table 4 lookup: {path}")
    sample_ids = np.asarray(cache["sample_id"], dtype=np.int64)
    lookup: Dict[int, Dict[str, np.ndarray | float]] = {}
    for row, sample_id in enumerate(sample_ids.tolist()):
        lookup[int(sample_id)] = {
            "road_corridor_tokens": np.asarray(cache["road_corridor_tokens"][row], dtype=np.float32),
            "road_corridor_mask": np.asarray(cache["road_corridor_mask"][row], dtype=np.bool_),
            "road_path_length_ratio": np.asarray(cache["road_path_length_ratio"][row], dtype=np.float32),
            "road_path_total_length_m": np.asarray(cache["road_path_total_length_m"][row], dtype=np.float32),
            "road_nearest_origin_dist_m": float(cache["road_nearest_origin_dist_m"][row]),
            "road_nearest_destination_dist_m": float(cache["road_nearest_destination_dist_m"][row]),
            "road_path_success": np.asarray(cache["road_path_success"][row], dtype=np.bool_),
        }
    return lookup


def apply_counterfactual_eval_selector_overrides(model, args: argparse.Namespace) -> Dict[str, object]:
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return {}
    overrides: Dict[str, object] = {}
    float_overrides = {
        "counterfactual_eval_selector_weight": args.counterfactual_eval_selector_weight,
        "counterfactual_eval_selector_instruction_weight": args.counterfactual_eval_selector_instruction_weight,
        "counterfactual_eval_selector_target_effect_weight": args.counterfactual_eval_selector_target_effect_weight,
        "counterfactual_eval_selector_collateral_weight": args.counterfactual_eval_selector_collateral_weight,
        "counterfactual_eval_selector_mode_switch_weight": args.counterfactual_eval_selector_mode_switch_weight,
        "counterfactual_eval_selector_reference_mode_switch_weight": (
            args.counterfactual_eval_selector_reference_mode_switch_weight
        ),
        "counterfactual_eval_selector_reference_reduction_weight": (
            args.counterfactual_eval_selector_reference_reduction_weight
        ),
        "counterfactual_eval_selector_reference_score_weight": (
            args.counterfactual_eval_selector_reference_score_weight
        ),
        "counterfactual_eval_selector_success_margin": args.counterfactual_eval_selector_success_margin,
        "counterfactual_eval_selector_collateral_margin": args.counterfactual_eval_selector_collateral_margin,
    }
    for name, value in float_overrides.items():
        if value is None:
            continue
        setattr(decoder, name, float(value))
        overrides[name] = float(value)
    if args.counterfactual_eval_selector_collateral_mode is not None:
        decoder.counterfactual_eval_selector_collateral_mode = str(
            args.counterfactual_eval_selector_collateral_mode
        )
        overrides["counterfactual_eval_selector_collateral_mode"] = str(
            args.counterfactual_eval_selector_collateral_mode
        )
    if bool(args.counterfactual_eval_selector_disable_feasible_first):
        decoder.counterfactual_eval_selector_feasible_first = False
        overrides["counterfactual_eval_selector_feasible_first"] = False
    if args.selector_noncollapse_mode_override is not None:
        decoder.selector_noncollapse_mode = str(args.selector_noncollapse_mode_override)
        overrides["selector_noncollapse_mode"] = str(args.selector_noncollapse_mode_override)
    return overrides


def read_records(
    path: Path,
    max_records: int = 0,
    source_num_shards: int = 1,
    source_shard_index: int = 0,
) -> List[Dict]:
    if source_num_shards < 1:
        raise ValueError("--source-num-shards must be >= 1")
    if source_shard_index < 0 or source_shard_index >= source_num_shards:
        raise ValueError("--source-shard-index must satisfy 0 <= index < source-num-shards")
    records = []
    compact_arrays = {}
    if is_compact_meta_path(Path(path)):
        compact_arrays = open_compact_arrays(load_compact_manifest(Path(path)))
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if source_num_shards > 1:
                cf = record.get("counterfactual", {})
                source_id = int(cf.get("source_sample_id", record.get("sample_id", -1)))
                if source_id % source_num_shards != source_shard_index:
                    continue
            if compact_arrays:
                record = expand_compact_record(record, compact_arrays)
            validate_official_record_protocol(record, row_index=len(records))
            records.append(record)
            if max_records > 0 and len(records) >= max_records:
                break
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def tensor_record(record: Dict, target_key: str) -> Dict:
    traj = np.asarray(record.get(target_key, record.get("trajectory", [])), dtype=np.float32)
    reference = np.asarray(record.get("reference", []), dtype=np.float32)
    if reference.ndim != 2 or reference.shape[-1] != 2 or reference.shape[0] == 0:
        raise ValueError(
            "Table 4 requires an explicit official OD-chord reference; "
            f"sample_id={record.get('sample_id', record.get('counterfactual_id', 'unknown'))!r} "
            f"has reference shape {reference.shape}."
        )
    labels = dict(record.get("labels", {}))
    sample_id = int(record.get("sample_id", record.get("counterfactual_id", 0)))
    item = {
        "sample_id": sample_id,
        "dataset": record.get("dataset", ""),
        "traj": torch.from_numpy(traj).float(),
        "reference": torch.from_numpy(reference).float(),
        "mobility_tokens": torch.tensor(DEFAULT_MOBILITY_VOCAB.encode(labels), dtype=torch.long),
        "labels": labels,
        "instruction": str(record.get("instruction", "")),
        "metrics": dict(record.get("metrics", {})),
        "trip_attributes": dict(record.get("trip_attributes", {})),
    }
    if _ROAD_CORRIDOR_LOOKUP:
        cf = record.get("counterfactual", {})
        road_sample_id = int(cf.get("source_sample_id", sample_id) if isinstance(cf, dict) else sample_id)
        road_item = _ROAD_CORRIDOR_LOOKUP.get(road_sample_id) or _ROAD_CORRIDOR_LOOKUP.get(sample_id)
        if road_item:
            item.update(road_item)
    return item


def is_allowed_transition(record: Dict, transition_table: Dict[str, Dict[int, List[tuple[int, float]]]]) -> bool:
    if not transition_table:
        return True
    cf = record.get("counterfactual", {})
    field = str(cf.get("field", ""))
    if field not in transition_table:
        return False
    source_token = str(cf.get("from", ""))
    target_token = str(cf.get("to", ""))
    token_to_id = DEFAULT_MOBILITY_VOCAB.token_to_id.get(field, {})
    if source_token not in token_to_id or target_token not in token_to_id:
        return False
    source_id = int(token_to_id[source_token])
    target_id = int(token_to_id[target_token])
    return any(int(candidate_id) == target_id for candidate_id, _ in transition_table.get(field, {}).get(source_id, []))


def group_counterfactual_pairs(
    records: List[Dict],
    fields: set[str],
    max_pairs: int,
    transition_table: Dict[str, Dict[int, List[tuple[int, float]]]] | None = None,
    pair_sampling: str = "balanced_field",
) -> Tuple[List[Tuple[Dict, Dict]], Dict[str, int]]:
    originals: Dict[int, Dict] = {}
    counterfactuals: Dict[int, List[Dict]] = defaultdict(list)
    filter_stats = {"candidate_pairs": 0, "filtered_by_transition": 0}
    for record in records:
        cf = record.get("counterfactual", {})
        source_id = int(cf.get("source_sample_id", record.get("sample_id", -1)))
        if cf.get("enabled"):
            if not fields or str(cf.get("field", "")) in fields:
                filter_stats["candidate_pairs"] += 1
                if transition_table and not is_allowed_transition(record, transition_table):
                    filter_stats["filtered_by_transition"] += 1
                    continue
                counterfactuals[source_id].append(record)
        else:
            originals[int(record.get("sample_id", source_id))] = record

    all_pairs: List[Tuple[Dict, Dict]] = []
    for source_id, cf_records in counterfactuals.items():
        original = originals.get(source_id)
        if original is None:
            continue
        for cf_record in cf_records:
            all_pairs.append((original, cf_record))
    if max_pairs <= 0 or len(all_pairs) <= max_pairs:
        return all_pairs, filter_stats
    if pair_sampling == "sequential":
        return all_pairs[:max_pairs], filter_stats
    if pair_sampling != "balanced_field":
        raise ValueError(f"Unsupported pair_sampling={pair_sampling}")

    field_order = [field for field in DEFAULT_MOBILITY_VOCAB.fields if not fields or field in fields]
    buckets: Dict[str, List[Tuple[Dict, Dict]]] = {field: [] for field in field_order}
    for pair in all_pairs:
        field = str(pair[1].get("counterfactual", {}).get("field", ""))
        if field in buckets:
            buckets[field].append(pair)
    selected: List[Tuple[Dict, Dict]] = []
    cursor = {field: 0 for field in field_order}
    while len(selected) < max_pairs:
        added = False
        for field in field_order:
            idx = cursor[field]
            if idx >= len(buckets[field]):
                continue
            selected.append(buckets[field][idx])
            cursor[field] = idx + 1
            added = True
            if len(selected) >= max_pairs:
                break
        if not added:
            break
    filter_stats["pair_sampling"] = pair_sampling
    filter_stats["available_by_field"] = {field: len(buckets[field]) for field in field_order}
    filter_stats["selected_by_field"] = {
        field: sum(1 for _, cf_record in selected if str(cf_record.get("counterfactual", {}).get("field", "")) == field)
        for field in field_order
    }
    return selected, filter_stats


def build_pair_calibrator(
    pairs: List[Tuple[Dict, Dict]],
    target_key: str,
    target_len: int,
    device: torch.device,
) -> GeometryTokenCalibrator:
    rows: List[tuple[Dict[str, str], Dict[str, float]]] = []
    with torch.no_grad():
        iterator = pairs
        if tqdm is not None:
            iterator = tqdm(
                pairs,
                desc="table4 calibrator pairs",
                total=len(pairs),
                dynamic_ncols=True,
                mininterval=5.0,
            )
        for original, counterfactual in iterator:
            for record in (original, counterfactual):
                batch = collate_instruction_batch([tensor_record(record, target_key=target_key)])
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                target, reference, route_scales = prepare_eval_batch(batch, target_len)
                geom = tensor_dict_to_rows(geometry_values(target.float(), reference.float(), route_scales[:, 0].float()))[0]
                geom = add_pace_metrics_to_rows([geom], batch.get("trip_attributes", [{}]))[0]
                rows.append((record.get("labels", {}), geom))
    return GeometryTokenCalibrator.from_target_batches(rows)


def generate_geometry(
    model,
    qwen_provider,
    record: Dict,
    attr_dim: int,
    target_key: str,
    device: torch.device,
    counterfactual_adapter_scale: float = 0.0,
    selection_prior: dict | None = None,
    selection_policy: str = "reranker",
    selection_route_weight: float = 1.0,
    selection_token_guard_margin: float = 0.0,
    selection_token_absolute_threshold: float = float("inf"),
    selection_min_route_improvement: float = 0.0,
    selection_logit_guard_margin: float = float("inf"),
    selection_preserve_direction: bool = False,
    selection_preserve_target_direction: bool = False,
    base_selected_trajectory: torch.Tensor | None = None,
    base_mobility_tokens: torch.Tensor | None = None,
    base_selected_index: torch.Tensor | None = None,
    counterfactual_direction_residual_bank: List[str] | None = None,
    counterfactual_curvature_residual_bank: List[str] | None = None,
    return_candidates: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], Dict]:
    item = tensor_record(record, target_key=target_key)
    batch = collate_instruction_batch([item])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    cf_candidates = generate_candidate_geometries_with_residual_banks(
        model,
        qwen_provider,
        record,
        attr_dim,
        target_key,
        device,
        direction_residual_bank=counterfactual_direction_residual_bank,
        curvature_residual_bank=counterfactual_curvature_residual_bank,
        counterfactual_adapter_scale=counterfactual_adapter_scale,
        base_selected_trajectory=base_selected_trajectory,
        base_mobility_tokens=base_mobility_tokens,
        base_selected_index=base_selected_index,
    )
    pred_out = (cf_candidates["candidates"], {"candidate_logits": cf_candidates["logits"], **cf_candidates.get("aux", {})})
    pred, selected_idx, _ = select_prediction_with_policy(
        pred_out,
        reference=cf_candidates["reference"],
        reference_lonlat=cf_candidates["reference_lonlat"],
        route_scales=cf_candidates["route_scales"],
        prior=selection_prior,
        policy=selection_policy,
        route_weight=selection_route_weight,
        token_guard_margin=selection_token_guard_margin,
        token_absolute_threshold=selection_token_absolute_threshold,
        min_route_improvement=selection_min_route_improvement,
        logit_guard_margin=selection_logit_guard_margin,
        preserve_direction=selection_preserve_direction,
        preserve_target_direction=selection_preserve_target_direction,
        mobility_tokens=batch["mobility_tokens"],
        dataset=str(record.get("dataset", "porto") or "porto"),
    )
    geom = tensor_dict_to_rows(
        geometry_values(pred.float(), cf_candidates["reference"].float(), cf_candidates["route_scales"].float())
    )[0]
    geom = add_pace_metrics_to_rows([geom], batch.get("trip_attributes", [{}]))[0]
    selected_mode = int(selected_idx.detach().view(-1)[0].item())
    geom["_selected_index"] = selected_mode
    proposal_family_by_candidate = cf_candidates.get("proposal_family_by_candidate", [])
    if 0 <= selected_mode < len(proposal_family_by_candidate):
        geom["_selected_proposal_family"] = str(proposal_family_by_candidate[selected_mode])
    if return_candidates:
        return geom, cf_candidates
    return geom


def generate_candidate_geometries(
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
    item = tensor_record(record, target_key=target_key)
    batch = collate_instruction_batch([item])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    _, reference, route_scales = prepare_eval_batch(batch, model.decoder.target_len)
    ref_len = int(batch["reference_mask"][0].sum().item())
    ref_lonlat = batch["reference"][0, : max(ref_len, 1)].detach().cpu().float().numpy()
    _, ref_origin, ref_scale, ref_route_scale = reference_to_local(ref_lonlat)
    attr_cond = make_attr_cond(batch, attr_dim, device)
    trip_duration_min = torch.tensor(
        [
            float(attrs.get("trip_duration_min", 0.0) or 0.0)
            if isinstance(attrs, dict)
            else 0.0
            for attrs in batch.get("trip_attributes", [{}])
        ],
        device=device,
        dtype=torch.float32,
    )
    semantic_plan_tokens = extract_qwen_semantic_bundle(qwen_provider, batch["instruction"], device)
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
    with torch.no_grad():
        pred_out = model(
            reference,
            batch["mobility_tokens"],
            attr_cond,
            semantic_plan_tokens=semantic_plan_tokens,
            **road_condition_kwargs(batch),
            counterfactual_mode=counterfactual_mode,
            counterfactual_adapter_scale=float(counterfactual_adapter_scale),
            counterfactual_field_ids=cf_field_ids,
            base_selected_trajectory=base_selected_trajectory,
            base_mobility_tokens=base_mobility_tokens,
            base_selected_index=base_selected_index,
            route_scales=route_scales[:, 0],
            trip_duration_min=trip_duration_min,
            return_aux=True,
        )
    if not isinstance(pred_out, tuple):
        pred = pred_out[:, None] if pred_out.ndim == 3 else pred_out
        aux = {}
    else:
        pred, aux = pred_out
        if pred.ndim == 3:
            pred = pred[:, None]
    logits = aux.get("candidate_logits") if isinstance(aux, dict) else None
    if logits is None:
        logits = pred.new_zeros(pred.shape[:2])
    else:
        logits = logits[:, : pred.shape[1]]

    geoms: List[Dict[str, float]] = []
    for mode_idx in range(pred.shape[1]):
        mode_pred = pred[:, mode_idx]
        geom = tensor_dict_to_rows(
            geometry_values(mode_pred.float(), reference.float(), route_scales[:, 0].float())
        )[0]
        geom = add_pace_metrics_to_rows([geom], batch.get("trip_attributes", [{}]))[0]
        geoms.append(geom)
    return {
        "geoms": geoms,
        "candidates": pred,
        "logits": logits,
        "aux": aux if isinstance(aux, dict) else {},
        "reference": reference,
        "reference_lonlat": batch["reference"],
        "denorm_meta": [{"origin": ref_origin, "scale": ref_scale, "route_scale": ref_route_scale}],
        "route_scales": route_scales[:, 0],
    }


def _merge_candidate_family_outputs(family_outputs: List[Tuple[str, Dict]]) -> Dict:
    if not family_outputs:
        raise ValueError("family_outputs must not be empty")
    if len(family_outputs) == 1:
        merged = dict(family_outputs[0][1])
        merged["proposal_family_by_candidate"] = [family_outputs[0][0] for _ in merged["geoms"]]
        merged["family_mode_by_candidate"] = list(range(len(merged["geoms"])))
        return merged
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
            for key, value in aux.items():
                if torch.is_tensor(value) and value.ndim >= 2:
                    family_aux[key].append(value)
    merged_aux: Dict[str, torch.Tensor] = {}
    for key, parts in family_aux.items():
        try:
            merged_aux[key] = torch.cat(parts, dim=1)
        except RuntimeError:
            continue
    return {
        "geoms": geoms,
        "candidates": torch.cat(candidate_parts, dim=1),
        "logits": torch.cat(logits_parts, dim=1),
        "aux": merged_aux,
        "reference": family_outputs[0][1]["reference"],
        "reference_lonlat": family_outputs[0][1]["reference_lonlat"],
        "denorm_meta": family_outputs[0][1].get("denorm_meta", []),
        "route_scales": family_outputs[0][1]["route_scales"],
        "proposal_family_by_candidate": proposal_family_by_candidate,
        "family_mode_by_candidate": family_mode_by_candidate,
    }


def generate_candidate_geometries_with_residual_banks(
    model,
    qwen_provider,
    record: Dict,
    attr_dim: int,
    target_key: str,
    device: torch.device,
    direction_residual_bank: List[str] | None = None,
    curvature_residual_bank: List[str] | None = None,
    counterfactual_adapter_scale: float = 0.0,
    base_selected_trajectory: torch.Tensor | None = None,
    base_mobility_tokens: torch.Tensor | None = None,
    base_selected_index: torch.Tensor | None = None,
) -> Dict:
    field_name = str(record.get("counterfactual", {}).get("field", ""))
    use_direction_bank = (
        bool(direction_residual_bank)
        and field_name == "direction"
        and float(counterfactual_adapter_scale) != 0.0
    )
    use_curvature_bank = (
        bool(curvature_residual_bank)
        and field_name == "curvature"
        and float(counterfactual_adapter_scale) != 0.0
    )
    if not use_direction_bank and not use_curvature_bank:
        return generate_candidate_geometries(
            model,
            qwen_provider,
            record,
            attr_dim,
            target_key,
            device,
            counterfactual_adapter_scale=counterfactual_adapter_scale,
            base_selected_trajectory=base_selected_trajectory,
            base_mobility_tokens=base_mobility_tokens,
            base_selected_index=base_selected_index,
        )
    decoder = getattr(model, "decoder", None)
    if use_direction_bank and (decoder is None or not hasattr(decoder, "set_counterfactual_direction_residual_type")):
        raise ValueError(
            "--counterfactual-direction-residual-bank requires a DiT decoder "
            "with counterfactual direction residual support."
        )
    if use_curvature_bank and (decoder is None or not hasattr(decoder, "counterfactual_curvature_residual_type")):
        raise ValueError(
            "--counterfactual-curvature-residual-bank requires a DiT decoder "
            "with counterfactual curvature residual support."
        )
    original_direction_residual_type = getattr(decoder, "counterfactual_direction_residual_type", None)
    original_curvature_residual_type = getattr(decoder, "counterfactual_curvature_residual_type", None)
    family_outputs: List[Tuple[str, Dict]] = []
    try:
        if use_direction_bank:
            family_specs = [
                (f"direction::{residual_type}", str(residual_type), str(original_curvature_residual_type or ""))
                for residual_type in (direction_residual_bank or [])
            ]
        else:
            family_specs = [
                (f"curvature::{residual_type}", str(original_direction_residual_type or ""), str(residual_type))
                for residual_type in (curvature_residual_bank or [])
            ]
        for family_name, direction_residual_type, curvature_residual_type in family_specs:
            if use_direction_bank:
                decoder.set_counterfactual_direction_residual_type(str(direction_residual_type))
            if use_curvature_bank:
                decoder.counterfactual_curvature_residual_type = str(curvature_residual_type)
            family_outputs.append(
                (
                    family_name,
                    generate_candidate_geometries(
                        model,
                        qwen_provider,
                        record,
                        attr_dim,
                        target_key,
                        device,
                        counterfactual_adapter_scale=counterfactual_adapter_scale,
                        base_selected_trajectory=base_selected_trajectory,
                        base_mobility_tokens=base_mobility_tokens,
                        base_selected_index=base_selected_index,
                    ),
                )
            )
    finally:
        if use_direction_bank and original_direction_residual_type is not None:
            decoder.set_counterfactual_direction_residual_type(str(original_direction_residual_type))
        if original_curvature_residual_type is not None:
            decoder.counterfactual_curvature_residual_type = str(original_curvature_residual_type)
    return _merge_candidate_family_outputs(family_outputs)


def _normalize_cost(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    centered = values - float(np.nanmin(values))
    scale = float(np.nanstd(centered))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanmean(np.abs(centered))) if centered.size else 1.0
    return centered / max(scale, 1e-6)


def _selection_target_effect_z(
    calibrator: GeometryTokenCalibrator,
    field_name: str,
    source_token: str,
    target_token: str,
    base_geom: Dict[str, float],
    cf_geom: Dict[str, float],
) -> tuple[float, float, float]:
    """Return selector target effect plus token/reference diagnostic effects.

    For reference_adherence, Table 4 should rank the candidate that actually
    moves toward the reference polyline.  The generic token-score distance can
    disagree with that local control metric when the base-selected path is
    already near a calibration center.  Keep the token-score effect as a
    diagnostic, but use the signed reference metric movement for selection.
    """
    base_score = calibrator.score_token(field_name, target_token, base_geom)
    cf_score = calibrator.score_token(field_name, target_token, cf_geom)
    token_effect = float(base_score - cf_score)
    reference_effect = float("nan")
    if field_name == "reference_adherence":
        _, _, _, _, signed_metric_delta_z = calibrator.transition_metric_delta(
            field_name,
            source_token,
            target_token,
            base_geom,
            cf_geom,
        )
        if np.isfinite(float(signed_metric_delta_z)):
            reference_effect = float(signed_metric_delta_z)
            return reference_effect, token_effect, reference_effect
    return token_effect, token_effect, reference_effect


def select_hybrid_counterfactual_geometries(
    model,
    qwen_provider,
    original: Dict,
    counterfactual: Dict,
    attr_dim: int,
    target_key: str,
    calibrator: GeometryTokenCalibrator,
    device: torch.device,
    success_margin_z: float,
    counterfactual_adapter_scale: float,
    selection_prior: dict | None,
    selection_route_weight: float,
    instruction_weight: float,
    target_effect_weight: float,
    collateral_weight: float,
    collateral_margin: float,
    raw_collateral_margin: float,
    logit_weight: float,
    priority: str,
    route_prior_base_pred: torch.Tensor | None = None,
    route_prior_blend_weight: float = 1.0,
    counterfactual_direction_residual_bank: List[str] | None = None,
    counterfactual_curvature_residual_bank: List[str] | None = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    cf_meta = counterfactual.get("counterfactual", {})
    field_name = str(cf_meta.get("field", "unknown"))
    target_token = str(cf_meta.get("to", counterfactual.get("labels", {}).get(field_name, "unknown")))
    base = generate_candidate_geometries(
        model,
        qwen_provider,
        original,
        attr_dim,
        target_key,
        device,
    )
    base_tokens = tensor_record(original, target_key=target_key)["mobility_tokens"].to(device).view(1, -1)
    base_logits = base.get("logits")
    if torch.is_tensor(base_logits) and base_logits.ndim >= 2 and base_logits.shape[1] > 0:
        base_selected_index = base_logits.argmax(dim=1)
    else:
        base_selected_index = torch.zeros(1, device=device, dtype=torch.long)
    base_candidates = base.get("candidates")
    base_selected_trajectory = None
    if torch.is_tensor(base_candidates) and base_candidates.ndim == 4 and base_candidates.shape[1] > 0:
        gather_idx = base_selected_index.to(device=base_candidates.device, dtype=torch.long).view(-1, 1, 1, 1)
        gather_idx = gather_idx.expand(-1, 1, base_candidates.shape[-2], base_candidates.shape[-1])
        base_selected_trajectory = base_candidates.gather(1, gather_idx).squeeze(1)
    if route_prior_base_pred is not None:
        target_dtype = base_selected_trajectory.dtype if torch.is_tensor(base_selected_trajectory) else torch.float32
        target_device = base_selected_trajectory.device if torch.is_tensor(base_selected_trajectory) else device
        base_selected_trajectory = blend_route_prior_base(
            base_selected_trajectory,
            route_prior_base_pred,
            route_prior_blend_weight,
            device=target_device,
            dtype=target_dtype,
        )
    cf = generate_candidate_geometries_with_residual_banks(
        model,
        qwen_provider,
        counterfactual,
        attr_dim,
        target_key,
        device,
        direction_residual_bank=counterfactual_direction_residual_bank,
        curvature_residual_bank=counterfactual_curvature_residual_bank,
        counterfactual_adapter_scale=counterfactual_adapter_scale,
        base_selected_trajectory=base_selected_trajectory,
        base_mobility_tokens=base_tokens,
        base_selected_index=base_selected_index,
    )
    num_modes = len(cf["geoms"])
    if num_modes <= 0:
        raise ValueError("hybrid_counterfactual selection requires at least one candidate mode.")
    base_selected_mode = int(base_selected_index.detach().cpu().long().view(-1)[0].item())
    if not (0 <= base_selected_mode < len(base["geoms"])):
        base_selected_mode = 0
    if route_prior_base_pred is not None:
        route_base = base_selected_trajectory.to(device=base["reference"].device, dtype=base["reference"].dtype)
        base_selected_geom = tensor_dict_to_rows(
            geometry_values(route_base.float(), base["reference"].float(), base["route_scales"].float())
        )[0]
        trip_attributes = original.get("trip_attributes", [{}])
        if not isinstance(trip_attributes, list):
            trip_attributes = [trip_attributes]
        base_selected_geom = add_pace_metrics_to_rows([base_selected_geom], trip_attributes)[0]
        base_selected_geom["_selected_index"] = base_selected_mode
        base_selected_geom["_route_prior_base"] = 1.0
        base_selected_geom["_route_prior_blend_weight"] = float(route_prior_blend_weight)
    else:
        base_selected_geom = base["geoms"][base_selected_mode]

    route_scores = np.zeros(num_modes, dtype=np.float64)
    if selection_prior is not None and "transition_nll" in selection_prior:
        with torch.no_grad():
            route_tensor = _transition_candidate_scores(
                cf["candidates"][:, :num_modes],
                cf["reference_lonlat"],
                cf["route_scales"],
                selection_prior,
            )
        route_scores = route_tensor[0, :num_modes].detach().cpu().float().numpy().astype(np.float64)
    route_cost = _normalize_cost(route_scores)
    logits = cf.get("logits")
    if torch.is_tensor(logits) and logits.ndim >= 2 and logits.shape[1] >= num_modes:
        logit_raw = logits[0, :num_modes].detach().cpu().float().numpy().astype(np.float64)
        logit_centered = logit_raw - float(np.nanmean(logit_raw))
        logit_scale = float(np.nanstd(logit_centered))
        if not np.isfinite(logit_scale) or logit_scale < 1e-6:
            logit_scale = 1.0
        logit_scores = logit_centered / logit_scale
    else:
        logit_raw = np.full(num_modes, np.nan, dtype=np.float64)
        logit_scores = np.zeros(num_modes, dtype=np.float64)
    effective_logit_weight = float(logit_weight) if np.isfinite(float(logit_weight)) else 0.0

    costs: List[float] = []
    target_effects: List[float] = []
    token_target_effects: List[float] = []
    reference_target_effects: List[float] = []
    collateral_values: List[float] = []
    raw_collateral_values: List[float] = []
    feasible: List[bool] = []
    instruction_scores: List[float] = []
    for mode_idx in range(num_modes):
        cf_geom = cf["geoms"][mode_idx]
        cf_target_score = calibrator.score_token(field_name, target_token, cf_geom)
        target_effect, token_effect, reference_effect = _selection_target_effect_z(
            calibrator,
            field_name,
            str(cf_meta.get("from", original.get("labels", {}).get(field_name, "unknown"))),
            target_token,
            base_selected_geom,
            cf_geom,
        )
        drift_by_field = collateral_drift_z_by_field(
            calibrator,
            field_name,
            original.get("labels", {}),
            base_selected_geom,
            cf_geom,
        )
        raw_collateral = float(np.mean(list(drift_by_field.values()))) if drift_by_field else float("nan")
        independent_drift_by_field = independent_collateral_fields(field_name, drift_by_field)
        collateral = (
            float(np.mean(list(independent_drift_by_field.values())))
            if independent_drift_by_field
            else float(np.mean(list(drift_by_field.values()))) if drift_by_field else 0.0
        )
        if not np.isfinite(collateral):
            collateral = 0.0
        cost = (
            float(selection_route_weight) * float(route_cost[mode_idx])
            + float(instruction_weight) * float(cf_target_score)
            - float(target_effect_weight) * target_effect
            + float(collateral_weight) * collateral
            - effective_logit_weight * float(logit_scores[mode_idx])
        )
        costs.append(float(cost))
        target_effects.append(target_effect)
        token_target_effects.append(token_effect)
        reference_target_effects.append(reference_effect)
        collateral_values.append(collateral)
        raw_collateral_values.append(raw_collateral)
        feasible.append(target_effect > float(success_margin_z) and collateral <= float(collateral_margin))
        instruction_scores.append(float(cf_target_score))

    cost_arr = np.asarray(costs, dtype=np.float64)
    feasible_arr = np.asarray(feasible, dtype=bool)
    target_arr = np.asarray(target_effects, dtype=np.float64)
    raw_arr = np.asarray(raw_collateral_values, dtype=np.float64)
    raw_feasible_arr = feasible_arr & np.isfinite(raw_arr) & (raw_arr <= float(raw_collateral_margin))
    adjusted_target_arr = target_arr + effective_logit_weight * logit_scores
    if str(priority) == "pareto_feasible_first" and bool(raw_feasible_arr.any()):
        pareto_cost = (
            -float(target_effect_weight) * target_arr
            + float(collateral_weight) * np.asarray(collateral_values, dtype=np.float64)
            + 0.25 * raw_arr
            - effective_logit_weight * logit_scores
        )
        masked = np.where(raw_feasible_arr, pareto_cost, np.inf)
        selected_idx = int(np.argmin(masked))
    elif str(priority) == "target_joint_feasible_first" and bool(raw_feasible_arr.any()):
        masked = np.where(raw_feasible_arr, adjusted_target_arr, -np.inf)
        selected_idx = int(np.argmax(masked))
    elif str(priority) == "target_feasible_first" and bool(feasible_arr.any()):
        masked = np.where(feasible_arr, adjusted_target_arr, -np.inf)
        selected_idx = int(np.argmax(masked))
    elif str(priority) == "feasible_first" and bool(feasible_arr.any()):
        masked = np.where(feasible_arr, cost_arr, np.inf)
        selected_idx = int(np.argmin(masked))
    else:
        selected_idx = int(np.argmin(cost_arr))
    stats = {
        "hybrid_selected_mode": float(selected_idx),
        "hybrid_selected_cost": float(cost_arr[selected_idx]),
        "hybrid_selected_route_score": float(route_scores[selected_idx]),
        "hybrid_selected_logit": float(logit_raw[selected_idx]) if np.isfinite(logit_raw[selected_idx]) else float("nan"),
        "hybrid_selected_logit_score": float(logit_scores[selected_idx]),
        "hybrid_selected_target_effect": float(target_effects[selected_idx]),
        "hybrid_selected_collateral": float(collateral_values[selected_idx]),
        "hybrid_selected_instruction_score": float(instruction_scores[selected_idx]),
        "hybrid_feasible": float(feasible_arr[selected_idx]),
        "hybrid_any_feasible": float(feasible_arr.any()),
    }
    stats["_candidate_target_effects"] = target_effects
    stats["_candidate_token_target_effects"] = token_target_effects
    stats["_candidate_reference_target_effects"] = reference_target_effects
    stats["_candidate_costs"] = [float(value) for value in cost_arr.tolist()]
    stats["_candidate_instruction_scores"] = [float(value) for value in instruction_scores]
    stats["_candidate_selector_feasible"] = [bool(value) for value in feasible_arr.tolist()]
    stats["_candidate_selector_raw_feasible"] = [bool(value) for value in raw_feasible_arr.tolist()]
    stats["_candidate_adjusted_targets"] = [float(value) for value in adjusted_target_arr.tolist()]
    stats["_candidate_independent_collateral"] = collateral_values
    stats["_candidate_raw_collateral"] = raw_collateral_values
    stats["_candidate_logits"] = [float(value) if np.isfinite(value) else float("nan") for value in logit_raw.tolist()]
    stats["_candidate_logit_scores"] = [float(value) for value in logit_scores.tolist()]
    control_mask_tensor = cf.get("aux", {}).get("candidate_control_family_mask")
    if torch.is_tensor(control_mask_tensor):
        stats["_candidate_control_family_mask"] = (
            control_mask_tensor.detach().cpu().float().reshape(-1).tolist()
        )
    proposal_families = cf.get("proposal_family_by_candidate", [])
    if proposal_families:
        stats["_candidate_proposal_families"] = [str(value) for value in proposal_families[:num_modes]]
    family_modes = cf.get("family_mode_by_candidate", [])
    if family_modes:
        stats["_candidate_family_modes"] = [int(value) for value in family_modes[:num_modes]]
    cf_aux = cf.get("aux", {})
    if isinstance(cf_aux, dict):
        forward_aux_stats = counterfactual_candidate_pool_guard_aux_diagnostics(cf_aux)
        stats["_forward_reference_l2t_diagnostics"] = {
            key: value
            for key, value in forward_aux_stats.items()
            if "_reference_inward_" not in key
        }
        stats["_forward_reference_inward_diagnostics"] = {
            key: value
            for key, value in forward_aux_stats.items()
            if "_reference_inward_" in key
        }
    stats["_selected_idx"] = int(selected_idx)
    visual_base_selected = (
        base_selected_trajectory.to(device=base_candidates.device, dtype=base_candidates.dtype)
        if route_prior_base_pred is not None
        and base_selected_trajectory is not None
        and torch.is_tensor(base_candidates)
        else base_candidates[:, base_selected_mode]
    )
    stats["_visual_arrays"] = {
        "reference": _tensor_to_numpy(cf["reference_lonlat"])[0],
        "target": _tensor_to_numpy(tensor_record(original, target_key=target_key)["traj"]),
        "base_selected": denormalize_predictions(
            visual_base_selected, base.get("denorm_meta", cf.get("denorm_meta", []))
        )[0],
        "edited_selected": denormalize_predictions(
            cf["candidates"][:, selected_idx], cf.get("denorm_meta", [])
        )[0],
        "base_candidates": denormalize_predictions(
            base_candidates.contiguous().reshape(-1, base_candidates.shape[-2], base_candidates.shape[-1]),
            base.get("denorm_meta", cf.get("denorm_meta", [])) * int(base_candidates.shape[1]),
        ).reshape(base_candidates.shape[1], base_candidates.shape[-2], base_candidates.shape[-1]),
        "edited_candidates": denormalize_predictions(
            cf["candidates"].contiguous().reshape(-1, cf["candidates"].shape[-2], cf["candidates"].shape[-1]),
            cf.get("denorm_meta", []) * int(cf["candidates"].shape[1]),
        ).reshape(cf["candidates"].shape[1], cf["candidates"].shape[-2], cf["candidates"].shape[-1]),
    }
    return base_selected_geom, cf["geoms"][selected_idx], stats


def collateral_drift_z(
    calibrator: GeometryTokenCalibrator,
    target_field: str,
    original_labels: Dict[str, str],
    base_geom: Dict[str, float],
    cf_geom: Dict[str, float],
) -> float:
    values: List[float] = []
    for field_name in DEFAULT_MOBILITY_VOCAB.fields:
        if field_name == target_field:
            continue
        token = str(original_labels.get(field_name, "unknown"))
        if token not in calibrator.stats[field_name]:
            continue
        values.append(calibrator.normalized_metric_delta(field_name, token, base_geom, cf_geom))
    return float(np.mean(values)) if values else float("nan")


def collateral_drift_z_by_field(
    calibrator: GeometryTokenCalibrator,
    target_field: str,
    original_labels: Dict[str, str],
    base_geom: Dict[str, float],
    cf_geom: Dict[str, float],
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for field_name in DEFAULT_MOBILITY_VOCAB.fields:
        if field_name == target_field:
            continue
        token = str(original_labels.get(field_name, "unknown"))
        if token not in calibrator.stats[field_name]:
            continue
        values[field_name] = calibrator.normalized_metric_delta(field_name, token, base_geom, cf_geom)
    return values


def independent_collateral_fields(target_field: str, drift_by_field: Dict[str, float]) -> Dict[str, float]:
    independent_fields = set(independent_collateral_field_names(target_field, fields=drift_by_field.keys()))
    return {
        field_name: value
        for field_name, value in drift_by_field.items()
        if field_name in independent_fields
    }


def _safe_collateral_mean(values: Dict[str, float] | List[float] | tuple[float, ...]) -> float:
    if isinstance(values, dict):
        seq = [float(value) for value in values.values()]
    else:
        seq = [float(value) for value in values]
    finite = [value for value in seq if np.isfinite(value)]
    if not finite:
        return 0.0
    return float(np.mean(finite))


def _candidate_logit(logits: object, mode_idx: int) -> float:
    if not torch.is_tensor(logits) or logits.ndim < 2 or logits.shape[1] <= mode_idx:
        return float("nan")
    return float(logits[0, mode_idx].detach().cpu().float().item())


def _candidate_step_norms(
    base_pred: torch.Tensor,
    candidates: object,
    route_scale: object,
    mode_idx: int,
) -> Tuple[float, float]:
    if not torch.is_tensor(candidates) or candidates.ndim < 4 or candidates.shape[1] <= mode_idx:
        return float("nan"), float("nan")
    base = base_pred.detach()
    if base.ndim == 3:
        base = base[0]
    candidate = candidates[0, mode_idx].detach()
    step_norm = torch.linalg.norm(candidate - base.to(candidate.device), dim=-1).mean()
    step_norm_norm = float(step_norm.cpu().float().item())
    scale = 1.0
    if torch.is_tensor(route_scale):
        scale = float(route_scale.detach().cpu().float().reshape(-1)[0].item())
    elif route_scale is not None:
        scale = float(route_scale)
    return step_norm_norm, step_norm_norm * scale


def _tensor_to_numpy(value: object) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def _selected_candidate_numpy(candidates: object, selected_idx: int) -> np.ndarray:
    arr = _tensor_to_numpy(candidates)
    if arr.ndim == 4:
        return arr[0, int(selected_idx)]
    if arr.ndim == 3:
        return arr[int(selected_idx)]
    return arr


def write_visual_cases(path: Path, records: List[Dict[str, object]], max_records: int) -> Dict[str, object]:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    def _visual_change(row: Dict[str, object]) -> float:
        arrays = row.get("_visual_arrays")
        if not isinstance(arrays, dict):
            return 0.0
        base = np.asarray(arrays.get("base_selected", []), dtype=np.float32)
        edited = np.asarray(arrays.get("edited_selected", []), dtype=np.float32)
        if base.shape != edited.shape or base.size == 0:
            return 0.0
        return float(np.nanmean(np.linalg.norm(base - edited, axis=-1)))

    def _rank_key(row: Dict[str, object]) -> tuple:
        effect = float(row.get("target_effect_z", 0.0))
        collateral = float(row.get("independent_collateral_z", float("inf")))
        visual_change = _visual_change(row)
        return (
            str(row.get("transition", "")) != "reference_adherence:loose->tight",
            not bool(row.get("success", False)),
            effect < 0.05,
            collateral > 0.03,
            visual_change <= 1e-8,
            -effect,
            collateral,
        )

    ranked = sorted(
        records,
        key=_rank_key,
    )
    selected = ranked[: max(0, int(max_records))]
    manifest_records: List[Dict[str, object]] = []
    for export_idx, record in enumerate(selected):
        arrays = record.get("_visual_arrays")
        if not isinstance(arrays, dict):
            continue
        file_name = (
            f"case_{export_idx:03d}_sample_{int(record.get('sample_id', -1))}"
            f"_cf_{int(record.get('counterfactual_id', -1))}.npz"
        )
        npz_path = path / file_name
        np.savez_compressed(
            npz_path,
            reference=np.asarray(arrays["reference"], dtype=np.float32),
            target=np.asarray(arrays["target"], dtype=np.float32),
            base_selected=np.asarray(arrays["base_selected"], dtype=np.float32),
            edited_selected=np.asarray(arrays["edited_selected"], dtype=np.float32),
            base_candidates=np.asarray(arrays["base_candidates"], dtype=np.float32),
            edited_candidates=np.asarray(arrays["edited_candidates"], dtype=np.float32),
        )
        clean_record = {
            key: value
            for key, value in record.items()
            if key != "_visual_arrays" and not str(key).startswith("_")
        }
        clean_record["npz"] = file_name
        clean_record["visual_change_lonlat_mean"] = _visual_change(record)
        manifest_records.append(clean_record)
    manifest = {
        "num_records": len(manifest_records),
        "records": manifest_records,
        "selection_note": (
            "Rows are sorted to prioritize successful reference_adherence:loose->tight "
            "cases with target_effect_z >= 0.05, independent_collateral_z <= 0.03, "
            "and nonzero visual trajectory change."
        ),
    }
    (path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (path / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for record in manifest_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"dir": str(path), "num_records": len(manifest_records)}


def write_failure_cases(path: Path, records: List[Dict[str, object]], max_records: int) -> Dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = max(0, int(max_records))
    selected = records[:limit] if limit > 0 else list(records)
    with path.open("w", encoding="utf-8") as f:
        for record in selected:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = path.with_suffix(".csv")
    scalar_keys = [
        "sample_id",
        "counterfactual_id",
        "transition",
        "changed_field",
        "source_token",
        "target_token",
        "target_metric",
        "target_effect_z",
        "success",
        "signed_metric_delta",
        "signed_metric_delta_z",
        "raw_collateral_z",
        "independent_collateral_z",
        "max_raw_collateral_field",
        "max_raw_collateral_z",
        "base_selected_index",
        "cf_selected_index",
        "base_length_ratio",
        "cf_length_ratio",
        "base_mean_reference_distance_norm",
        "cf_mean_reference_distance_norm",
        "base_local_total_abs_turn_rad",
        "cf_local_total_abs_turn_rad",
        "base_avg_speed_kmh",
        "cf_avg_speed_kmh",
    ]
    drift_fields = sorted(
        {
            field_name
            for record in selected
            for field_name in dict(record.get("collateral_by_non_target", {})).keys()
        }
    )
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=scalar_keys + [f"drift_{field_name}" for field_name in drift_fields],
        )
        writer.writeheader()
        for record in selected:
            row = {key: record.get(key) for key in scalar_keys}
            drift_values = dict(record.get("collateral_by_non_target", {}))
            for field_name in drift_fields:
                row[f"drift_{field_name}"] = drift_values.get(field_name)
            writer.writerow(row)
    return {
        "jsonl_path": str(path),
        "csv_path": str(csv_path),
        "num_exported": int(len(selected)),
        "num_available": int(len(records)),
    }


def write_candidate_rows(path: Path, records: List[Dict[str, object]]) -> Dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "jsonl_path": str(path),
        "num_exported": int(len(records)),
    }


def _record_selectable_candidate_diagnostics(
    *,
    target_effects: List[float],
    raw_collateral: List[float],
    independent_collateral: List[float],
    selected_idx: int,
    transition_key: str,
    success_margin_z: float,
    independent_margin: float,
    raw_margin: float,
    selectable_candidate_stats: Dict[str, List[float]],
    selectable_candidate_by_transition: Dict[str, Dict[str, List[float]]],
    control_mask: List[bool] | np.ndarray | None = None,
) -> None:
    target_arr = np.asarray(target_effects, dtype=np.float64)
    raw_arr = np.asarray(raw_collateral, dtype=np.float64)
    independent_arr = np.asarray(independent_collateral, dtype=np.float64)
    if target_arr.size == 0:
        return
    control_arr = None
    if control_mask is not None:
        control_arr = np.asarray(control_mask, dtype=bool)
        if control_arr.shape != target_arr.shape:
            control_arr = np.zeros(target_arr.shape, dtype=bool)

    success_mask = target_arr > float(success_margin_z)
    independent_feasible = success_mask & np.isfinite(independent_arr) & (
        independent_arr <= float(independent_margin)
    )
    raw_feasible = success_mask & np.isfinite(raw_arr) & (
        raw_arr <= float(raw_margin)
    )
    joint_feasible = independent_feasible & raw_feasible
    selected_target_effect = (
        float(target_arr[selected_idx])
        if 0 <= int(selected_idx) < target_arr.size
        else float("nan")
    )
    selected_independent = (
        float(independent_arr[selected_idx])
        if 0 <= int(selected_idx) < independent_arr.size
        else float("nan")
    )

    def _record(prefix: str, mask: np.ndarray) -> None:
        if not bool(mask.any()):
            return
        masked_target = np.where(mask, target_arr, np.nan)
        masked_independent = np.where(mask, independent_arr, np.nan)
        masked_raw = np.where(mask, raw_arr, np.nan)
        masked_independent_feasible = mask & independent_feasible
        masked_raw_feasible = mask & raw_feasible
        masked_joint_feasible = mask & joint_feasible
        best_target = float(np.nanmax(masked_target))
        values = {
            f"{prefix}_num_modes": float(mask.sum()),
            f"{prefix}_best_target_effect": best_target,
            f"{prefix}_selected_target_effect": selected_target_effect,
            f"{prefix}_selected_minus_best_target_effect": (
                selected_target_effect - best_target
                if np.isfinite(selected_target_effect) and np.isfinite(best_target)
                else float("nan")
            ),
            f"{prefix}_selected_independent_collateral": selected_independent,
            f"{prefix}_min_independent_collateral": float(np.nanmin(masked_independent)),
            f"{prefix}_min_raw_collateral": float(np.nanmin(masked_raw)),
            f"{prefix}_any_independent_feasible": float(masked_independent_feasible.any()),
            f"{prefix}_any_raw_feasible": float(masked_raw_feasible.any()),
            f"{prefix}_any_joint_feasible": float(masked_joint_feasible.any()),
            f"{prefix}_best_target_under_independent_margin": (
                float(np.nanmax(np.where(masked_independent_feasible, target_arr, np.nan)))
                if bool(masked_independent_feasible.any())
                else float("nan")
            ),
            f"{prefix}_best_target_under_raw_margin": (
                float(np.nanmax(np.where(masked_raw_feasible, target_arr, np.nan)))
                if bool(masked_raw_feasible.any())
                else float("nan")
            ),
            f"{prefix}_best_target_under_joint_margin": (
                float(np.nanmax(np.where(masked_joint_feasible, target_arr, np.nan)))
                if bool(masked_joint_feasible.any())
                else float("nan")
            ),
        }
        best_under_independent = values[f"{prefix}_best_target_under_independent_margin"]
        values[f"{prefix}_selected_minus_best_target_under_independent_margin"] = (
            selected_target_effect - best_under_independent
            if np.isfinite(selected_target_effect) and np.isfinite(best_under_independent)
            else float("nan")
        )
        for metric_key, metric_value in values.items():
            selectable_candidate_stats[metric_key].append(metric_value)
            selectable_candidate_by_transition[transition_key][metric_key].append(metric_value)

    _record("selectable", np.ones(target_arr.shape, dtype=bool))
    if control_arr is not None:
        _record("selectable_control", control_arr)


def evaluate_pairs(
    model,
    qwen_provider,
    pairs: List[Tuple[Dict, Dict]],
    attr_dim: int,
    target_key: str,
    calibrator: GeometryTokenCalibrator,
    device: torch.device,
    success_margin_z: float,
    counterfactual_adapter_scale: float = 0.0,
    selection_prior: dict | None = None,
    selection_policy: str = "reranker",
    selection_route_weight: float = 1.0,
    selection_token_guard_margin: float = 0.0,
    selection_token_absolute_threshold: float = float("inf"),
    selection_min_route_improvement: float = 0.0,
    selection_logit_guard_margin: float = float("inf"),
    selection_preserve_direction: bool = False,
    selection_preserve_target_direction: bool = False,
    selection_hybrid_instruction_weight: float = 0.5,
    selection_hybrid_target_effect_weight: float = 1.0,
    selection_hybrid_collateral_weight: float = 0.5,
    selection_hybrid_collateral_margin: float = 0.08,
    selection_hybrid_raw_collateral_margin: float = 0.15,
    selection_hybrid_logit_weight: float = 0.0,
    selection_hybrid_priority: str = "weighted",
    collect_case_records: bool = False,
    collect_oracle_candidate_diagnostics: bool = False,
    oracle_candidate_independent_margin: float = 0.06,
    oracle_candidate_raw_margin: float = 0.15,
    counterfactual_direction_residual_bank: List[str] | None = None,
    counterfactual_curvature_residual_bank: List[str] | None = None,
    route_prior_lookup: Dict[int, np.ndarray] | None = None,
    route_prior_required: bool = False,
    route_prior_blend_weight: float = 1.0,
) -> Dict:
    target_effects: List[float] = []
    success_flags: List[float] = []
    collateral_values: List[float] = []
    independent_collateral_values: List[float] = []
    independent_collateral_defined_flags: List[float] = []
    metric_deltas: List[float] = []
    metric_delta_z_values: List[float] = []
    hybrid_selection_stats = defaultdict(list)
    oracle_candidate_stats = defaultdict(list)
    oracle_candidate_by_transition = defaultdict(lambda: defaultdict(list))
    selectable_candidate_stats = defaultdict(list)
    selectable_candidate_by_transition = defaultdict(lambda: defaultdict(list))
    case_records: List[Dict[str, object]] = []
    candidate_records: List[Dict[str, object]] = []
    forward_reference_l2t_stats = defaultdict(list)
    forward_reference_inward_stats = defaultdict(list)
    route_prior_used = 0
    route_prior_missing = 0
    collateral_by_non_target = defaultdict(list)
    independent_collateral_by_non_target = defaultdict(list)
    by_field = defaultdict(
        lambda: {
            "target_effect_z": [],
            "success": [],
            "collateral_drift_z": [],
            "independent_collateral_drift_z": [],
            "metric_delta": [],
            "metric_delta_z": [],
            "collateral_by_non_target": defaultdict(list),
            "independent_collateral_by_non_target": defaultdict(list),
        }
    )
    by_transition = defaultdict(
        lambda: {
            "target_effect_z": [],
            "success": [],
            "collateral_drift_z": [],
            "independent_collateral_drift_z": [],
            "metric_delta": [],
            "metric_delta_z": [],
            "collateral_by_non_target": defaultdict(list),
            "independent_collateral_by_non_target": defaultdict(list),
        }
    )

    iterator = pairs
    if tqdm is not None:
        iterator = tqdm(
            pairs,
            desc="table4 counterfactual pairs",
            total=len(pairs),
            dynamic_ncols=True,
            mininterval=5.0,
        )
    for original, counterfactual in iterator:
        cf_meta = counterfactual.get("counterfactual", {})
        field_name = str(cf_meta.get("field", "unknown"))
        source_token = str(cf_meta.get("from", original.get("labels", {}).get(field_name, "unknown")))
        target_token = str(cf_meta.get("to", counterfactual.get("labels", {}).get(field_name, "unknown")))
        transition_key = f"{field_name}:{source_token}->{target_token}"
        if field_name not in DEFAULT_MOBILITY_VOCAB.fields or target_token not in calibrator.stats[field_name]:
            continue
        route_prior_base_pred = route_prior_to_local(
            original,
            route_prior_lookup or {},
            int(getattr(model.decoder, "target_len", 128)),
            device,
        )
        if route_prior_base_pred is None and route_prior_lookup:
            route_prior_missing += 1
            if route_prior_required:
                raise ValueError(
                    f"Missing route prior for original sample_id={int(original.get('sample_id', -1))}"
                )
        elif route_prior_base_pred is not None:
            route_prior_used += 1

        if selection_policy == "hybrid_counterfactual":
            base_geom, cf_geom, hybrid_stats = select_hybrid_counterfactual_geometries(
                model,
                qwen_provider,
                original,
                counterfactual,
                attr_dim,
                target_key,
                calibrator,
                device,
                success_margin_z=success_margin_z,
                counterfactual_adapter_scale=float(counterfactual_adapter_scale),
                selection_prior=selection_prior,
                selection_route_weight=selection_route_weight,
                instruction_weight=selection_hybrid_instruction_weight,
                target_effect_weight=selection_hybrid_target_effect_weight,
                collateral_weight=selection_hybrid_collateral_weight,
                collateral_margin=selection_hybrid_collateral_margin,
                raw_collateral_margin=selection_hybrid_raw_collateral_margin,
                logit_weight=selection_hybrid_logit_weight,
                priority=selection_hybrid_priority,
                route_prior_base_pred=route_prior_base_pred,
                route_prior_blend_weight=route_prior_blend_weight,
                counterfactual_direction_residual_bank=counterfactual_direction_residual_bank,
                counterfactual_curvature_residual_bank=counterfactual_curvature_residual_bank,
            )
            for key_name, value in hybrid_stats.items():
                if str(key_name).startswith("_"):
                    continue
                hybrid_selection_stats[key_name].append(value)
            for aux_key, value in dict(
                hybrid_stats.get("_forward_reference_l2t_diagnostics", {})
            ).items():
                forward_reference_l2t_stats[aux_key].append(value)
            for aux_key, value in dict(
                hybrid_stats.get("_forward_reference_inward_diagnostics", {})
            ).items():
                forward_reference_inward_stats[aux_key].append(value)
            if collect_oracle_candidate_diagnostics:
                _record_selectable_candidate_diagnostics(
                    target_effects=list(hybrid_stats.get("_candidate_target_effects", [])),
                    raw_collateral=list(hybrid_stats.get("_candidate_raw_collateral", [])),
                    independent_collateral=list(
                        hybrid_stats.get("_candidate_independent_collateral", [])
                    ),
                    selected_idx=int(hybrid_stats.get("_selected_idx", -1)),
                    transition_key=transition_key,
                    success_margin_z=float(success_margin_z),
                    independent_margin=float(oracle_candidate_independent_margin),
                    raw_margin=float(oracle_candidate_raw_margin),
                    selectable_candidate_stats=selectable_candidate_stats,
                    selectable_candidate_by_transition=selectable_candidate_by_transition,
                )
            if collect_case_records:
                hybrid_targets = list(hybrid_stats.get("_candidate_target_effects", []))
                hybrid_token_targets = list(hybrid_stats.get("_candidate_token_target_effects", []))
                hybrid_reference_targets = list(hybrid_stats.get("_candidate_reference_target_effects", []))
                hybrid_raw = list(hybrid_stats.get("_candidate_raw_collateral", []))
                hybrid_independent = list(hybrid_stats.get("_candidate_independent_collateral", []))
                hybrid_logits = list(hybrid_stats.get("_candidate_logits", []))
                hybrid_logit_scores = list(hybrid_stats.get("_candidate_logit_scores", []))
                hybrid_costs = list(hybrid_stats.get("_candidate_costs", []))
                hybrid_instruction_scores = list(hybrid_stats.get("_candidate_instruction_scores", []))
                hybrid_selector_feasible = list(hybrid_stats.get("_candidate_selector_feasible", []))
                hybrid_selector_raw_feasible = list(hybrid_stats.get("_candidate_selector_raw_feasible", []))
                hybrid_adjusted_targets = list(hybrid_stats.get("_candidate_adjusted_targets", []))
                hybrid_control_family = list(hybrid_stats.get("_candidate_control_family_mask", []))
                hybrid_proposal_families = list(hybrid_stats.get("_candidate_proposal_families", []))
                hybrid_family_modes = list(hybrid_stats.get("_candidate_family_modes", []))
                selected_idx = int(hybrid_stats.get("_selected_idx", -1))
                for mode_idx, target_effect_i in enumerate(hybrid_targets):
                    raw_i = (
                        float(hybrid_raw[mode_idx])
                        if mode_idx < len(hybrid_raw) and np.isfinite(float(hybrid_raw[mode_idx]))
                        else float("nan")
                    )
                    independent_i = (
                        float(hybrid_independent[mode_idx])
                        if mode_idx < len(hybrid_independent) and np.isfinite(float(hybrid_independent[mode_idx]))
                        else float("nan")
                    )
                    logit_i = (
                        float(hybrid_logits[mode_idx])
                        if mode_idx < len(hybrid_logits) and np.isfinite(float(hybrid_logits[mode_idx]))
                        else float("nan")
                    )
                    logit_score_i = (
                        float(hybrid_logit_scores[mode_idx])
                        if mode_idx < len(hybrid_logit_scores)
                        else float("nan")
                    )
                    cost_i = (
                        float(hybrid_costs[mode_idx])
                        if mode_idx < len(hybrid_costs) and np.isfinite(float(hybrid_costs[mode_idx]))
                        else float("nan")
                    )
                    instruction_score_i = (
                        float(hybrid_instruction_scores[mode_idx])
                        if mode_idx < len(hybrid_instruction_scores)
                        and np.isfinite(float(hybrid_instruction_scores[mode_idx]))
                        else float("nan")
                    )
                    adjusted_target_i = (
                        float(hybrid_adjusted_targets[mode_idx])
                        if mode_idx < len(hybrid_adjusted_targets)
                        and np.isfinite(float(hybrid_adjusted_targets[mode_idx]))
                        else float("nan")
                    )
                    selector_feasible_i = (
                        bool(hybrid_selector_feasible[mode_idx])
                        if mode_idx < len(hybrid_selector_feasible)
                        else False
                    )
                    selector_raw_feasible_i = (
                        bool(hybrid_selector_raw_feasible[mode_idx])
                        if mode_idx < len(hybrid_selector_raw_feasible)
                        else False
                    )
                    control_family_i = (
                        bool(float(hybrid_control_family[mode_idx]) > 0.5)
                        if mode_idx < len(hybrid_control_family)
                        else False
                    )
                    candidate_records.append(
                        {
                            "sample_id": int(original.get("sample_id", -1)),
                            "counterfactual_id": int(
                                counterfactual.get(
                                    "counterfactual_id",
                                    counterfactual.get("sample_id", -1),
                                )
                            ),
                            "transition": transition_key,
                            "changed_field": field_name,
                            "source_token": source_token,
                            "target_token": target_token,
                            "selection_policy": "hybrid_counterfactual",
                            "mode": int(mode_idx),
                            "selected": bool(mode_idx == selected_idx),
                            "cf_selected_index": int(selected_idx),
                            "control_family": control_family_i,
                            "proposal_family": (
                                str(hybrid_proposal_families[mode_idx])
                                if mode_idx < len(hybrid_proposal_families)
                                else None
                            ),
                            "family_mode": (
                                int(hybrid_family_modes[mode_idx])
                                if mode_idx < len(hybrid_family_modes)
                                else int(mode_idx)
                            ),
                            "logit": logit_i,
                            "logit_score": logit_score_i,
                            "selector_cost": cost_i,
                            "instruction_score": instruction_score_i,
                            "adjusted_target_z": adjusted_target_i,
                            "target_effect_z": float(target_effect_i),
                            "token_target_effect_z": (
                                float(hybrid_token_targets[mode_idx])
                                if mode_idx < len(hybrid_token_targets)
                                and np.isfinite(float(hybrid_token_targets[mode_idx]))
                                else float("nan")
                            ),
                            "reference_target_effect_z": (
                                float(hybrid_reference_targets[mode_idx])
                                if mode_idx < len(hybrid_reference_targets)
                                and np.isfinite(float(hybrid_reference_targets[mode_idx]))
                                else float("nan")
                            ),
                            "raw_collateral_z": raw_i,
                            "independent_collateral_z": independent_i,
                            "raw_feasible": bool(
                                float(target_effect_i) > float(success_margin_z)
                                and np.isfinite(raw_i)
                                and raw_i <= float(oracle_candidate_raw_margin)
                            ),
                            "independent_feasible": bool(
                                float(target_effect_i) > float(success_margin_z)
                                and np.isfinite(independent_i)
                                and independent_i <= float(oracle_candidate_independent_margin)
                            ),
                            "selector_feasible": selector_feasible_i,
                            "selector_raw_feasible": selector_raw_feasible_i,
                        }
                    )
            base_idx = None
        else:
            base_candidates = generate_candidate_geometries(
                model,
                qwen_provider,
                original,
                attr_dim,
                target_key,
                device,
            )
            base_pred_out = (base_candidates["candidates"], {"candidate_logits": base_candidates["logits"]})
            base_pred, base_idx, _ = select_prediction_with_policy(
                base_pred_out,
                reference=base_candidates["reference"],
                reference_lonlat=base_candidates["reference_lonlat"],
                route_scales=base_candidates["route_scales"],
                prior=selection_prior,
                policy=selection_policy,
                route_weight=selection_route_weight,
                token_guard_margin=selection_token_guard_margin,
                token_absolute_threshold=selection_token_absolute_threshold,
                min_route_improvement=selection_min_route_improvement,
                logit_guard_margin=selection_logit_guard_margin,
                preserve_direction=selection_preserve_direction,
                preserve_target_direction=selection_preserve_target_direction,
                mobility_tokens=tensor_record(original, target_key=target_key)["mobility_tokens"].to(device).view(1, -1),
                dataset=str(original.get("dataset", "porto") or "porto"),
            )
            if route_prior_base_pred is not None:
                base_pred = blend_route_prior_base(
                    base_pred,
                    route_prior_base_pred,
                    route_prior_blend_weight,
                    device=device,
                    dtype=base_pred.dtype,
                )
                base_idx = torch.zeros(1, device=device, dtype=torch.long)
            base_geom = tensor_dict_to_rows(
                geometry_values(
                    base_pred.float(),
                    base_candidates["reference"].float(),
                    base_candidates["route_scales"].float(),
                )
            )[0]
            base_geom = add_pace_metrics_to_rows([base_geom], original.get("trip_attributes", [{}]) if isinstance(original.get("trip_attributes", [{}]), list) else [original.get("trip_attributes", {})])[0]
            base_geom["_selected_index"] = int(base_idx.detach().view(-1)[0].item())
            base_tokens = tensor_record(original, target_key=target_key)["mobility_tokens"].to(device).view(1, -1)
            if collect_oracle_candidate_diagnostics:
                cf_candidates_for_oracle = generate_candidate_geometries_with_residual_banks(
                    model,
                    qwen_provider,
                    counterfactual,
                    attr_dim,
                    target_key,
                    device,
                    direction_residual_bank=counterfactual_direction_residual_bank,
                    curvature_residual_bank=counterfactual_curvature_residual_bank,
                    counterfactual_adapter_scale=float(counterfactual_adapter_scale),
                    base_mobility_tokens=base_tokens,
                )
                num_oracle_modes = min(len(base_candidates["geoms"]), len(cf_candidates_for_oracle["geoms"]))
                if (
                    counterfactual_direction_residual_bank
                    and field_name == "direction"
                    and len(cf_candidates_for_oracle["geoms"]) != len(base_candidates["geoms"])
                ) or (
                    counterfactual_curvature_residual_bank
                    and field_name == "curvature"
                    and len(cf_candidates_for_oracle["geoms"]) != len(base_candidates["geoms"])
                ):
                    num_oracle_modes = 0
                candidate_target_effects: List[float] = []
                candidate_raw_collateral: List[float] = []
                candidate_independent_collateral: List[float] = []
                selected_base_candidate_target_effects: List[float] = []
                selected_base_candidate_raw_collateral: List[float] = []
                selected_base_candidate_independent_collateral: List[float] = []
                selected_base_target_score = calibrator.score_token(field_name, target_token, base_geom)
                for mode_idx in range(num_oracle_modes):
                    oracle_base_geom = base_candidates["geoms"][mode_idx]
                    oracle_cf_geom = cf_candidates_for_oracle["geoms"][mode_idx]
                    base_target_score_i = calibrator.score_token(field_name, target_token, oracle_base_geom)
                    cf_target_score_i = calibrator.score_token(field_name, target_token, oracle_cf_geom)
                    target_effect_i = float(base_target_score_i - cf_target_score_i)
                    drift_by_field_i = collateral_drift_z_by_field(
                        calibrator,
                        field_name,
                        original.get("labels", {}),
                        oracle_base_geom,
                        oracle_cf_geom,
                    )
                    raw_i = float(np.mean(list(drift_by_field_i.values()))) if drift_by_field_i else float("nan")
                    independent_by_field_i = independent_collateral_fields(field_name, drift_by_field_i)
                    independent_i = _safe_collateral_mean(independent_by_field_i)
                    candidate_target_effects.append(target_effect_i)
                    candidate_raw_collateral.append(raw_i)
                    candidate_independent_collateral.append(independent_i)
                    selected_base_target_effect_i = float(selected_base_target_score - cf_target_score_i)
                    selected_base_drift_by_field_i = collateral_drift_z_by_field(
                        calibrator,
                        field_name,
                        original.get("labels", {}),
                        base_geom,
                        oracle_cf_geom,
                    )
                    selected_base_raw_i = (
                        float(np.mean(list(selected_base_drift_by_field_i.values())))
                        if selected_base_drift_by_field_i
                        else float("nan")
                    )
                    selected_base_independent_by_field_i = independent_collateral_fields(
                        field_name,
                        selected_base_drift_by_field_i,
                    )
                    selected_base_independent_i = _safe_collateral_mean(selected_base_independent_by_field_i)
                    selected_base_candidate_target_effects.append(selected_base_target_effect_i)
                    selected_base_candidate_raw_collateral.append(selected_base_raw_i)
                    selected_base_candidate_independent_collateral.append(selected_base_independent_i)
                target_arr = np.asarray(candidate_target_effects, dtype=np.float64)
                raw_arr = np.asarray(candidate_raw_collateral, dtype=np.float64)
                independent_arr = np.asarray(candidate_independent_collateral, dtype=np.float64)
                selected_base_target_arr = np.asarray(selected_base_candidate_target_effects, dtype=np.float64)
                selected_base_raw_arr = np.asarray(selected_base_candidate_raw_collateral, dtype=np.float64)
                selected_base_independent_arr = np.asarray(
                    selected_base_candidate_independent_collateral,
                    dtype=np.float64,
                )
                success_mask = target_arr > float(success_margin_z)
                independent_feasible = success_mask & np.isfinite(independent_arr) & (
                    independent_arr <= float(oracle_candidate_independent_margin)
                )
                raw_feasible = success_mask & np.isfinite(raw_arr) & (
                    raw_arr <= float(oracle_candidate_raw_margin)
                )
                joint_feasible = independent_feasible & raw_feasible
                selected_base_success_mask = selected_base_target_arr > float(success_margin_z)
                selected_base_independent_feasible = (
                    selected_base_success_mask
                    & np.isfinite(selected_base_independent_arr)
                    & (selected_base_independent_arr <= float(oracle_candidate_independent_margin))
                )
                selected_base_raw_feasible = (
                    selected_base_success_mask
                    & np.isfinite(selected_base_raw_arr)
                    & (selected_base_raw_arr <= float(oracle_candidate_raw_margin))
                )
                selected_base_joint_feasible = selected_base_independent_feasible & selected_base_raw_feasible
                if target_arr.size:
                    oracle_values = {
                        "num_modes": float(target_arr.size),
                        "best_target_effect": float(np.nanmax(target_arr)),
                        "best_success": float(np.nanmax(target_arr) > float(success_margin_z)),
                        "min_raw_collateral": float(np.nanmin(raw_arr)),
                        "min_independent_collateral": float(np.nanmin(independent_arr)),
                        "any_independent_feasible": float(independent_feasible.any()),
                        "any_raw_feasible": float(raw_feasible.any()),
                        "any_joint_feasible": float(joint_feasible.any()),
                        "best_target_under_independent_margin": (
                            float(np.nanmax(np.where(independent_feasible, target_arr, np.nan)))
                            if bool(independent_feasible.any())
                            else float("nan")
                        ),
                        "best_target_under_raw_margin": (
                            float(np.nanmax(np.where(raw_feasible, target_arr, np.nan)))
                            if bool(raw_feasible.any())
                            else float("nan")
                        ),
                        "best_target_under_joint_margin": (
                            float(np.nanmax(np.where(joint_feasible, target_arr, np.nan)))
                            if bool(joint_feasible.any())
                            else float("nan")
                        ),
                        "raw_at_best_target": float(raw_arr[int(np.nanargmax(target_arr))]),
                        "independent_at_best_target": float(independent_arr[int(np.nanargmax(target_arr))]),
                        "selected_base_best_target_effect": float(np.nanmax(selected_base_target_arr)),
                        "selected_base_best_success": float(
                            np.nanmax(selected_base_target_arr) > float(success_margin_z)
                        ),
                        "selected_base_min_raw_collateral": float(np.nanmin(selected_base_raw_arr)),
                        "selected_base_min_independent_collateral": float(
                            np.nanmin(selected_base_independent_arr)
                        ),
                        "selected_base_any_independent_feasible": float(
                            selected_base_independent_feasible.any()
                        ),
                        "selected_base_any_raw_feasible": float(selected_base_raw_feasible.any()),
                        "selected_base_any_joint_feasible": float(selected_base_joint_feasible.any()),
                        "selected_base_best_target_under_independent_margin": (
                            float(np.nanmax(np.where(selected_base_independent_feasible, selected_base_target_arr, np.nan)))
                            if bool(selected_base_independent_feasible.any())
                            else float("nan")
                        ),
                        "selected_base_best_target_under_raw_margin": (
                            float(np.nanmax(np.where(selected_base_raw_feasible, selected_base_target_arr, np.nan)))
                            if bool(selected_base_raw_feasible.any())
                            else float("nan")
                        ),
                        "selected_base_best_target_under_joint_margin": (
                            float(np.nanmax(np.where(selected_base_joint_feasible, selected_base_target_arr, np.nan)))
                            if bool(selected_base_joint_feasible.any())
                            else float("nan")
                        ),
                        "selected_base_raw_at_best_target": float(
                            selected_base_raw_arr[int(np.nanargmax(selected_base_target_arr))]
                        ),
                        "selected_base_independent_at_best_target": float(
                            selected_base_independent_arr[int(np.nanargmax(selected_base_target_arr))]
                        ),
                    }
                    key_for_oracle = f"{field_name}:{source_token}->{target_token}"
                    for oracle_key, oracle_value in oracle_values.items():
                        oracle_candidate_stats[oracle_key].append(oracle_value)
                        oracle_candidate_by_transition[key_for_oracle][oracle_key].append(oracle_value)
            cf_geom, cf_candidates = generate_geometry(
                model,
                qwen_provider,
                counterfactual,
                attr_dim,
                target_key,
                device,
                counterfactual_adapter_scale=float(counterfactual_adapter_scale),
                selection_prior=selection_prior,
                selection_policy=selection_policy,
                selection_route_weight=selection_route_weight,
                selection_token_guard_margin=selection_token_guard_margin,
                selection_token_absolute_threshold=selection_token_absolute_threshold,
                selection_min_route_improvement=selection_min_route_improvement,
                selection_logit_guard_margin=selection_logit_guard_margin,
                selection_preserve_direction=selection_preserve_direction,
                selection_preserve_target_direction=selection_preserve_target_direction,
                base_selected_trajectory=base_pred,
                base_mobility_tokens=base_tokens,
                base_selected_index=base_idx,
                counterfactual_direction_residual_bank=counterfactual_direction_residual_bank,
                counterfactual_curvature_residual_bank=counterfactual_curvature_residual_bank,
                return_candidates=True,
            )
            cf_aux = cf_candidates.get("aux", {}) if isinstance(cf_candidates, dict) else {}
            if isinstance(cf_aux, dict):
                forward_aux_stats = counterfactual_candidate_pool_guard_aux_diagnostics(cf_aux)
                for aux_key, value in forward_aux_stats.items():
                    if "_reference_inward_" in aux_key:
                        forward_reference_inward_stats[aux_key].append(value)
                    else:
                        forward_reference_l2t_stats[aux_key].append(value)
            if collect_oracle_candidate_diagnostics:
                cf_selected_index = int(cf_geom.get("_selected_index", -1))
                selected_base_target_score = calibrator.score_token(field_name, target_token, base_geom)
                selectable_target_effects: List[float] = []
                selectable_raw_collateral: List[float] = []
                selectable_independent_collateral: List[float] = []
                selectable_control_mask: List[bool] = []
                control_mask_tensor = None
                if isinstance(cf_aux, dict):
                    control_mask_tensor = cf_aux.get("candidate_control_family_mask")
                if torch.is_tensor(control_mask_tensor):
                    control_mask_values = (
                        control_mask_tensor.detach().cpu().float().reshape(-1).numpy().astype(np.float64)
                    )
                else:
                    control_mask_values = np.zeros((len(cf_candidates.get("geoms", [])),), dtype=np.float64)
                for mode_idx, candidate_geom in enumerate(cf_candidates.get("geoms", [])):
                    candidate_target_score = calibrator.score_token(field_name, target_token, candidate_geom)
                    target_effect_i = float(selected_base_target_score - candidate_target_score)
                    drift_by_field_i = collateral_drift_z_by_field(
                        calibrator,
                        field_name,
                        original.get("labels", {}),
                        base_geom,
                        candidate_geom,
                    )
                    raw_i = (
                        float(np.mean(list(drift_by_field_i.values())))
                        if drift_by_field_i
                        else float("nan")
                    )
                    independent_i = _safe_collateral_mean(
                        independent_collateral_fields(field_name, drift_by_field_i)
                    )
                    selectable_target_effects.append(target_effect_i)
                    selectable_raw_collateral.append(raw_i)
                    selectable_independent_collateral.append(independent_i)
                    selectable_control_mask.append(
                        bool(mode_idx < len(control_mask_values) and control_mask_values[mode_idx] > 0.5)
                    )
                _record_selectable_candidate_diagnostics(
                    target_effects=selectable_target_effects,
                    raw_collateral=selectable_raw_collateral,
                    independent_collateral=selectable_independent_collateral,
                    selected_idx=cf_selected_index,
                    transition_key=transition_key,
                    success_margin_z=float(success_margin_z),
                    independent_margin=float(oracle_candidate_independent_margin),
                    raw_margin=float(oracle_candidate_raw_margin),
                    selectable_candidate_stats=selectable_candidate_stats,
                    selectable_candidate_by_transition=selectable_candidate_by_transition,
                    control_mask=selectable_control_mask,
                )
            if collect_case_records:
                cf_selected_index = int(cf_geom.get("_selected_index", -1))
                proposal_families = list(cf_candidates.get("proposal_family_by_candidate", []))
                family_modes = list(cf_candidates.get("family_mode_by_candidate", []))
                cf_logits = cf_candidates.get("logits")
                cf_tensor = cf_candidates.get("candidates")
                route_scale = cf_candidates.get("route_scales")
                control_mask_tensor = cf_aux.get("candidate_control_family_mask") if isinstance(cf_aux, dict) else None
                if torch.is_tensor(control_mask_tensor):
                    control_mask_values = (
                        control_mask_tensor.detach().cpu().float().reshape(-1).numpy().astype(np.float64)
                    )
                else:
                    control_mask_values = np.zeros((len(cf_candidates.get("geoms", [])),), dtype=np.float64)
                selected_base_target_score = calibrator.score_token(field_name, target_token, base_geom)
                num_candidate_rows = int(len(cf_candidates.get("geoms", [])))
                for mode_idx in range(num_candidate_rows):
                    candidate_geom = cf_candidates["geoms"][mode_idx]
                    candidate_target_score = calibrator.score_token(field_name, target_token, candidate_geom)
                    candidate_target_effect, candidate_token_effect, candidate_reference_effect = (
                        _selection_target_effect_z(
                            calibrator,
                            field_name,
                            source_token,
                            target_token,
                            base_geom,
                            candidate_geom,
                        )
                    )
                    candidate_drift_by_field = collateral_drift_z_by_field(
                        calibrator,
                        field_name,
                        original.get("labels", {}),
                        base_geom,
                        candidate_geom,
                    )
                    candidate_raw_collateral = (
                        float(np.mean(list(candidate_drift_by_field.values())))
                        if candidate_drift_by_field
                        else float("nan")
                    )
                    candidate_independent = _safe_collateral_mean(
                        independent_collateral_fields(field_name, candidate_drift_by_field)
                    )
                    base_ref_distance = float(base_geom.get("mean_reference_distance_norm", float("nan")))
                    candidate_ref_distance = float(
                        candidate_geom.get("mean_reference_distance_norm", float("nan"))
                    )
                    step_norm_norm, step_norm_m = _candidate_step_norms(
                        base_pred,
                        cf_tensor,
                        route_scale,
                        mode_idx,
                    )
                    candidate_records.append(
                        {
                            "sample_id": int(original.get("sample_id", -1)),
                            "counterfactual_id": int(
                                counterfactual.get(
                                    "counterfactual_id",
                                    counterfactual.get("sample_id", -1),
                                )
                            ),
                            "transition": transition_key,
                            "changed_field": field_name,
                            "source_token": source_token,
                            "target_token": target_token,
                            "mode": int(mode_idx),
                            "selected": bool(mode_idx == cf_selected_index),
                            "base_selected_index": int(base_geom.get("_selected_index", -1)),
                            "cf_selected_index": int(cf_selected_index),
                            "proposal_family": (
                                str(proposal_families[mode_idx])
                                if mode_idx < len(proposal_families)
                                else None
                            ),
                            "control_family": bool(
                                mode_idx < len(control_mask_values) and control_mask_values[mode_idx] > 0.5
                            ),
                            "family_mode": (
                                int(family_modes[mode_idx])
                                if mode_idx < len(family_modes)
                                else int(mode_idx)
                            ),
                            "logit": _candidate_logit(cf_logits, mode_idx),
                            "target_effect_z": candidate_target_effect,
                            "token_target_effect_z": candidate_token_effect,
                            "reference_target_effect_z": candidate_reference_effect,
                            "target_token_score": float(candidate_target_score),
                            "raw_collateral_z": candidate_raw_collateral,
                            "independent_collateral_z": candidate_independent,
                            "base_mean_reference_distance_norm": base_ref_distance,
                            "candidate_mean_reference_distance_norm": candidate_ref_distance,
                            "reference_distance_delta_norm": (
                                float(candidate_ref_distance - base_ref_distance)
                                if np.isfinite(candidate_ref_distance) and np.isfinite(base_ref_distance)
                                else float("nan")
                            ),
                            "reference_distance_improvement_norm": (
                                float(base_ref_distance - candidate_ref_distance)
                                if np.isfinite(candidate_ref_distance) and np.isfinite(base_ref_distance)
                                else float("nan")
                            ),
                            "step_vector_norm_mean_norm": step_norm_norm,
                            "step_vector_norm_mean_m": step_norm_m,
                            "candidate_length_ratio": float(candidate_geom.get("length_ratio", float("nan"))),
                            "candidate_local_total_abs_turn_rad": float(
                                candidate_geom.get("local_total_abs_turn_rad", float("nan"))
                            ),
                            "candidate_avg_speed_kmh": float(candidate_geom.get("avg_speed_kmh", float("nan"))),
                        }
                    )
        target_effect, token_target_effect, reference_target_effect = _selection_target_effect_z(
            calibrator,
            field_name,
            source_token,
            target_token,
            base_geom,
            cf_geom,
        )
        success = float(target_effect > float(success_margin_z))
        drift_by_field = collateral_drift_z_by_field(
            calibrator,
            field_name,
            original.get("labels", {}),
            base_geom,
            cf_geom,
        )
        drift = float(np.mean(list(drift_by_field.values()))) if drift_by_field else float("nan")
        independent_drift_by_field = independent_collateral_fields(field_name, drift_by_field)
        independent_drift = _safe_collateral_mean(independent_drift_by_field)
        independent_defined = bool(independent_drift_by_field)

        (
            metric_name,
            base_to_target_delta,
            observed_delta,
            signed_metric_delta,
            signed_metric_delta_z,
        ) = calibrator.transition_metric_delta(field_name, source_token, target_token, base_geom, cf_geom)

        key = f"{field_name}:{source_token}->{target_token}"
        for bucket in (by_field[field_name], by_transition[key]):
            bucket["target_effect_z"].append(target_effect)
            bucket["success"].append(success)
            bucket["collateral_drift_z"].append(drift)
            bucket["independent_collateral_drift_z"].append(independent_drift)
            bucket["metric_delta"].append(signed_metric_delta)
            bucket["metric_delta_z"].append(signed_metric_delta_z)
        for collateral_field, collateral_value in drift_by_field.items():
            collateral_by_non_target[collateral_field].append(collateral_value)
            by_field[field_name]["collateral_by_non_target"][collateral_field].append(collateral_value)
            by_transition[key]["collateral_by_non_target"][collateral_field].append(collateral_value)
        for collateral_field, collateral_value in independent_drift_by_field.items():
            independent_collateral_by_non_target[collateral_field].append(collateral_value)
            by_field[field_name]["independent_collateral_by_non_target"][collateral_field].append(collateral_value)
            by_transition[key]["independent_collateral_by_non_target"][collateral_field].append(collateral_value)
        target_effects.append(target_effect)
        success_flags.append(success)
        collateral_values.append(drift)
        independent_collateral_values.append(independent_drift)
        independent_collateral_defined_flags.append(1.0 if independent_defined else 0.0)
        metric_deltas.append(signed_metric_delta)
        metric_delta_z_values.append(signed_metric_delta_z)
        if collect_case_records:
            visual_arrays = hybrid_stats.get("_visual_arrays") if selection_policy == "hybrid_counterfactual" else None
            max_raw_field = None
            max_raw_value = float("-inf")
            for drift_field, drift_value in drift_by_field.items():
                if np.isfinite(float(drift_value)) and float(drift_value) > max_raw_value:
                    max_raw_field = drift_field
                    max_raw_value = float(drift_value)
            case_records.append(
                {
                    "sample_id": int(original.get("sample_id", -1)),
                    "counterfactual_id": int(counterfactual.get("counterfactual_id", counterfactual.get("sample_id", -1))),
                    "transition": key,
                    "changed_field": field_name,
                    "source_token": source_token,
                    "target_token": target_token,
                    "target_metric": metric_name,
                    "target_effect_z": target_effect,
                    "success": bool(success),
                    "signed_metric_delta": float(signed_metric_delta),
                    "signed_metric_delta_z": float(signed_metric_delta_z),
                    "raw_collateral_z": drift,
                    "independent_collateral_z": independent_drift,
                    "max_raw_collateral_field": max_raw_field,
                    "max_raw_collateral_z": max_raw_value if max_raw_field is not None else float("nan"),
                    "collateral_by_non_target": {
                        drift_field: float(drift_value)
                        for drift_field, drift_value in sorted(drift_by_field.items())
                    },
                    "independent_collateral_by_non_target": {
                        drift_field: float(drift_value)
                        for drift_field, drift_value in sorted(independent_drift_by_field.items())
                    },
                    "base_selected_index": (
                        int(base_geom["_selected_index"])
                        if "_selected_index" in base_geom
                        else None
                    ),
                    "cf_selected_index": (
                        int(cf_geom["_selected_index"])
                        if "_selected_index" in cf_geom
                        else None
                    ),
                    "base_length_ratio": float(base_geom.get("length_ratio", float("nan"))),
                    "cf_length_ratio": float(cf_geom.get("length_ratio", float("nan"))),
                    "base_mean_reference_distance_norm": float(
                        base_geom.get("mean_reference_distance_norm", float("nan"))
                    ),
                    "cf_mean_reference_distance_norm": float(
                        cf_geom.get("mean_reference_distance_norm", float("nan"))
                    ),
                    "base_local_total_abs_turn_rad": float(
                        base_geom.get("local_total_abs_turn_rad", float("nan"))
                    ),
                    "cf_local_total_abs_turn_rad": float(
                        cf_geom.get("local_total_abs_turn_rad", float("nan"))
                    ),
                    "base_avg_speed_kmh": float(base_geom.get("avg_speed_kmh", float("nan"))),
                    "cf_avg_speed_kmh": float(cf_geom.get("avg_speed_kmh", float("nan"))),
                    "base_instruction": str(original.get("instruction", "")),
                    "cf_instruction": str(counterfactual.get("instruction", "")),
                    "_visual_arrays": visual_arrays,
                }
            )

    def _summarize_bucket(values: Dict[str, List[float]]) -> Dict:
        return {
            "counterfactual_target_effect_z": summarize(values["target_effect_z"]),
            "counterfactual_success_rate": float(np.mean(values["success"])) if values["success"] else float("nan"),
            "collateral_drift_z": summarize(values["collateral_drift_z"]),
            "independent_collateral_drift_z": summarize(values["independent_collateral_drift_z"]),
            "collateral_by_non_target": {
                field: summarize(field_values)
                for field, field_values in sorted(values["collateral_by_non_target"].items())
            },
            "independent_collateral_by_non_target": {
                field: summarize(field_values)
                for field, field_values in sorted(values["independent_collateral_by_non_target"].items())
            },
            "signed_target_metric_delta": summarize(values["metric_delta"]),
            "signed_target_metric_delta_z": summarize(values["metric_delta_z"]),
            "num_pairs": int(len(values["success"])),
        }

    result = {
        "num_pairs": int(len(success_flags)),
        "table4_counterfactual_controllability": {
            "counterfactual_target_effect": float(np.mean(target_effects)) if target_effects else float("nan"),
            "counterfactual_success_rate": float(np.mean(success_flags)) if success_flags else float("nan"),
            "collateral_drift": float(np.nanmean(collateral_values)) if collateral_values else float("nan"),
            "raw_collateral_drift": float(np.nanmean(collateral_values)) if collateral_values else float("nan"),
            "independent_collateral_drift": float(np.nanmean(independent_collateral_values))
            if independent_collateral_values
            else float("nan"),
        },
        "counterfactual_target_effect_z": summarize(target_effects),
        "counterfactual_success_rate": float(np.mean(success_flags)) if success_flags else float("nan"),
        "collateral_drift_z": summarize(collateral_values),
        "raw_collateral_drift_z": summarize(collateral_values),
        "independent_collateral_drift_z": summarize(independent_collateral_values),
        "collateral_by_non_target": {
            field: summarize(values) for field, values in sorted(collateral_by_non_target.items())
        },
        "independent_collateral_by_non_target": {
            field: summarize(values) for field, values in sorted(independent_collateral_by_non_target.items())
        },
        "independent_collateral_defined_rate": (
            float(np.mean(independent_collateral_defined_flags))
            if independent_collateral_defined_flags
            else float("nan")
        ),
        "independent_collateral_empty_scope_pairs": int(
            sum(1 for flag in independent_collateral_defined_flags if flag <= 0.0)
        ),
        "independent_collateral_exclusions": serializable_collateral_exclusions(),
        "signed_target_metric_delta": summarize(metric_deltas),
        "signed_target_metric_delta_z": summarize(metric_delta_z_values),
        "route_prior": {
            "enabled": bool(route_prior_lookup),
            "used_pairs": int(route_prior_used),
            "missing_pairs": int(route_prior_missing),
            "required": bool(route_prior_required),
        },
        "by_field": {field: _summarize_bucket(values) for field, values in sorted(by_field.items())},
        "by_transition": {key: _summarize_bucket(values) for key, values in sorted(by_transition.items())},
    }
    if forward_reference_l2t_stats:
        result["forward_reference_l2t_diagnostics"] = {
            key: summarize(values) for key, values in sorted(forward_reference_l2t_stats.items())
        }
    if forward_reference_inward_stats:
        result["forward_reference_inward_diagnostics"] = {
            key: summarize(values) for key, values in sorted(forward_reference_inward_stats.items())
        }
    if hybrid_selection_stats:
        result["hybrid_selection"] = {
            key: summarize(values) for key, values in sorted(hybrid_selection_stats.items())
        }
        modes = [int(value) for value in hybrid_selection_stats.get("hybrid_selected_mode", [])]
        result["hybrid_selection"]["selected_mode_counts"] = {
            str(mode): int(modes.count(mode)) for mode in sorted(set(modes))
        }
    if oracle_candidate_stats:
        result["oracle_candidate_diagnostics"] = {
            "enabled": True,
            "independent_margin": float(oracle_candidate_independent_margin),
            "raw_margin": float(oracle_candidate_raw_margin),
            "summary": {
                key: summarize(values) for key, values in sorted(oracle_candidate_stats.items())
            },
            "by_transition": {
                key: {
                    metric_key: summarize(metric_values)
                    for metric_key, metric_values in sorted(values.items())
                }
                for key, values in sorted(oracle_candidate_by_transition.items())
            },
        }
    if selectable_candidate_stats:
        result["selectable_candidate_diagnostics"] = {
            "enabled": True,
            "independent_margin": float(oracle_candidate_independent_margin),
            "raw_margin": float(oracle_candidate_raw_margin),
            "summary": {
                key: summarize(values) for key, values in sorted(selectable_candidate_stats.items())
            },
            "by_transition": {
                key: {
                    metric_key: summarize(metric_values)
                    for metric_key, metric_values in sorted(values.items())
                }
                for key, values in sorted(selectable_candidate_by_transition.items())
            },
        }
    if collect_case_records:
        result["case_records"] = sorted(
            case_records,
            key=lambda item: (
                -float(item.get("raw_collateral_z", float("-inf")))
                if np.isfinite(float(item.get("raw_collateral_z", float("nan"))))
                else float("inf")
            ),
        )
        result["candidate_records"] = candidate_records
    return result


def main() -> None:
    global _ROAD_CORRIDOR_LOOKUP
    args = parse_args()
    log_stage(f"start device={args.device} max_pairs={args.max_pairs} route_prior={args.route_prior_npz is not None}")
    device = resolve_device(args.device)
    log_stage(f"resolved device={device}")
    log_stage(f"load checkpoint {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device)
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    args.selection_policy = str(
        args.selection_policy
        or saved_args.get("table4_selection_policy", "reranker")
    )
    args.selection_hybrid_instruction_weight = float(
        args.selection_hybrid_instruction_weight
        if args.selection_hybrid_instruction_weight is not None
        else saved_args.get("table4_selection_hybrid_instruction_weight", 0.5)
    )
    args.selection_hybrid_target_effect_weight = float(
        args.selection_hybrid_target_effect_weight
        if args.selection_hybrid_target_effect_weight is not None
        else saved_args.get("table4_selection_hybrid_target_effect_weight", 1.0)
    )
    args.selection_hybrid_collateral_weight = float(
        args.selection_hybrid_collateral_weight
        if args.selection_hybrid_collateral_weight is not None
        else saved_args.get("table4_selection_hybrid_collateral_weight", 0.5)
    )
    args.selection_hybrid_collateral_margin = float(
        args.selection_hybrid_collateral_margin
        if args.selection_hybrid_collateral_margin is not None
        else saved_args.get("table4_selection_hybrid_collateral_margin", 0.08)
    )
    args.selection_hybrid_raw_collateral_margin = float(
        args.selection_hybrid_raw_collateral_margin
        if args.selection_hybrid_raw_collateral_margin is not None
        else saved_args.get("table4_selection_hybrid_raw_collateral_margin", 0.15)
    )
    args.selection_hybrid_logit_weight = float(
        args.selection_hybrid_logit_weight
        if args.selection_hybrid_logit_weight is not None
        else saved_args.get("table4_selection_hybrid_logit_weight", 0.0)
    )
    args.selection_hybrid_priority = str(
        args.selection_hybrid_priority
        or saved_args.get("table4_selection_hybrid_priority", "weighted")
    )
    log_stage("build model")
    model = build_model(
        checkpoint,
        device,
        token_calibration_path=args.token_calibration_path,
        arg_overrides={
            "qwen_plan_scale": args.qwen_plan_scale_override,
            "route_proposal_logit_weight": args.route_proposal_logit_weight_override,
            "route_control_selection_adapter_weight": args.route_control_selection_adapter_weight_override,
            "route_control_selection_adapter_clip": args.route_control_selection_adapter_clip_override,
        },
    )
    if args.counterfactual_direction_residual_type_override:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "set_counterfactual_direction_residual_type"):
            raise ValueError(
                "--counterfactual-direction-residual-type-override requires a DiT decoder "
                "with counterfactual direction residual support."
            )
        decoder.set_counterfactual_direction_residual_type(
            str(args.counterfactual_direction_residual_type_override)
        )
    if args.counterfactual_curvature_residual_type_override:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "counterfactual_curvature_residual_type"):
            raise ValueError(
                "--counterfactual-curvature-residual-type-override requires a DiT decoder "
                "with counterfactual curvature residual support."
            )
        decoder.counterfactual_curvature_residual_type = str(
            args.counterfactual_curvature_residual_type_override
        )
    if args.counterfactual_reference_residual_type_override:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "counterfactual_reference_residual_type"):
            raise ValueError(
                "--counterfactual-reference-residual-type-override requires a DiT decoder "
                "with counterfactual reference residual support."
            )
        decoder.counterfactual_reference_residual_type = str(
            args.counterfactual_reference_residual_type_override
        )
    eval_selector_overrides = apply_counterfactual_eval_selector_overrides(model, args)
    if args.counterfactual_residual_strength is not None:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "counterfactual_residual_strength"):
            raise ValueError("--counterfactual-residual-strength requires a DiT decoder with counterfactual residual support.")
        decoder.counterfactual_residual_strength = float(args.counterfactual_residual_strength)
    if args.counterfactual_route_frame_budget_mode is not None:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "counterfactual_route_frame_budget_mode"):
            raise ValueError(
                "--counterfactual-route-frame-budget-mode requires a DiT decoder with route-frame budget support."
            )
        decoder.counterfactual_route_frame_budget_mode = str(args.counterfactual_route_frame_budget_mode)
    if args.counterfactual_route_frame_tangent_budget is not None:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "counterfactual_route_frame_tangent_budget"):
            raise ValueError(
                "--counterfactual-route-frame-tangent-budget requires a DiT decoder with route-frame budget support."
            )
        decoder.counterfactual_route_frame_tangent_budget = float(args.counterfactual_route_frame_tangent_budget)
    if args.counterfactual_route_frame_normal_budget is not None:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "counterfactual_route_frame_normal_budget"):
            raise ValueError(
                "--counterfactual-route-frame-normal-budget requires a DiT decoder with route-frame budget support."
            )
        decoder.counterfactual_route_frame_normal_budget = float(args.counterfactual_route_frame_normal_budget)
    if args.counterfactual_route_frame_fields is not None:
        decoder = getattr(model, "decoder", None)
        parser_fn = getattr(decoder, "parse_route_frame_budget_fields", None) if decoder is not None else None
        if parser_fn is None:
            raise ValueError(
                "--counterfactual-route-frame-fields requires a DiT decoder with route-frame budget support."
            )
        decoder.counterfactual_route_frame_fields = parser_fn(args.counterfactual_route_frame_fields)
    if args.counterfactual_adapter_scale is None:
        args.counterfactual_adapter_scale = float(saved_args.get("counterfactual_adapter_scale", 0.0))
    if args.reference_positive_candidate_strength_override is not None:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "set_reference_positive_candidate_strength"):
            raise ValueError(
                "--reference-positive-candidate-strength-override requires a compatible DiT decoder."
            )
        decoder.set_reference_positive_candidate_strength(
            float(args.reference_positive_candidate_strength_override)
        )
    if args.reference_positive_candidate_scope_override is not None:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "set_reference_positive_candidate_scope"):
            raise ValueError(
                "--reference-positive-candidate-scope-override requires a compatible DiT decoder."
            )
        decoder.set_reference_positive_candidate_scope(
            str(args.reference_positive_candidate_scope_override)
        )
    if args.reference_distance_candidate_strength_override is not None:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "set_reference_distance_candidate_strength"):
            raise ValueError(
                "--reference-distance-candidate-strength-override requires a compatible DiT decoder."
            )
        decoder.set_reference_distance_candidate_strength(
            float(args.reference_distance_candidate_strength_override)
        )
    if args.reference_distance_candidate_scope_override is not None:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "set_reference_distance_candidate_scope"):
            raise ValueError(
                "--reference-distance-candidate-scope-override requires a compatible DiT decoder."
            )
        decoder.set_reference_distance_candidate_scope(
            str(args.reference_distance_candidate_scope_override)
        )
    if args.reference_distance_candidate_mode_override is not None:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "set_reference_distance_candidate_mode"):
            raise ValueError(
                "--reference-distance-candidate-mode-override requires a compatible DiT decoder."
            )
        decoder.set_reference_distance_candidate_mode(
            str(args.reference_distance_candidate_mode_override)
        )
    if args.counterfactual_local_proposal_type_override is not None:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "counterfactual_local_proposal_type"):
            raise ValueError(
                "--counterfactual-local-proposal-type-override requires a compatible DiT decoder."
            )
        decoder.counterfactual_local_proposal_type = str(
            args.counterfactual_local_proposal_type_override
        )
    if args.counterfactual_local_proposal_strength_override is not None:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "counterfactual_local_proposal_strength"):
            raise ValueError(
                "--counterfactual-local-proposal-strength-override requires a compatible DiT decoder."
            )
        decoder.counterfactual_local_proposal_strength = float(
            args.counterfactual_local_proposal_strength_override
        )
    if args.base_preserving_selector_enabled_override:
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "base_preserving_selector_enabled"):
            raise ValueError(
                "--base-preserving-selector-enabled-override requires a compatible DiT decoder."
            )
        decoder.base_preserving_selector_enabled = True
    attr_dim = int(saved_args.get("attr_dim", 6))
    if args.disable_qwen:
        args.qwen_planner_checkpoint = None
    elif args.qwen_planner_checkpoint is None:
        saved_qwen_checkpoint = saved_args.get("qwen_planner_checkpoint")
        if saved_qwen_checkpoint:
            args.qwen_planner_checkpoint = Path(saved_qwen_checkpoint)
    log_stage("build qwen provider" if args.qwen_planner_checkpoint else "qwen provider disabled")
    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device) if args.qwen_planner_checkpoint else None
    log_stage("load selection prior")
    selection_prior = load_selection_prior(args.selection_prior_path, device)
    log_stage("load route prior")
    route_prior_lookup = load_route_prior_lookup(args.route_prior_npz, args.route_prior_key)

    jsonl_path = args.input or (
        REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_counterfactual_instruction_samples.filtered.jsonl"
    )
    log_stage("load road corridor lookup")
    _ROAD_CORRIDOR_LOOKUP = load_road_corridor_lookup(args.road_corridor_cache)
    log_stage(f"read records {jsonl_path}")
    if int(args.source_num_shards) > 1:
        log_stage(
            f"source shard {int(args.source_shard_index) + 1}/{int(args.source_num_shards)} "
            "(by source_sample_id)"
        )
    records = read_records(
        jsonl_path,
        max_records=args.max_records,
        source_num_shards=int(args.source_num_shards),
        source_shard_index=int(args.source_shard_index),
    )
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
    log_stage(f"grouped pairs={len(pairs)}")
    if not pairs:
        raise ValueError(
            "No original/counterfactual pairs found. Increase --max-records or use an input containing original records."
        )
    if args.token_calibration_path is not None and args.token_calibration_path.exists():
        log_stage(f"load token calibration {args.token_calibration_path}")
        calibrator = GeometryTokenCalibrator.from_json(str(args.token_calibration_path))
        calibration_source = str(args.token_calibration_path)
    else:
        log_stage("build pair calibrator")
        calibrator = build_pair_calibrator(pairs, args.target_key, model.decoder.target_len, device)
        calibration_source = "paired target trajectories"
    log_stage("evaluate pairs")
    metrics = evaluate_pairs(
        model=model,
        qwen_provider=qwen_provider,
        pairs=pairs,
        attr_dim=attr_dim,
        target_key=args.target_key,
        calibrator=calibrator,
        device=device,
        success_margin_z=args.success_margin_z,
        counterfactual_adapter_scale=args.counterfactual_adapter_scale,
        selection_prior=selection_prior,
        selection_policy=args.selection_policy,
        selection_route_weight=args.selection_route_weight,
        selection_token_guard_margin=args.selection_token_guard_margin,
        selection_token_absolute_threshold=args.selection_token_absolute_threshold,
        selection_min_route_improvement=args.selection_min_route_improvement,
        selection_logit_guard_margin=args.selection_logit_guard_margin,
        selection_preserve_direction=args.selection_preserve_direction,
        selection_preserve_target_direction=args.selection_preserve_target_direction,
        selection_hybrid_instruction_weight=args.selection_hybrid_instruction_weight,
        selection_hybrid_target_effect_weight=args.selection_hybrid_target_effect_weight,
        selection_hybrid_collateral_weight=args.selection_hybrid_collateral_weight,
        selection_hybrid_collateral_margin=args.selection_hybrid_collateral_margin,
        selection_hybrid_raw_collateral_margin=args.selection_hybrid_raw_collateral_margin,
        selection_hybrid_logit_weight=args.selection_hybrid_logit_weight,
        selection_hybrid_priority=args.selection_hybrid_priority,
        collect_case_records=(
            args.export_failure_cases is not None
            or args.export_candidate_rows is not None
            or args.export_visual_cases is not None
        ),
        collect_oracle_candidate_diagnostics=bool(args.oracle_candidate_diagnostics),
        oracle_candidate_independent_margin=float(args.oracle_candidate_independent_margin),
        oracle_candidate_raw_margin=float(args.oracle_candidate_raw_margin),
        counterfactual_direction_residual_bank=args.counterfactual_direction_residual_bank,
        counterfactual_curvature_residual_bank=args.counterfactual_curvature_residual_bank,
        route_prior_lookup=route_prior_lookup,
        route_prior_required=bool(args.route_prior_required),
        route_prior_blend_weight=float(args.route_prior_blend_weight),
    )
    log_stage("write outputs")
    case_records = metrics.pop("case_records", [])
    candidate_records = metrics.pop("candidate_records", [])
    failure_case_export = None
    candidate_row_export = None
    visual_case_export = None
    if args.export_failure_cases is not None:
        failure_case_export = write_failure_cases(
            args.export_failure_cases,
            case_records,
            max_records=args.max_failure_cases,
        )
        candidate_row_export = write_candidate_rows(
            Path(args.export_failure_cases).with_name("candidate_rows.jsonl"),
            candidate_records,
        )
    if args.export_visual_cases is not None:
        visual_case_export = write_visual_cases(
            args.export_visual_cases,
            case_records,
            max_records=args.max_visual_cases,
        )
    if args.export_candidate_rows is not None:
        candidate_row_export = write_candidate_rows(args.export_candidate_rows, candidate_records)
    metrics.update(
        {
            "dataset": args.dataset,
            "input": str(jsonl_path),
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "target_key": args.target_key,
            "fields": args.fields,
            "source_shard": {
                "index": int(args.source_shard_index),
                "num_shards": int(args.source_num_shards),
                "mode": "source_sample_id_modulo",
            },
            "success_margin_z": float(args.success_margin_z),
            "counterfactual_adapter_scale": float(args.counterfactual_adapter_scale),
            "route_prior_npz": str(args.route_prior_npz) if args.route_prior_npz else None,
            "route_prior_key": str(args.route_prior_key),
            "route_prior_required": bool(args.route_prior_required),
            "route_prior_blend_weight": float(args.route_prior_blend_weight),
            "reference_positive_candidate_strength_override": (
                args.reference_positive_candidate_strength_override
            ),
            "reference_positive_candidate_scope_override": (
                args.reference_positive_candidate_scope_override
            ),
            "reference_distance_candidate_strength_override": (
                args.reference_distance_candidate_strength_override
            ),
            "reference_distance_candidate_scope_override": (
                args.reference_distance_candidate_scope_override
            ),
            "reference_distance_candidate_mode_override": (
                args.reference_distance_candidate_mode_override
            ),
            "counterfactual_local_proposal_type_override": (
                args.counterfactual_local_proposal_type_override
            ),
            "counterfactual_local_proposal_strength_override": (
                args.counterfactual_local_proposal_strength_override
            ),
            "base_preserving_selector_enabled_override": bool(
                args.base_preserving_selector_enabled_override
            ),
            "counterfactual_curvature_residual_type": str(
                getattr(model.decoder, "counterfactual_curvature_residual_type", "unknown")
            ),
            "counterfactual_reference_residual_type": str(
                getattr(model.decoder, "counterfactual_reference_residual_type", "unknown")
            ),
            "counterfactual_direction_residual_type": str(
                getattr(model.decoder, "counterfactual_direction_residual_type", "unknown")
            ),
            "counterfactual_direction_residual_bank": args.counterfactual_direction_residual_bank,
            "counterfactual_curvature_residual_bank": args.counterfactual_curvature_residual_bank,
            "counterfactual_route_frame_budget": {
                "mode": str(getattr(model.decoder, "counterfactual_route_frame_budget_mode", "unknown")),
                "tangent_budget": float(
                    getattr(model.decoder, "counterfactual_route_frame_tangent_budget", float("nan"))
                ),
                "normal_budget": float(
                    getattr(model.decoder, "counterfactual_route_frame_normal_budget", float("nan"))
                ),
                "fields": (
                    sorted(getattr(model.decoder, "counterfactual_route_frame_fields", []) or [])
                    if getattr(model.decoder, "counterfactual_route_frame_fields", None) is not None
                    else None
                ),
            },
            "selection_policy": args.selection_policy,
            "selection_prior_path": str(args.selection_prior_path) if args.selection_prior_path else None,
            "selection_route_weight": float(args.selection_route_weight),
            "selection_token_guard_margin": float(args.selection_token_guard_margin),
            "selection_token_absolute_threshold": float(args.selection_token_absolute_threshold),
            "selection_min_route_improvement": float(args.selection_min_route_improvement),
            "selection_logit_guard_margin": float(args.selection_logit_guard_margin),
            "selection_preserve_direction": bool(args.selection_preserve_direction),
            "selection_preserve_target_direction": bool(args.selection_preserve_target_direction),
            "selection_hybrid": {
                "instruction_weight": float(args.selection_hybrid_instruction_weight),
                "target_effect_weight": float(args.selection_hybrid_target_effect_weight),
                "collateral_weight": float(args.selection_hybrid_collateral_weight),
                "collateral_margin": float(args.selection_hybrid_collateral_margin),
                "raw_collateral_margin": float(args.selection_hybrid_raw_collateral_margin),
                "logit_weight": float(args.selection_hybrid_logit_weight),
                "priority": args.selection_hybrid_priority,
            },
            "counterfactual_eval_selector": {
                "weight": float(getattr(model.decoder, "counterfactual_eval_selector_weight", 0.0)),
                "instruction_weight": float(getattr(model.decoder, "counterfactual_eval_selector_instruction_weight", 0.5)),
                "target_effect_weight": float(getattr(model.decoder, "counterfactual_eval_selector_target_effect_weight", 1.0)),
                "collateral_weight": float(getattr(model.decoder, "counterfactual_eval_selector_collateral_weight", 0.5)),
                "mode_switch_weight": float(getattr(model.decoder, "counterfactual_eval_selector_mode_switch_weight", 0.0)),
                "reference_mode_switch_weight": float(
                    getattr(model.decoder, "counterfactual_eval_selector_reference_mode_switch_weight", 0.0)
                ),
                "reference_reduction_weight": float(
                    getattr(model.decoder, "counterfactual_eval_selector_reference_reduction_weight", 0.0)
                ),
                "reference_score_weight": float(
                    getattr(model.decoder, "counterfactual_eval_selector_reference_score_weight", 0.0)
                ),
                "collateral_mode": str(getattr(model.decoder, "counterfactual_eval_selector_collateral_mode", "independent")),
                "feasible_first": bool(getattr(model.decoder, "counterfactual_eval_selector_feasible_first", True)),
                "success_margin": float(getattr(model.decoder, "counterfactual_eval_selector_success_margin", 0.0)),
                "collateral_margin": float(getattr(model.decoder, "counterfactual_eval_selector_collateral_margin", 0.08)),
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
                "field_min_success_rates": dict(sorted(transition_field_min_success_rates.items())),
                "field_min_effect_z": dict(sorted(transition_field_min_effect_z.items())),
                "field_min_token_match_rates": dict(sorted(transition_field_min_token_match_rates.items())),
                "field_min_pairs": dict(sorted(transition_field_min_pairs.items())),
                "field_max_collateral_z": dict(sorted(transition_field_max_collateral_z.items())),
                "allowlist": args.transition_allowlist,
                "num_fields": int(len(transition_table)),
                "stats": transition_filter_stats,
                "pair_sampling": args.pair_sampling,
            },
            "qwen_planner_checkpoint": str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None,
            "qwen_semantic_conditioning": qwen_provider is not None,
            "calibration_source": calibration_source,
            "calibration": calibrator.as_dict(),
            "failure_case_export": failure_case_export,
            "candidate_row_export": candidate_row_export,
            "visual_case_export": visual_case_export,
        }
    )

    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**metrics, "metrics_path": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
