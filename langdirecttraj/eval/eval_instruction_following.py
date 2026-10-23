from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from langdirecttraj.data.instruction_dataset import InstructionTrajectoryDataset, collate_instruction_batch
from langdirecttraj.eval.eval_anchor_decoder import (
    build_model,
    load_checkpoint,
    prepare_eval_batch,
)
from langdirecttraj.eval.eval_direction_calibration import (
    hybrid_predictions,
    imotion_style_direction_from_points,
)
from langdirecttraj.eval.instruction_geometry import (
    GeometryTokenCalibrator,
    add_pace_metrics_to_rows,
    apply_pace_progress_labels,
    contract_labels_from_tensors,
    geometry_values,
    macro_f1,
    tensor_dict_to_rows,
)
from langdirecttraj.eval.route_selection import load_selection_prior, select_prediction_with_policy
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.train.train_anchor_decoder import (
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
    parser = argparse.ArgumentParser(description="Table 3: instruction-following fidelity evaluation.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "instruction_following_eval")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--disable-qwen", action="store_true", help="Do not inherit or load a saved Qwen planner.")
    parser.add_argument("--token-calibration-path", type=Path, default=None)
    parser.add_argument("--qwen-plan-cache", type=Path, default=None)
    parser.add_argument("--road-corridor-cache", type=Path, default=None)
    parser.add_argument(
        "--reference-positive-candidate-strength-override",
        type=float,
        default=None,
        help="Temporarily override explicit close-reference candidate construction strength.",
    )
    parser.add_argument(
        "--reference-positive-candidate-scope-override",
        choices=["all", "none", "counterfactual_reference_adherence"],
        default=None,
        help="Temporarily override explicit close-reference candidate construction scope.",
    )
    parser.add_argument(
        "--reference-distance-candidate-strength-override",
        type=float,
        default=None,
        help="Temporarily override bounded reference-distance candidate proposal strength.",
    )
    parser.add_argument(
        "--reference-distance-candidate-scope-override",
        choices=["all", "none", "instruction", "counterfactual_reference_adherence"],
        default=None,
        help="Temporarily override bounded reference-distance candidate proposal scope.",
    )
    parser.add_argument(
        "--reference-distance-candidate-mode-override",
        choices=["global_normal", "local_direction_preserving"],
        default=None,
        help="Temporarily override bounded reference-distance candidate proposal shape.",
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
    parser.add_argument(
        "--qwen-plan-scale-override",
        type=float,
        default=None,
        help="Evaluation-only override for the dense Qwen plan-token conditioning scale.",
    )
    parser.add_argument("--route-proposal-logit-weight-override", type=float, default=None)
    parser.add_argument("--route-control-selection-adapter-weight-override", type=float, default=None)
    parser.add_argument("--route-control-selection-adapter-clip-override", type=float, default=None)
    parser.add_argument(
        "--direction-calibration-mode",
        choices=["contract", "token", "imotion_style", "hybrid", "all"],
        default="contract",
        help="How to infer the direction token for Table 3.",
    )
    parser.add_argument(
        "--label-decoder",
        choices=["contract", "calibration"],
        default="contract",
        help="Use the official shared geometry contract or the calibration nearest-center decoder for Table 3.",
    )
    parser.add_argument(
        "--direction-hybrid-policy",
        choices=[
            "auto_lr_else_imotion",
            "auto_lr_when_imotion_multi_else_imotion",
            "auto_straight_lr_else_imotion",
        ],
        default="auto_lr_when_imotion_multi_else_imotion",
    )
    parser.add_argument("--imotion-straight-heading-rad", type=float, default=math.pi / 6.0)
    parser.add_argument("--imotion-straight-lateral", type=float, default=0.08)
    parser.add_argument("--imotion-uturn-longitudinal", type=float, default=-0.05)
    parser.add_argument("--imotion-uturn-heading-rad", type=float, default=2.0)
    parser.add_argument("--imotion-multiturn-total-abs-rad", type=float, default=4.5)
    parser.add_argument("--imotion-multiturn-count", type=int, default=3)
    parser.add_argument(
        "--include-empty-labels",
        action="store_true",
        help="Include zero-support labels in macro-F1. Default excludes labels absent from the evaluated split.",
    )
    return parser.parse_args()


def build_eval_target_calibrator(
    loader: DataLoader,
    target_len: int,
    device: torch.device,
) -> GeometryTokenCalibrator:
    rows: List[tuple[Dict[str, str], Dict[str, float]]] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            target, reference, route_scales = prepare_eval_batch(batch, target_len)
            geom_rows = tensor_dict_to_rows(geometry_values(target.float(), reference.float(), route_scales[:, 0].float()))
            geom_rows = add_pace_metrics_to_rows(geom_rows, batch.get("trip_attributes", [{} for _ in geom_rows]))
            rows.extend((labels, geom) for labels, geom in zip(batch["labels"], geom_rows))
    return GeometryTokenCalibrator.from_target_batches(rows)


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    attr_dim: int,
    calibrator: GeometryTokenCalibrator,
    dataset: str,
    qwen_provider=None,
    include_empty_labels: bool = False,
    label_decoder: str = "contract",
    direction_calibration_mode: str = "token",
    direction_hybrid_policy: str = "auto_lr_when_imotion_multi_else_imotion",
    imotion_params: Dict[str, float] | None = None,
    selection_prior: dict | None = None,
    selection_policy: str = "reranker",
    selection_route_weight: float = 1.0,
    selection_token_guard_margin: float = 0.0,
    selection_token_absolute_threshold: float = float("inf"),
    selection_min_route_improvement: float = 0.0,
    selection_logit_guard_margin: float = float("inf"),
    selection_preserve_direction: bool = False,
    selection_preserve_target_direction: bool = False,
) -> Dict:
    true_by_field: Dict[str, List[str]] = {field: [] for field in DEFAULT_MOBILITY_VOCAB.fields}
    pred_by_field: Dict[str, List[str]] = {field: [] for field in DEFAULT_MOBILITY_VOCAB.fields}
    direction_pred_by_mode: Dict[str, List[str]] = defaultdict(list)
    selected_indices: List[int] = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            _, reference, route_scales = prepare_eval_batch(batch, model.decoder.target_len)
            attr_cond = make_attr_cond(batch, attr_dim, device)
            semantic_plan_tokens = qwen_semantic_bundle_for_batch(qwen_provider, batch, device)
            pred_out = model(
                reference,
                batch["mobility_tokens"],
                attr_cond,
                semantic_plan_tokens=semantic_plan_tokens,
                **road_condition_kwargs(batch),
                return_aux=True,
            )
            if isinstance(pred_out, tuple):
                pred_raw, aux = pred_out
            else:
                pred_raw, aux = pred_out, {}
            pred, selected_idx, _ = select_prediction_with_policy(
                (pred_raw, aux),
                reference=reference,
                reference_lonlat=raw_batch["reference"].to(device),
                route_scales=route_scales[:, 0],
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
                dataset=dataset,
            )
            logits = aux.get("candidate_logits") if isinstance(aux, dict) else None
            if pred_raw.ndim == 4 and logits is not None:
                selected_indices.extend(selected_idx.detach().cpu().tolist())

            geom_rows = tensor_dict_to_rows(geometry_values(pred.float(), reference.float(), route_scales[:, 0].float()))
            geom_rows = add_pace_metrics_to_rows(geom_rows, batch.get("trip_attributes", [{} for _ in geom_rows]))
            contract_preds = contract_labels_from_tensors(
                pred.float(),
                reference.float(),
                route_scales[:, 0].float(),
                dataset=dataset,
            )
            contract_preds = apply_pace_progress_labels(
                contract_preds,
                geom_rows,
                batch.get("trip_attributes", [{} for _ in geom_rows]),
            )
            token_direction_pred = [calibrator.predict_field("direction", geom) for geom in geom_rows]
            imotion_direction_pred = [
                imotion_style_direction_from_points(
                    points,
                    straight_heading_rad=float((imotion_params or {}).get("straight_heading_rad", math.pi / 6.0)),
                    straight_lateral=float((imotion_params or {}).get("straight_lateral", 0.08)),
                    uturn_longitudinal=float((imotion_params or {}).get("uturn_longitudinal", -0.05)),
                    uturn_heading_rad=float((imotion_params or {}).get("uturn_heading_rad", 2.0)),
                    multiturn_total_abs_rad=float((imotion_params or {}).get("multiturn_total_abs_rad", 4.5)),
                    multiturn_count=int((imotion_params or {}).get("multiturn_count", 3)),
                )
                for points in pred.detach().cpu().float().numpy()
            ]
            if direction_calibration_mode == "contract":
                direction_pred = [labels["direction"] for labels in contract_preds]
            elif direction_calibration_mode == "imotion_style":
                direction_pred = imotion_direction_pred
            elif direction_calibration_mode == "hybrid":
                direction_pred = hybrid_predictions(token_direction_pred, imotion_direction_pred, direction_hybrid_policy)
            elif direction_calibration_mode == "all":
                hybrid_direction_pred = hybrid_predictions(token_direction_pred, imotion_direction_pred, direction_hybrid_policy)
                contract_direction_pred = [labels["direction"] for labels in contract_preds]
                for mode_name, mode_pred in {
                    "contract": contract_direction_pred,
                    "token": token_direction_pred,
                    "imotion_style": imotion_direction_pred,
                    f"hybrid_{direction_hybrid_policy}": hybrid_direction_pred,
                }.items():
                    direction_pred_by_mode[mode_name].extend(str(item) for item in mode_pred)
                direction_pred = contract_direction_pred if label_decoder == "contract" else token_direction_pred
            else:
                direction_pred = token_direction_pred

            for idx, (labels, geom) in enumerate(zip(batch["labels"], geom_rows)):
                pred_labels = dict(contract_preds[idx]) if label_decoder == "contract" else calibrator.predict_labels(geom)
                pred_labels["direction"] = direction_pred[idx]
                for field in DEFAULT_MOBILITY_VOCAB.fields:
                    true_by_field[field].append(str(labels.get(field, "unknown")))
                    pred_by_field[field].append(str(pred_labels[field]))

    field_metrics = {}
    for field in DEFAULT_MOBILITY_VOCAB.fields:
        field_metrics[field] = macro_f1(
            true_by_field[field],
            pred_by_field[field],
            DEFAULT_MOBILITY_VOCAB.token_to_id[field].keys(),
            include_empty=include_empty_labels,
        )
    mean_f1 = float(np.mean([field_metrics[field]["macro_f1"] for field in DEFAULT_MOBILITY_VOCAB.fields]))
    direction_mode_metrics = {}
    if direction_calibration_mode == "all":
        for mode_name, mode_pred in sorted(direction_pred_by_mode.items()):
            direction_metric = macro_f1(
                true_by_field["direction"],
                mode_pred,
                DEFAULT_MOBILITY_VOCAB.token_to_id["direction"].keys(),
                include_empty=include_empty_labels,
            )
            table3_with_mode = {
                field: field_metrics[field]["macro_f1"] for field in DEFAULT_MOBILITY_VOCAB.fields
            }
            table3_with_mode["direction"] = direction_metric["macro_f1"]
            direction_mode_metrics[mode_name] = {
                "direction_f1": float(direction_metric["macro_f1"]),
                "direction_accuracy": float(direction_metric["accuracy"]),
                "mean_instruction_following_f1": float(np.mean(list(table3_with_mode.values()))),
                "per_label": direction_metric["per_label"],
            }
    short_names = {
        "distance": "distance_f1",
        "direction": "direction_f1",
        "curvature": "curvature_f1",
        "progress": "progress_f1",
        "reference_adherence": "reference_adherence_f1",
    }
    return {
        "num_samples": int(len(true_by_field[DEFAULT_MOBILITY_VOCAB.fields[0]])),
        "table3_instruction_following": {
            short_names[field]: float(field_metrics[field]["macro_f1"])
            for field in DEFAULT_MOBILITY_VOCAB.fields
        }
        | {"mean_instruction_following_f1": mean_f1},
        "field_metrics": field_metrics,
        "selected_index_counts": {
            str(idx): int(selected_indices.count(idx)) for idx in sorted(set(selected_indices))
        },
        "direction_calibration_mode": direction_calibration_mode,
        "direction_hybrid_policy": direction_hybrid_policy if direction_calibration_mode in {"hybrid", "all"} else None,
        "direction_mode_comparison": direction_mode_metrics,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
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
    saved_args = checkpoint.get("args", {})
    attr_dim = int(saved_args.get("attr_dim", 6))
    if args.disable_qwen or args.qwen_plan_cache is not None:
        args.qwen_planner_checkpoint = None
    elif args.qwen_planner_checkpoint is None:
        saved_qwen_checkpoint = saved_args.get("qwen_planner_checkpoint")
        if saved_qwen_checkpoint:
            args.qwen_planner_checkpoint = Path(saved_qwen_checkpoint)
    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device) if args.qwen_planner_checkpoint else None
    selection_prior = load_selection_prior(args.selection_prior_path, device)

    jsonl_path = args.input or (
        REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_counterfactual_instruction_samples.filtered.jsonl"
    )
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
    if args.token_calibration_path is not None and args.token_calibration_path.exists():
        calibrator = GeometryTokenCalibrator.from_json(str(args.token_calibration_path))
        calibration_source = str(args.token_calibration_path)
    else:
        calibrator = build_eval_target_calibrator(loader, model.decoder.target_len, device)
        calibration_source = "evaluated target trajectories"
    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        attr_dim=attr_dim,
        calibrator=calibrator,
        dataset=args.dataset,
        qwen_provider=qwen_provider,
        include_empty_labels=args.include_empty_labels,
        label_decoder=args.label_decoder,
        direction_calibration_mode=args.direction_calibration_mode,
        direction_hybrid_policy=args.direction_hybrid_policy,
        imotion_params={
            "straight_heading_rad": args.imotion_straight_heading_rad,
            "straight_lateral": args.imotion_straight_lateral,
            "uturn_longitudinal": args.imotion_uturn_longitudinal,
            "uturn_heading_rad": args.imotion_uturn_heading_rad,
            "multiturn_total_abs_rad": args.imotion_multiturn_total_abs_rad,
            "multiturn_count": args.imotion_multiturn_count,
        },
        selection_prior=selection_prior,
        selection_policy=args.selection_policy,
        selection_route_weight=args.selection_route_weight,
        selection_token_guard_margin=args.selection_token_guard_margin,
        selection_token_absolute_threshold=args.selection_token_absolute_threshold,
        selection_min_route_improvement=args.selection_min_route_improvement,
        selection_logit_guard_margin=args.selection_logit_guard_margin,
        selection_preserve_direction=args.selection_preserve_direction,
        selection_preserve_target_direction=args.selection_preserve_target_direction,
    )
    metrics.update(
        {
            "dataset": args.dataset,
            "input": str(jsonl_path),
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "target_key": args.target_key,
            "qwen_planner_checkpoint": str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None,
            "qwen_semantic_conditioning": qwen_provider is not None,
            "calibration_source": calibration_source,
            "label_decoder": args.label_decoder,
            "selection_policy": args.selection_policy,
            "selection_prior_path": str(args.selection_prior_path) if args.selection_prior_path else None,
            "selection_route_weight": float(args.selection_route_weight),
            "selection_token_guard_margin": float(args.selection_token_guard_margin),
            "selection_token_absolute_threshold": float(args.selection_token_absolute_threshold),
            "selection_min_route_improvement": float(args.selection_min_route_improvement),
            "selection_logit_guard_margin": float(args.selection_logit_guard_margin),
            "selection_preserve_direction": bool(args.selection_preserve_direction),
            "selection_preserve_target_direction": bool(args.selection_preserve_target_direction),
            "reference_positive_candidate_strength_override": (
                args.reference_positive_candidate_strength_override
            ),
            "reference_positive_candidate_scope_override": (
                args.reference_positive_candidate_scope_override
            ),
            "reference_distance_candidate_strength_override": (
                None
                if args.reference_distance_candidate_strength_override is None
                else float(args.reference_distance_candidate_strength_override)
            ),
            "reference_distance_candidate_scope_override": (
                args.reference_distance_candidate_scope_override
            ),
            "reference_distance_candidate_mode_override": (
                args.reference_distance_candidate_mode_override
            ),
            "calibration": calibrator.as_dict(),
        }
    )

    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**metrics, "metrics_path": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
