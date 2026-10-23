from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from langdirecttraj.data.instruction_dataset import InstructionTrajectoryDataset, collate_instruction_batch
from langdirecttraj.eval.eval_anchor_decoder import prepare_eval_batch
from langdirecttraj.eval.eval_direction_calibration import (
    hybrid_predictions,
    imotion_style_direction_from_points,
)
from langdirecttraj.eval.instruction_geometry import (
    GeometryTokenCalibrator,
    contract_labels_from_tensors,
    geometry_values,
    macro_f1,
    tensor_dict_to_rows,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Table-3 ceiling by decoding mobility labels directly "
            "from the target trajectory geometry, without passing through the model."
        )
    )
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "target_instruction_ceiling")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--target-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--token-calibration-path", type=Path, required=True)
    parser.add_argument(
        "--direction-calibration-mode",
        choices=["contract", "token", "imotion_style", "hybrid", "all"],
        default="contract",
        help="How to infer the direction token for the target-trajectory ceiling.",
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


def evaluate_target_ceiling(
    loader: DataLoader,
    calibrator: GeometryTokenCalibrator,
    target_len: int,
    dataset: str,
    include_empty_labels: bool = False,
    label_decoder: str = "contract",
    direction_calibration_mode: str = "token",
    direction_hybrid_policy: str = "auto_lr_when_imotion_multi_else_imotion",
    imotion_params: Dict[str, float] | None = None,
) -> Dict:
    true_by_field: Dict[str, List[str]] = {field: [] for field in DEFAULT_MOBILITY_VOCAB.fields}
    pred_by_field: Dict[str, List[str]] = {field: [] for field in DEFAULT_MOBILITY_VOCAB.fields}
    direction_pred_by_mode: Dict[str, List[str]] = {}

    with torch.no_grad():
        for batch in loader:
            target, reference, route_scales = prepare_eval_batch(batch, target_len)
            geom_rows = tensor_dict_to_rows(geometry_values(target.float(), reference.float(), route_scales[:, 0].float()))
            contract_preds = contract_labels_from_tensors(
                target.float(),
                reference.float(),
                route_scales[:, 0].float(),
                dataset=dataset,
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
                for points in target.detach().cpu().float().numpy()
            ]
            if direction_calibration_mode == "contract":
                direction_pred = [labels["direction"] for labels in contract_preds]
            elif direction_calibration_mode == "imotion_style":
                direction_pred = imotion_direction_pred
            elif direction_calibration_mode == "hybrid":
                direction_pred = hybrid_predictions(token_direction_pred, imotion_direction_pred, direction_hybrid_policy)
            elif direction_calibration_mode == "all":
                hybrid_direction_pred = hybrid_predictions(
                    token_direction_pred,
                    imotion_direction_pred,
                    direction_hybrid_policy,
                )
                contract_direction_pred = [labels["direction"] for labels in contract_preds]
                direction_pred_by_mode.setdefault("contract", []).extend(str(item) for item in contract_direction_pred)
                direction_pred_by_mode.setdefault("token", []).extend(str(item) for item in token_direction_pred)
                direction_pred_by_mode.setdefault("imotion_style", []).extend(str(item) for item in imotion_direction_pred)
                direction_pred_by_mode.setdefault(
                    f"hybrid_{direction_hybrid_policy}",
                    [],
                ).extend(str(item) for item in hybrid_direction_pred)
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
        "direction_calibration_mode": direction_calibration_mode,
        "direction_hybrid_policy": direction_hybrid_policy if direction_calibration_mode in {"hybrid", "all"} else None,
        "direction_mode_comparison": direction_mode_metrics,
    }


def main() -> None:
    args = parse_args()
    jsonl_path = args.input or (
        REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_counterfactual_instruction_samples.filtered.jsonl"
    )
    dataset = InstructionTrajectoryDataset(jsonl_path, max_samples=args.max_samples, target_key=args.target_key)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_instruction_batch,
    )
    calibrator = GeometryTokenCalibrator.from_json(str(args.token_calibration_path))
    metrics = evaluate_target_ceiling(
        loader=loader,
        calibrator=calibrator,
        target_len=args.target_len,
        dataset=args.dataset,
        include_empty_labels=args.include_empty_labels,
        direction_calibration_mode=args.direction_calibration_mode,
        label_decoder=args.label_decoder,
        direction_hybrid_policy=args.direction_hybrid_policy,
        imotion_params={
            "straight_heading_rad": args.imotion_straight_heading_rad,
            "straight_lateral": args.imotion_straight_lateral,
            "uturn_longitudinal": args.imotion_uturn_longitudinal,
            "uturn_heading_rad": args.imotion_uturn_heading_rad,
            "multiturn_total_abs_rad": args.imotion_multiturn_total_abs_rad,
            "multiturn_count": args.imotion_multiturn_count,
        },
    )
    metrics.update(
        {
            "dataset": args.dataset,
            "input": str(jsonl_path),
            "target_key": args.target_key,
            "target_len": int(args.target_len),
            "calibration_source": str(args.token_calibration_path),
            "label_decoder": args.label_decoder,
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
