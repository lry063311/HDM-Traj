from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch

from langdirecttraj.data.instruction_dataset import collate_instruction_batch
from langdirecttraj.eval.eval_anchor_decoder import (
    build_model,
    load_checkpoint,
    prepare_eval_batch,
    select_prediction,
    summarize,
)
from langdirecttraj.eval.eval_counterfactual_controllability import (
    group_counterfactual_pairs,
    read_records,
    tensor_record,
)
from langdirecttraj.eval.eval_direction_calibration import (
    hybrid_predictions,
    imotion_style_direction_from_points,
)
from langdirecttraj.eval.instruction_geometry import (
    GeometryTokenCalibrator,
    geometry_values,
    tensor_dict_to_rows,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether direction labels, calibration predictions, "
            "iMotion-style rules, hybrid Table-3 predictions, and Table-4 "
            "signed target movement use the same semantic contract."
        )
    )
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "direction_contract_alignment")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--fields", nargs="+", default=["direction"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--token-calibration-path", type=Path, required=True)
    parser.add_argument(
        "--counterfactual-adapter-scale",
        type=float,
        default=0.0,
        help="Use the same adapter scale as Table-4 evaluation for changed-token records.",
    )
    parser.add_argument("--success-margin-z", type=float, default=0.0)
    parser.add_argument(
        "--hybrid-policy",
        default="auto_lr_when_imotion_multi_else_imotion",
        choices=[
            "auto_lr_else_imotion",
            "auto_lr_when_imotion_multi_else_imotion",
            "auto_straight_lr_else_imotion",
        ],
    )
    parser.add_argument("--imotion-straight-heading-rad", type=float, default=math.pi / 6.0)
    parser.add_argument("--imotion-straight-lateral", type=float, default=0.08)
    parser.add_argument("--imotion-uturn-longitudinal", type=float, default=-0.05)
    parser.add_argument("--imotion-uturn-heading-rad", type=float, default=2.0)
    parser.add_argument("--imotion-multiturn-total-abs-rad", type=float, default=4.5)
    parser.add_argument("--imotion-multiturn-count", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=128)
    return parser.parse_args()


def _resolve_input(dataset: str, path: Path | None) -> Path:
    if path is not None:
        return path
    return REPO_ROOT / "outputs" / "instruction_data" / f"{dataset}_counterfactual_instruction_samples.jsonl"


def _decode_tokens(token_ids: torch.Tensor) -> Dict[str, str]:
    return DEFAULT_MOBILITY_VOCAB.decode([int(x) for x in token_ids.detach().cpu().tolist()])


def _imotion_direction_from_geom(
    geom: Mapping[str, float],
    args: argparse.Namespace,
) -> str:
    # Reconstructing from points would be more exact, but geometry_values already
    # exposes the local-frame descriptors used by the differentiable loss. This
    # mirrors eval_direction_calibration's rule order without extra model passes.
    dx = float(geom.get("local_dx", 0.0))
    dy = float(geom.get("local_dy", 0.0))
    heading = float(geom.get("local_heading_diff_rad", 0.0))
    total_abs = float(geom.get("local_total_abs_turn_rad", 0.0))
    turn_count = float(geom.get("local_turn_count", 0.0))
    signed_turn = float(geom.get("local_signed_total_turn_rad", 0.0))
    if dx < float(args.imotion_uturn_longitudinal) and abs(heading) >= float(args.imotion_uturn_heading_rad):
        return "u_turn"
    if turn_count >= float(args.imotion_multiturn_count) or total_abs >= float(args.imotion_multiturn_total_abs_rad):
        return "multi_turn"
    if abs(heading) < float(args.imotion_straight_heading_rad) and abs(dy) < float(args.imotion_straight_lateral):
        return "straight"
    if dy > 0 or signed_turn > 0:
        return "left_turn"
    return "right_turn"


def _hybrid_direction(auto_pred: str, imotion_pred: str, policy: str) -> str:
    return hybrid_predictions([auto_pred], [imotion_pred], policy)[0]


def _prediction_for_record(
    model,
    qwen_provider,
    record: Dict,
    attr_dim: int,
    target_key: str,
    device: torch.device,
    adapter_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    item = tensor_record(record, target_key=target_key)
    batch = collate_instruction_batch([item])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    target, reference, route_scales = prepare_eval_batch(batch, model.decoder.target_len)
    attr_cond = make_attr_cond(batch, attr_dim, device)
    semantic_plan_tokens = extract_qwen_semantic_tokens(qwen_provider, batch["instruction"], device)
    counterfactual_mode = bool(record.get("counterfactual", {}).get("enabled")) and float(adapter_scale) != 0.0
    with torch.no_grad():
        pred_out = model(
            reference,
            batch["mobility_tokens"],
            attr_cond,
            semantic_plan_tokens=semantic_plan_tokens,
            counterfactual_mode=counterfactual_mode,
            counterfactual_adapter_scale=float(adapter_scale),
            return_aux=True,
        )
        raw_pred = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        pred = select_prediction(pred_out)
        aux = pred_out[1] if isinstance(pred_out, tuple) and isinstance(pred_out[1], dict) else {}
    return pred, raw_pred, target, reference, route_scales[:, 0], aux


def _geom_row(pred: torch.Tensor, reference: torch.Tensor, scale: torch.Tensor) -> Dict[str, float]:
    return tensor_dict_to_rows(geometry_values(pred.float(), reference.float(), scale.float()))[0]


def _selected_oracle(aux: Dict, candidates: torch.Tensor, target: torch.Tensor, scale: torch.Tensor) -> Dict[str, float | int | None]:
    logits = aux.get("candidate_logits") if isinstance(aux, dict) else None
    if candidates.ndim == 3:
        candidates = candidates[:, None]
    if logits is None or candidates.ndim != 4:
        return {
            "selected_mode": None,
            "oracle_mode": None,
            "selected_logit": None,
            "oracle_logit": None,
            "selected_ade_m": None,
            "oracle_ade_m": None,
        }
    ade = torch.linalg.norm(candidates - target[:, None], dim=-1).mean(dim=-1) * scale.view(-1, 1)
    selected = int(logits.argmax(dim=1)[0].detach().cpu().item())
    oracle = int(ade.argmin(dim=1)[0].detach().cpu().item())
    return {
        "selected_mode": selected,
        "oracle_mode": oracle,
        "selected_logit": float(logits[0, selected].detach().cpu().item()),
        "oracle_logit": float(logits[0, oracle].detach().cpu().item()),
        "selected_ade_m": float(ade[0, selected].detach().cpu().item()),
        "oracle_ade_m": float(ade[0, oracle].detach().cpu().item()),
    }


def _table4_direction_row(
    calibrator: GeometryTokenCalibrator,
    base_geom: Dict[str, float],
    cf_geom: Dict[str, float],
    source_token: str,
    target_token: str,
    success_margin_z: float,
) -> Dict[str, float | str | bool]:
    field = "direction"
    base_score = calibrator.score_token(field, target_token, base_geom)
    cf_score = calibrator.score_token(field, target_token, cf_geom)
    target_effect = float(base_score - cf_score)
    (
        metric_name,
        base_to_target_delta,
        observed_delta,
        signed_delta,
        signed_delta_z,
    ) = calibrator.transition_metric_delta(field, source_token, target_token, base_geom, cf_geom)
    return {
        "table4_metric": metric_name,
        "table4_base_to_target_delta": float(base_to_target_delta),
        "table4_observed_delta": float(observed_delta),
        "table4_signed_delta": float(signed_delta),
        "table4_signed_delta_z": float(signed_delta_z),
        "table4_target_effect_z": target_effect,
        "table4_success": bool(target_effect > float(success_margin_z)),
    }


def _direction_descriptor_row(prefix: str, geom: Mapping[str, float]) -> Dict[str, float]:
    keys = [
        "local_dx",
        "local_dy",
        "local_heading_diff_rad",
        "local_abs_heading_diff_rad",
        "local_signed_total_turn_rad",
        "local_total_abs_turn_rad",
        "local_turn_count",
    ]
    return {f"{prefix}_{key}": float(geom.get(key, 0.0)) for key in keys}


def _direction_rule_gaps(row: Mapping, args: argparse.Namespace) -> Dict[str, float]:
    target = str(row.get("target_token", ""))
    if target == "straight":
        return {
            "straight_heading_gap": max(
                0.0,
                float(row.get("cf_local_abs_heading_diff_rad", 0.0)) - float(args.imotion_straight_heading_rad),
            ),
            "straight_lateral_gap": max(
                0.0,
                abs(float(row.get("cf_local_dy", 0.0))) - float(args.imotion_straight_lateral),
            ),
            "straight_turn_count_gap": max(
                0.0,
                float(row.get("cf_local_turn_count", 0.0)) - float(args.imotion_multiturn_count),
            ),
            "straight_turn_abs_gap": max(
                0.0,
                float(row.get("cf_local_total_abs_turn_rad", 0.0)) - float(args.imotion_multiturn_total_abs_rad),
            ),
        }
    if target == "u_turn":
        return {
            "uturn_dx_gap": max(
                0.0,
                float(row.get("cf_local_dx", 0.0)) - float(args.imotion_uturn_longitudinal),
            ),
            "uturn_heading_gap": max(
                0.0,
                float(args.imotion_uturn_heading_rad) - abs(float(row.get("cf_local_heading_diff_rad", 0.0))),
            ),
        }
    if target == "multi_turn":
        return {
            "multi_turn_count_gap": max(
                0.0,
                float(args.imotion_multiturn_count) - float(row.get("cf_local_turn_count", 0.0)),
            ),
            "multi_turn_abs_gap": max(
                0.0,
                float(args.imotion_multiturn_total_abs_rad)
                - float(row.get("cf_local_total_abs_turn_rad", 0.0)),
            ),
        }
    return {}


def evaluate(args: argparse.Namespace) -> Dict:
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint, device, token_calibration_path=args.token_calibration_path)
    saved_args = checkpoint.get("args", {})
    attr_dim = int(saved_args.get("attr_dim", 6))
    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device)
    calibrator = GeometryTokenCalibrator.from_json(str(args.token_calibration_path))

    records = read_records(_resolve_input(args.dataset, args.input), args.max_records)
    pairs, _filter_stats = group_counterfactual_pairs(records, set(args.fields), args.max_pairs)
    if not pairs:
        raise ValueError("No counterfactual pairs found for direction contract alignment.")

    rows: List[Dict] = []
    for original, counterfactual in pairs:
        cf_meta = counterfactual.get("counterfactual", {})
        field = str(cf_meta.get("field", "unknown"))
        if field != "direction":
            continue
        source_token = str(cf_meta.get("from", original.get("labels", {}).get("direction", "unknown")))
        target_token = str(cf_meta.get("to", counterfactual.get("labels", {}).get("direction", "unknown")))
        if target_token not in DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]:
            continue

        base_pred, base_candidates, base_target, base_ref, base_scale, base_aux = _prediction_for_record(
            model,
            qwen_provider,
            original,
            attr_dim,
            args.target_key,
            device,
            adapter_scale=0.0,
        )
        cf_pred, cf_candidates, cf_target, cf_ref, cf_scale, cf_aux = _prediction_for_record(
            model,
            qwen_provider,
            counterfactual,
            attr_dim,
            args.target_key,
            device,
            adapter_scale=float(args.counterfactual_adapter_scale),
        )
        base_geom = _geom_row(base_pred, base_ref, base_scale)
        cf_geom = _geom_row(cf_pred, cf_ref, cf_scale)
        base_auto = calibrator.predict_field("direction", base_geom)
        cf_auto = calibrator.predict_field("direction", cf_geom)
        base_imotion = _imotion_direction_from_geom(base_geom, args)
        cf_imotion = _imotion_direction_from_geom(cf_geom, args)
        base_hybrid = _hybrid_direction(base_auto, base_imotion, args.hybrid_policy)
        cf_hybrid = _hybrid_direction(cf_auto, cf_imotion, args.hybrid_policy)
        table4 = _table4_direction_row(
            calibrator,
            base_geom,
            cf_geom,
            source_token=source_token,
            target_token=target_token,
            success_margin_z=float(args.success_margin_z),
        )
        base_modes = _selected_oracle(base_aux, base_candidates, base_target, base_scale)
        cf_modes = _selected_oracle(cf_aux, cf_candidates, cf_target, cf_scale)
        row = {
            "source_sample_id": int(cf_meta.get("source_sample_id", original.get("sample_id", -1))),
            "counterfactual_id": int(counterfactual.get("counterfactual_id", counterfactual.get("sample_id", -1))),
            "transition": f"{source_token}->{target_token}",
            "source_token": source_token,
            "target_token": target_token,
            "base_label": str(original.get("labels", {}).get("direction", "unknown")),
            "cf_label": str(counterfactual.get("labels", {}).get("direction", "unknown")),
            "base_auto_direction": base_auto,
            "cf_auto_direction": cf_auto,
            "base_imotion_direction": base_imotion,
            "cf_imotion_direction": cf_imotion,
            "base_hybrid_direction": base_hybrid,
            "cf_hybrid_direction": cf_hybrid,
            "auto_hits_target": bool(cf_auto == target_token),
            "imotion_hits_target": bool(cf_imotion == target_token),
            "hybrid_hits_target": bool(cf_hybrid == target_token),
            "hybrid_changed": bool(base_hybrid != cf_hybrid),
            **_direction_descriptor_row("base", base_geom),
            **_direction_descriptor_row("cf", cf_geom),
            **table4,
            "base_selected_mode": base_modes["selected_mode"],
            "base_oracle_mode": base_modes["oracle_mode"],
            "cf_selected_mode": cf_modes["selected_mode"],
            "cf_oracle_mode": cf_modes["oracle_mode"],
            "cf_selected_ade_m": cf_modes["selected_ade_m"],
            "cf_oracle_ade_m": cf_modes["oracle_ade_m"],
            "cf_selected_oracle_gap_m": (
                None
                if cf_modes["selected_ade_m"] is None or cf_modes["oracle_ade_m"] is None
                else float(cf_modes["selected_ade_m"]) - float(cf_modes["oracle_ade_m"])
            ),
            "instruction": str(counterfactual.get("instruction", "")),
        }
        row.update(_direction_rule_gaps(row, args))
        rows.append(row)
        if len(rows) >= int(args.max_cases):
            break

    if not rows:
        raise ValueError("No direction counterfactual rows were evaluated.")
    return summarize_rows(rows, args)


def _mean_bool(rows: List[Dict], key: str) -> float:
    return float(np.mean([1.0 if row.get(key) else 0.0 for row in rows])) if rows else float("nan")


def _mean_number(rows: List[Dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def summarize_rows(rows: List[Dict], args: argparse.Namespace) -> Dict:
    by_transition: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_transition[str(row["transition"])].append(row)
    gap_keys = sorted({key for row in rows for key in row if key.endswith("_gap")})
    summary = {
        "num_rows": len(rows),
        "auto_target_hit_rate": _mean_bool(rows, "auto_hits_target"),
        "imotion_target_hit_rate": _mean_bool(rows, "imotion_hits_target"),
        "hybrid_target_hit_rate": _mean_bool(rows, "hybrid_hits_target"),
        "table4_success_rate": _mean_bool(rows, "table4_success"),
        "table4_target_effect_z": summarize([float(row["table4_target_effect_z"]) for row in rows]),
        "table4_signed_delta_z": summarize([float(row["table4_signed_delta_z"]) for row in rows]),
        "cf_selected_ade_m": summarize(
            [float(row["cf_selected_ade_m"]) for row in rows if row.get("cf_selected_ade_m") is not None]
        ),
        "cf_selected_oracle_gap_m": summarize(
            [float(row["cf_selected_oracle_gap_m"]) for row in rows if row.get("cf_selected_oracle_gap_m") is not None]
        ),
        "cf_selected_mode_counts": dict(Counter(str(row.get("cf_selected_mode")) for row in rows)),
        "cf_oracle_mode_counts": dict(Counter(str(row.get("cf_oracle_mode")) for row in rows)),
        "rule_gaps": {
            key: summarize([float(row[key]) for row in rows if key in row])
            for key in gap_keys
        },
        "by_transition": {
            transition: {
                "num_rows": len(items),
                "auto_target_hit_rate": _mean_bool(items, "auto_hits_target"),
                "imotion_target_hit_rate": _mean_bool(items, "imotion_hits_target"),
                "hybrid_target_hit_rate": _mean_bool(items, "hybrid_hits_target"),
                "table4_success_rate": _mean_bool(items, "table4_success"),
                "table4_target_effect_z_mean": _mean_number(items, "table4_target_effect_z"),
                "table4_signed_delta_z_mean": _mean_number(items, "table4_signed_delta_z"),
                "cf_selected_mode_counts": dict(Counter(str(row.get("cf_selected_mode")) for row in items)),
                "cf_oracle_mode_counts": dict(Counter(str(row.get("cf_oracle_mode")) for row in items)),
                "rule_gaps": {
                    key: _mean_number(items, key)
                    for key in sorted({key for row in items for key in row if key.endswith("_gap")})
                },
            }
            for transition, items in sorted(by_transition.items())
        },
        "checkpoint": str(args.checkpoint),
        "input": str(_resolve_input(args.dataset, args.input)),
        "token_calibration_path": str(args.token_calibration_path),
        "hybrid_policy": args.hybrid_policy,
        "rows": rows,
    }
    return summary


def write_outputs(summary: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(summary.pop("rows"))
    metrics_path = output_dir / "metrics.json"
    rows_path = output_dir / "rows.jsonl"
    csv_path = output_dir / "rows.csv"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys())
        seen = set(fieldnames)
        for row in rows[1:]:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                **summary,
                "metrics_path": str(metrics_path),
                "rows_path": str(rows_path),
                "csv_path": str(csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    summary = evaluate(args)
    write_outputs(summary, args.output_dir)


if __name__ == "__main__":
    main()
