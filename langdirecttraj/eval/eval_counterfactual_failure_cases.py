from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from langdirecttraj.data.instruction_dataset import collate_instruction_batch
from langdirecttraj.eval.eval_anchor_decoder import (
    build_model,
    load_checkpoint,
    prepare_eval_batch,
    summarize,
)
from langdirecttraj.eval.eval_counterfactual_controllability import (
    build_pair_calibrator,
    collateral_drift_z,
    group_counterfactual_pairs,
    read_records,
    tensor_record,
)
from langdirecttraj.eval.eval_multi_candidate_diagnostics import candidate_errors_m
from langdirecttraj.eval.instruction_geometry import (
    GeometryTokenCalibrator,
    geometry_values,
    tensor_dict_to_rows,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.train.control_losses import load_counterfactual_transition_table
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export sample-level counterfactual failures for Table 4 diagnosis."
    )
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "counterfactual_failure_cases")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=128)
    parser.add_argument("--max-cases", type=int, default=128)
    parser.add_argument(
        "--pair-sampling",
        choices=["sequential", "balanced_field"],
        default="balanced_field",
        help="Match Table-4 pair sampling when exporting failure cases.",
    )
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["distance", "direction", "curvature", "progress", "reference_adherence"],
    )
    parser.add_argument("--success-margin-z", type=float, default=0.0)
    parser.add_argument(
        "--drift-warning-z",
        type=float,
        default=0.12,
        help="Rows with collateral drift above this value are flagged as drift-heavy.",
    )
    parser.add_argument(
        "--only-failures",
        action="store_true",
        help="Write only rows with diagnostic warning tags such as failed target effect, high drift, wrong predicted token, or wrong selected mode.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--token-calibration-path", type=Path, default=None)
    parser.add_argument("--counterfactual-adapter-scale", type=float, default=0.0)
    parser.add_argument("--transition-diagnostics-path", type=Path, default=None)
    parser.add_argument("--transition-min-success-rate", type=float, default=0.6)
    parser.add_argument("--transition-min-effect-z", type=float, default=0.05)
    parser.add_argument("--transition-min-token-match-rate", type=float, default=0.5)
    parser.add_argument("--transition-min-pairs", type=int, default=2)
    parser.add_argument("--transition-max-collateral-z", type=float, default=0.0)
    return parser.parse_args()


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _compact_float_list(values: torch.Tensor) -> List[float]:
    return [float(x) for x in values.detach().cpu().float().tolist()]


def _make_prediction_details(
    model,
    qwen_provider,
    record: Dict,
    attr_dim: int,
    target_key: str,
    calibrator: GeometryTokenCalibrator,
    device: torch.device,
    counterfactual_adapter_scale: float = 0.0,
) -> Dict:
    item = tensor_record(record, target_key=target_key)
    batch = collate_instruction_batch([item])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    target, reference, route_scales = prepare_eval_batch(batch, model.decoder.target_len)
    route_scale = route_scales[:, 0]
    attr_cond = make_attr_cond(batch, attr_dim, device)
    semantic_plan_tokens = extract_qwen_semantic_tokens(qwen_provider, batch["instruction"], device)

    with torch.no_grad():
        counterfactual_mode = (
            bool(record.get("counterfactual", {}).get("enabled"))
            and float(counterfactual_adapter_scale) != 0.0
        )
        pred_out = model(
            reference,
            batch["mobility_tokens"],
            attr_cond,
            semantic_plan_tokens=semantic_plan_tokens,
            counterfactual_mode=counterfactual_mode,
            counterfactual_adapter_scale=float(counterfactual_adapter_scale),
            return_aux=True,
        )
    pred, aux = pred_out if isinstance(pred_out, tuple) else (pred_out, {})
    if pred.ndim == 3:
        pred = pred[:, None]

    logits = aux.get("candidate_logits") if isinstance(aux, dict) else None
    if logits is None:
        selected_idx = torch.zeros(pred.shape[0], dtype=torch.long, device=device)
        logits = torch.zeros(pred.shape[:2], dtype=pred.dtype, device=device)
    else:
        selected_idx = logits.argmax(dim=1)

    ade, fde = candidate_errors_m(pred, target, route_scale)
    oracle_idx = ade.argmin(dim=1)
    batch_idx = torch.arange(pred.shape[0], device=device)
    selected = pred[batch_idx, selected_idx]
    oracle = pred[batch_idx, oracle_idx]
    selected_geom = tensor_dict_to_rows(
        geometry_values(selected.float(), reference.float(), route_scale.float())
    )[0]
    oracle_geom = tensor_dict_to_rows(
        geometry_values(oracle.float(), reference.float(), route_scale.float())
    )[0]
    target_geom = tensor_dict_to_rows(
        geometry_values(target.float(), reference.float(), route_scale.float())
    )[0]
    probs = torch.softmax(logits, dim=1)

    return {
        "sample_id": int(record.get("sample_id", record.get("counterfactual_id", 0))),
        "instruction": str(record.get("instruction", "")),
        "labels": dict(record.get("labels", {})),
        "predicted_labels": calibrator.predict_labels(selected_geom),
        "oracle_predicted_labels": calibrator.predict_labels(oracle_geom),
        "selected_geom": selected_geom,
        "oracle_geom": oracle_geom,
        "target_geom": target_geom,
        "selected_index": int(selected_idx[0].item()),
        "oracle_index": int(oracle_idx[0].item()),
        "selected_ade_m": float(ade[0, selected_idx[0]].item()),
        "oracle_ade_m": float(ade[0, oracle_idx[0]].item()),
        "selected_fde_m": float(fde[0, selected_idx[0]].item()),
        "oracle_fde_m": float(fde[0, oracle_idx[0]].item()),
        "candidate_ade_m": _compact_float_list(ade[0]),
        "candidate_fde_m": _compact_float_list(fde[0]),
        "candidate_logits": _compact_float_list(logits[0]),
        "candidate_probs": _compact_float_list(probs[0]),
    }


def _failure_tags(
    success: float,
    drift: float,
    drift_warning_z: float,
    field_name: str,
    target_token: str,
    base_details: Dict,
    cf_details: Dict,
) -> List[str]:
    tags: List[str] = []
    if success < 1.0:
        tags.append("target_not_improved")
    if np.isfinite(drift) and drift > float(drift_warning_z):
        tags.append("high_collateral_drift")
    if cf_details["predicted_labels"].get(field_name) != target_token:
        tags.append("predicted_token_not_target")
    if base_details["selected_index"] == cf_details["selected_index"]:
        tags.append("selected_mode_unchanged")
    if cf_details["selected_index"] != cf_details["oracle_index"]:
        tags.append("cf_selected_not_oracle")
    return tags


def _summarize_rows(rows: List[Dict], fields: Iterable[str]) -> Dict:
    by_field = defaultdict(list)
    by_transition = defaultdict(list)
    for row in rows:
        by_field[row["field"]].append(row)
        by_transition[row["transition"]].append(row)

    def summarize_group(group: List[Dict]) -> Dict:
        target_effects = [row["target_effect_z"] for row in group]
        success = [row["success"] for row in group]
        drift = [row["collateral_drift_z"] for row in group]
        token_match = [float(bool(row.get("cf_predicted_target_match", False))) for row in group]
        mode_changed = [
            float(row.get("base_selected_index") != row.get("cf_selected_index"))
            for row in group
        ]
        selected_oracle = [
            float(row.get("cf_selected_index") == row.get("cf_oracle_index"))
            for row in group
        ]
        return {
            "num_cases": int(len(group)),
            "success_rate": float(np.mean(success)) if success else float("nan"),
            "predicted_target_match_rate": float(np.mean(token_match)) if token_match else float("nan"),
            "selected_mode_changed_rate": float(np.mean(mode_changed)) if mode_changed else float("nan"),
            "cf_selected_oracle_rate": float(np.mean(selected_oracle)) if selected_oracle else float("nan"),
            "target_effect_z": summarize(target_effects),
            "collateral_drift_z": summarize(drift),
            "failure_tags": {
                tag: int(sum(tag in row["failure_tags"] for row in group))
                for tag in sorted({tag for row in group for tag in row["failure_tags"]})
            },
        }

    return {
        "num_rows": int(len(rows)),
        "fields": list(fields),
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else float("nan"),
        "target_effect_z": summarize([row["target_effect_z"] for row in rows]),
        "collateral_drift_z": summarize([row["collateral_drift_z"] for row in rows]),
        "by_field": {field: summarize_group(group) for field, group in sorted(by_field.items())},
        "by_transition": {
            key: summarize_group(group)
            for key, group in sorted(by_transition.items(), key=lambda item: (item[0].split(":")[0], item[0]))
        },
    }


def evaluate_failure_cases(
    model,
    qwen_provider,
    pairs: List[Tuple[Dict, Dict]],
    attr_dim: int,
    target_key: str,
    calibrator: GeometryTokenCalibrator,
    device: torch.device,
    success_margin_z: float,
    drift_warning_z: float,
    only_failures: bool,
    max_cases: int,
    counterfactual_adapter_scale: float = 0.0,
) -> tuple[List[Dict], Dict]:
    rows: List[Dict] = []
    total_pairs = 0

    for original, counterfactual in pairs:
        cf_meta = counterfactual.get("counterfactual", {})
        field_name = str(cf_meta.get("field", "unknown"))
        source_token = str(cf_meta.get("from", original.get("labels", {}).get(field_name, "unknown")))
        target_token = str(cf_meta.get("to", counterfactual.get("labels", {}).get(field_name, "unknown")))
        if field_name not in DEFAULT_MOBILITY_VOCAB.fields or target_token not in calibrator.stats[field_name]:
            continue
        if source_token not in calibrator.stats[field_name]:
            continue

        base = _make_prediction_details(model, qwen_provider, original, attr_dim, target_key, calibrator, device)
        cf = _make_prediction_details(
            model,
            qwen_provider,
            counterfactual,
            attr_dim,
            target_key,
            calibrator,
            device,
            counterfactual_adapter_scale=float(counterfactual_adapter_scale),
        )
        base_target_score = calibrator.score_token(field_name, target_token, base["selected_geom"])
        cf_target_score = calibrator.score_token(field_name, target_token, cf["selected_geom"])
        target_effect = float(base_target_score - cf_target_score)
        success = float(target_effect > float(success_margin_z))
        drift = collateral_drift_z(calibrator, field_name, original.get("labels", {}), base["selected_geom"], cf["selected_geom"])
        (
            metric_name,
            expected_center_delta,
            observed_delta,
            signed_metric_delta,
            signed_metric_delta_z,
        ) = calibrator.transition_metric_delta(
            field_name,
            source_token,
            target_token,
            base["selected_geom"],
            cf["selected_geom"],
        )

        tags = _failure_tags(success, drift, drift_warning_z, field_name, target_token, base, cf)
        row = {
            "source_sample_id": int(original.get("sample_id", 0)),
            "counterfactual_id": int(counterfactual.get("counterfactual_id", counterfactual.get("sample_id", 0))),
            "field": field_name,
            "from_token": source_token,
            "to_token": target_token,
            "transition": f"{field_name}:{source_token}->{target_token}",
            "success": success,
            "target_effect_z": target_effect,
            "collateral_drift_z": drift,
            "metric_name": metric_name,
            "expected_center_delta": float(expected_center_delta),
            "observed_metric_delta": float(observed_delta),
            "signed_target_metric_delta": float(signed_metric_delta),
            "signed_target_metric_delta_z": float(signed_metric_delta_z),
            "base_predicted_token": base["predicted_labels"].get(field_name),
            "cf_predicted_token": cf["predicted_labels"].get(field_name),
            "cf_predicted_target_match": bool(cf["predicted_labels"].get(field_name) == target_token),
            "base_selected_index": base["selected_index"],
            "cf_selected_index": cf["selected_index"],
            "base_oracle_index": base["oracle_index"],
            "cf_oracle_index": cf["oracle_index"],
            "base_selected_ade_m": base["selected_ade_m"],
            "cf_selected_ade_m": cf["selected_ade_m"],
            "base_oracle_ade_m": base["oracle_ade_m"],
            "cf_oracle_ade_m": cf["oracle_ade_m"],
            "base_instruction": base["instruction"],
            "cf_instruction": cf["instruction"],
            "base_labels": base["labels"],
            "cf_labels": cf["labels"],
            "base_predicted_labels": base["predicted_labels"],
            "cf_predicted_labels": cf["predicted_labels"],
            "base_candidate_ade_m": base["candidate_ade_m"],
            "cf_candidate_ade_m": cf["candidate_ade_m"],
            "base_candidate_logits": base["candidate_logits"],
            "cf_candidate_logits": cf["candidate_logits"],
            "failure_tags": tags,
        }
        total_pairs += 1
        if (not only_failures) or tags:
            rows.append(row)
        if max_cases > 0 and len(rows) >= max_cases:
            break

    summary = _summarize_rows(rows, fields=DEFAULT_MOBILITY_VOCAB.fields)
    summary.update(
        {
            "num_pairs_scanned": int(total_pairs),
            "only_failures": bool(only_failures),
            "success_margin_z": float(success_margin_z),
            "drift_warning_z": float(drift_warning_z),
            "counterfactual_adapter_scale": float(counterfactual_adapter_scale),
        }
    )
    return rows, summary


def _flatten_for_csv(row: Dict) -> Dict:
    flat = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    return flat


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint, device, token_calibration_path=args.token_calibration_path)
    saved_args = checkpoint.get("args", {})
    attr_dim = int(saved_args.get("attr_dim", 6))
    if args.qwen_planner_checkpoint is None:
        saved_qwen_checkpoint = saved_args.get("qwen_planner_checkpoint")
        if saved_qwen_checkpoint:
            args.qwen_planner_checkpoint = Path(saved_qwen_checkpoint)
    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device) if args.qwen_planner_checkpoint else None

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
    )
    pairs, transition_filter_stats = group_counterfactual_pairs(
        records,
        fields=set(args.fields),
        max_pairs=args.max_pairs,
        transition_table=transition_table,
        pair_sampling=args.pair_sampling,
    )
    if not pairs:
        raise ValueError("No original/counterfactual pairs found.")

    if args.token_calibration_path is not None and args.token_calibration_path.exists():
        calibrator = GeometryTokenCalibrator.from_json(str(args.token_calibration_path))
        calibration_source = str(args.token_calibration_path)
    else:
        calibrator = build_pair_calibrator(pairs, args.target_key, model.decoder.target_len, device)
        calibration_source = "paired target trajectories"

    rows, summary = evaluate_failure_cases(
        model=model,
        qwen_provider=qwen_provider,
        pairs=pairs,
        attr_dim=attr_dim,
        target_key=args.target_key,
        calibrator=calibrator,
        device=device,
        success_margin_z=args.success_margin_z,
        drift_warning_z=args.drift_warning_z,
        only_failures=args.only_failures,
        max_cases=args.max_cases,
        counterfactual_adapter_scale=args.counterfactual_adapter_scale,
    )
    summary.update(
        {
            "dataset": args.dataset,
            "input": str(jsonl_path),
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "target_key": args.target_key,
            "requested_fields": args.fields,
            "qwen_planner_checkpoint": str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None,
            "qwen_semantic_conditioning": qwen_provider is not None,
            "counterfactual_adapter_scale": float(args.counterfactual_adapter_scale),
            "calibration_source": calibration_source,
            "transition_filter": {
                "diagnostics_path": str(args.transition_diagnostics_path) if args.transition_diagnostics_path else None,
                "enabled": bool(transition_table),
                "min_success_rate": float(args.transition_min_success_rate),
                "min_effect_z": float(args.transition_min_effect_z),
                "min_token_match_rate": float(args.transition_min_token_match_rate),
                "min_pairs": int(args.transition_min_pairs),
                "max_collateral_z": float(args.transition_max_collateral_z),
                "stats": transition_filter_stats,
            },
        }
    )

    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_jsonl = output_dir / "failure_cases.jsonl"
    with cases_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cases_csv = output_dir / "failure_cases.csv"
    flat_rows = [_flatten_for_csv(row) for row in rows]
    if flat_rows:
        with cases_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    summary_path = output_dir / "summary.json"
    summary.update({"cases_jsonl": str(cases_jsonl), "cases_csv": str(cases_csv)})
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "summary_path": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
