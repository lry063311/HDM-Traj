from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from langdirecttraj.data.instruction_dataset import (
    InstructionTrajectoryDataset,
    collate_instruction_batch,
)
from langdirecttraj.eval.eval_anchor_decoder import (
    build_model,
    load_checkpoint,
    prepare_eval_batch,
)
from langdirecttraj.eval.instruction_geometry import (
    contract_labels_from_tensors,
    geometry_values,
    macro_f1,
    tensor_dict_to_rows,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    move_batch,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    attr_dim: int,
    field: str,
    dataset: str,
    qwen_provider=None,
    counterfactual_adapter_scale: float = 0.0,
) -> dict[str, Any]:
    if field not in DEFAULT_MOBILITY_VOCAB.fields:
        raise ValueError(f"Unsupported mobility token field: {field}")

    labels_all = list(DEFAULT_MOBILITY_VOCAB.token_to_id[field].keys())
    selected_true: list[str] = []
    selected_pred: list[str] = []
    label_oracle_pred: list[str] = []
    true_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    label_oracle_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    has_true_values: list[float] = []
    has_true_by_label: dict[str, list[float]] = defaultdict(list)
    missed_available_true = 0
    no_true_candidate = 0
    cases: list[dict[str, Any]] = []

    with torch.no_grad():
        for sample_idx, raw_batch in enumerate(loader):
            batch = move_batch(raw_batch, device)
            target, reference, route_scales = prepare_eval_batch(batch, model.decoder.target_len)
            route_scale = route_scales[:, 0]
            attr_cond = make_attr_cond(batch, attr_dim, device)
            semantic_plan_tokens = extract_qwen_semantic_tokens(
                qwen_provider,
                batch["instruction"],
                device,
            )
            counterfactual_enabled = [
                bool(item.get("enabled")) if isinstance(item, dict) else False
                for item in batch.get("counterfactual", [])
            ]
            counterfactual_mode = any(counterfactual_enabled) and float(counterfactual_adapter_scale) != 0.0
            counterfactual_field_ids = None
            if counterfactual_mode:
                field_ids: list[int] = []
                for item in batch.get("counterfactual", []):
                    field_name = str(item.get("field", "")) if isinstance(item, dict) else ""
                    field_ids.append(
                        DEFAULT_MOBILITY_VOCAB.fields.index(field_name)
                        if field_name in DEFAULT_MOBILITY_VOCAB.fields
                        else 0
                    )
                counterfactual_field_ids = torch.tensor(field_ids, device=device, dtype=torch.long)
            pred_out = model(
                reference,
                batch["mobility_tokens"],
                attr_cond,
                semantic_plan_tokens=semantic_plan_tokens,
                counterfactual_mode=counterfactual_mode,
                counterfactual_adapter_scale=float(counterfactual_adapter_scale),
                counterfactual_field_ids=counterfactual_field_ids,
                route_scales=route_scale,
                return_aux=True,
            )
            if isinstance(pred_out, tuple):
                candidates, aux = pred_out
            else:
                candidates, aux = pred_out, {}
            if candidates.ndim == 3:
                candidates = candidates[:, None]

            logits = aux.get("candidate_logits") if isinstance(aux, dict) else None
            if torch.is_tensor(logits):
                selected_idx = logits.argmax(dim=1)
            else:
                selected_idx = torch.zeros(candidates.shape[0], dtype=torch.long, device=device)

            batch_size, num_candidates, target_len, dim = candidates.shape
            flat_candidates = candidates.reshape(batch_size * num_candidates, target_len, dim)
            flat_reference = reference[:, None].expand_as(candidates).reshape_as(flat_candidates)
            flat_scales = route_scale[:, None].expand(-1, num_candidates).reshape(-1)
            decoded_flat = contract_labels_from_tensors(
                flat_candidates.float(),
                flat_reference.float(),
                flat_scales.float(),
                dataset=dataset,
            )
            decoded = np.asarray([str(row[field]) for row in decoded_flat], dtype=object).reshape(
                batch_size,
                num_candidates,
            )
            geom_rows = tensor_dict_to_rows(
                geometry_values(flat_candidates.float(), flat_reference.float(), flat_scales.float())
            )
            if field == "reference_adherence":
                geom_values = [
                    float(row.get("mean_reference_distance_m", float("nan"))) for row in geom_rows
                ]
            else:
                geom_values = []

            errors = torch.linalg.norm(candidates - target[:, None], dim=-1).mean(dim=-1)
            for row_idx in range(batch_size):
                true_label = str(batch["labels"][row_idx].get(field, "unknown"))
                row_labels = [str(item) for item in decoded[row_idx].tolist()]
                row_selected_idx = int(selected_idx[row_idx].detach().cpu().item())
                selected_label = row_labels[row_selected_idx]
                true_counts[true_label] += 1
                selected_counts[selected_label] += 1
                selected_true.append(true_label)
                selected_pred.append(selected_label)
                for label in row_labels:
                    candidate_counts[label] += 1

                match_indices = [idx for idx, label in enumerate(row_labels) if label == true_label]
                has_true = bool(match_indices)
                has_true_values.append(float(has_true))
                has_true_by_label[true_label].append(float(has_true))
                if has_true:
                    best_match = min(
                        match_indices,
                        key=lambda idx: float(errors[row_idx, idx].detach().cpu().item()),
                    )
                    oracle_label = row_labels[best_match]
                    if selected_label != true_label:
                        missed_available_true += 1
                else:
                    oracle_label = selected_label
                    no_true_candidate += 1
                label_oracle_counts[oracle_label] += 1
                label_oracle_pred.append(oracle_label)

                case: dict[str, Any] = {
                    "sample_idx": int(sample_idx * batch_size + row_idx),
                    "instruction": str(batch["instruction"][row_idx]),
                    "true": true_label,
                    "selected_idx": row_selected_idx,
                    "selected": selected_label,
                    "has_true_candidate": has_true,
                    "candidate_labels": row_labels,
                }
                if torch.is_tensor(logits):
                    case["candidate_logits"] = [
                        float(x)
                        for x in logits[row_idx, :num_candidates].detach().cpu().float().tolist()
                    ]
                if geom_values:
                    start = row_idx * num_candidates
                    case["candidate_mean_reference_distance_m"] = geom_values[
                        start : start + num_candidates
                    ]
                cases.append(case)

    selected_metrics = macro_f1(selected_true, selected_pred, labels_all, include_empty=False)
    label_oracle_metrics = macro_f1(selected_true, label_oracle_pred, labels_all, include_empty=False)
    return {
        "num_samples": int(len(selected_true)),
        "num_candidates": int(getattr(model.decoder, "num_candidates", 1)),
        "field": field,
        "counterfactual_adapter_scale": float(counterfactual_adapter_scale),
        "selected_f1": float(selected_metrics["macro_f1"]),
        "candidate_label_oracle_f1": float(label_oracle_metrics["macro_f1"]),
        "selected_metrics": selected_metrics,
        "candidate_label_oracle_metrics": label_oracle_metrics,
        "candidate_has_true_rate": (
            float(np.mean(has_true_values)) if has_true_values else float("nan")
        ),
        "candidate_has_true_rate_by_label": {
            label: float(np.mean(values)) if values else float("nan")
            for label, values in sorted(has_true_by_label.items())
        },
        "true_counts": _counter_to_dict(true_counts),
        "selected_counts": _counter_to_dict(selected_counts),
        "candidate_counts_total": _counter_to_dict(candidate_counts),
        "label_oracle_counts": _counter_to_dict(label_oracle_counts),
        "failure_count_no_true_candidate": int(no_true_candidate),
        "selected_missed_available_true_count": int(missed_available_true),
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether a multi-candidate pool covers a target mobility token field."
    )
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "token_candidate_pool")
    parser.add_argument("--field", choices=DEFAULT_MOBILITY_VOCAB.fields, default="reference_adherence")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument(
        "--counterfactual-field-filter",
        choices=DEFAULT_MOBILITY_VOCAB.fields,
        default=None,
        help=(
            "Optional diagnostic filter for counterfactual JSONL records. When set, records are "
            "filtered by record['counterfactual']['field'] before --max-samples is applied."
        ),
    )
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--token-calibration-path", type=Path, default=None)
    parser.add_argument("--counterfactual-adapter-scale", type=float, default=0.0)
    parser.add_argument(
        "--direction-control-strength-override",
        type=float,
        default=None,
        help="Temporarily override decoder direction control strength for coverage diagnostics.",
    )
    parser.add_argument(
        "--reference-control-strength-override",
        type=float,
        default=None,
        help="Temporarily override decoder reference-adherence control strength for coverage diagnostics.",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint, device, token_calibration_path=args.token_calibration_path)
    decoder = getattr(model, "decoder", None)
    if args.direction_control_strength_override is not None:
        if decoder is None or not hasattr(decoder, "set_direction_control_strength"):
            raise ValueError("--direction-control-strength-override requires a compatible decoder.")
        decoder.set_direction_control_strength(float(args.direction_control_strength_override))
    if args.reference_control_strength_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_control_strength"):
            raise ValueError("--reference-control-strength-override requires a compatible decoder.")
        decoder.set_reference_control_strength(float(args.reference_control_strength_override))
    if args.reference_positive_candidate_strength_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_positive_candidate_strength"):
            raise ValueError("--reference-positive-candidate-strength-override requires a compatible decoder.")
        decoder.set_reference_positive_candidate_strength(
            float(args.reference_positive_candidate_strength_override)
        )
    if args.reference_positive_candidate_scope_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_positive_candidate_scope"):
            raise ValueError("--reference-positive-candidate-scope-override requires a compatible decoder.")
        decoder.set_reference_positive_candidate_scope(
            str(args.reference_positive_candidate_scope_override)
        )
    if args.reference_distance_candidate_strength_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_distance_candidate_strength"):
            raise ValueError("--reference-distance-candidate-strength-override requires a compatible decoder.")
        decoder.set_reference_distance_candidate_strength(
            float(args.reference_distance_candidate_strength_override)
        )
    if args.reference_distance_candidate_scope_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_distance_candidate_scope"):
            raise ValueError("--reference-distance-candidate-scope-override requires a compatible decoder.")
        decoder.set_reference_distance_candidate_scope(
            str(args.reference_distance_candidate_scope_override)
        )
    saved_args = checkpoint.get("args", {})
    attr_dim = int(saved_args.get("attr_dim", 6))
    dataset = InstructionTrajectoryDataset(
        args.input,
        max_samples=None if args.counterfactual_field_filter else args.max_samples,
        target_key=args.target_key,
    )
    if args.counterfactual_field_filter:
        filtered_records = [
            record
            for record in dataset.records
            if str(record.get("counterfactual", {}).get("field", "")) == str(args.counterfactual_field_filter)
        ]
        if args.max_samples is not None and int(args.max_samples) > 0:
            filtered_records = filtered_records[: int(args.max_samples)]
        if not filtered_records:
            raise ValueError(
                "No records remain after --counterfactual-field-filter="
                f"{args.counterfactual_field_filter}"
            )
        dataset.records = filtered_records
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_instruction_batch,
    )
    qwen_provider = (
        build_qwen_provider(args.qwen_planner_checkpoint, args, device)
        if args.qwen_planner_checkpoint
        else None
    )
    metrics = evaluate(
        model,
        loader,
        device,
        attr_dim,
        field=args.field,
        dataset=args.dataset,
        qwen_provider=qwen_provider,
        counterfactual_adapter_scale=float(args.counterfactual_adapter_scale),
    )
    cases = metrics.pop("cases")
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "input": str(args.input),
            "target_key": args.target_key,
            "counterfactual_field_filter": args.counterfactual_field_filter,
            "qwen_planner_checkpoint": (
                str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None
            ),
            "direction_control_strength_override": args.direction_control_strength_override,
            "reference_control_strength_override": args.reference_control_strength_override,
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
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    cases_path = args.output_dir / "cases.jsonl"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    cases_path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {**metrics, "metrics_path": str(metrics_path), "cases_path": str(cases_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
