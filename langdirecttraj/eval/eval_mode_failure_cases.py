from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from langdirecttraj.data.instruction_dataset import InstructionTrajectoryDataset, collate_instruction_batch
from langdirecttraj.eval.eval_anchor_decoder import build_model, load_checkpoint, prepare_eval_batch, summarize
from langdirecttraj.eval.eval_multi_candidate_diagnostics import candidate_errors_m
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    move_batch,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

GEOMETRY_FEATURE_NAMES = [
    "mean_ref_dist",
    "max_ref_dist",
    "final_ref_dist",
    "length_ratio",
    "progress",
    "curvature",
    "residual_norm_mean",
    "residual_norm_max",
    "raw_tangent_mean",
    "raw_normal_mean",
    "gate_mean",
    "gate_std",
    "ref_token_match",
    "curvature_token_match",
    "progress_token_match",
    "direction_token_match",
]


def _decode_tokens(token_ids: torch.Tensor) -> Dict[str, str]:
    return DEFAULT_MOBILITY_VOCAB.decode([int(x) for x in token_ids.detach().cpu().tolist()])


def _feature_dict(values: np.ndarray) -> Dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(GEOMETRY_FEATURE_NAMES[: len(values)])}


def _diff_dict(mode_values: np.ndarray, selected_values: np.ndarray) -> Dict[str, float]:
    return {
        f"delta_{name}": float(mode_values[i] - selected_values[i])
        for i, name in enumerate(GEOMETRY_FEATURE_NAMES[: min(len(mode_values), len(selected_values))])
    }


def _tensor_value(tensor: torch.Tensor | None, batch_idx: int, mode_idx: int) -> float | None:
    if tensor is None:
        return None
    value = tensor
    if value.ndim == 0:
        return float(value.detach().cpu().item())
    if value.ndim == 1:
        return float(value[mode_idx].detach().cpu().item())
    return float(value[batch_idx, mode_idx].detach().cpu().item())


def _maybe_delta(mode_value: float | None, selected_value: float | None) -> float | None:
    if mode_value is None or selected_value is None:
        return None
    return float(mode_value - selected_value)


def _case_summary(rows: List[Dict]) -> Dict:
    if not rows:
        return {"num_cases": 0}
    selected_ade = [row["selected_ade_m"] for row in rows]
    mode_ade = [row["mode_ade_m"] for row in rows]
    selected_logit = [row["selected_logit"] for row in rows]
    mode_logit = [row["mode_logit"] for row in rows]
    oracle_count = sum(1 for row in rows if row["mode_is_oracle"])
    component_summaries = {}
    for key in [
        "base_logit_delta_mode_minus_selected",
        "delta_logit_delta_mode_minus_selected",
        "global_bias_delta_mode_minus_selected",
        "pre_bias_logit_delta_mode_minus_selected",
    ]:
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            component_summaries[key] = summarize(values)
    return {
        "num_cases": len(rows),
        "mode_is_oracle_count": int(oracle_count),
        "mode_is_near_oracle_count": int(sum(1 for row in rows if row["mode_is_near_oracle"])),
        "selected_ade_m": summarize(selected_ade),
        "mode_ade_m": summarize(mode_ade),
        "ade_delta_mode_minus_selected_m": summarize([m - s for m, s in zip(mode_ade, selected_ade)]),
        "logit_delta_mode_minus_selected": summarize([m - s for m, s in zip(mode_logit, selected_logit)]),
        "selected_index_counts": {
            str(idx): int(sum(1 for row in rows if int(row["selected_index"]) == idx))
            for idx in sorted({int(row["selected_index"]) for row in rows})
        },
        "score_component_deltas_mode_minus_selected": component_summaries,
    }


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    attr_dim: int,
    mode_index: int,
    near_oracle_margin_m: float,
    max_cases: int,
    qwen_provider=None,
) -> tuple[List[Dict], Dict]:
    rows: List[Dict] = []
    total_seen = 0
    total_mode_oracle = 0
    total_mode_near = 0
    num_candidates = int(getattr(model.decoder, "num_candidates", 1))
    if mode_index < 0 or mode_index >= num_candidates:
        raise ValueError(f"mode_index must be in [0, {num_candidates - 1}], got {mode_index}.")

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            target, reference, route_scales = prepare_eval_batch(batch, model.decoder.target_len)
            route_scale = route_scales[:, 0]
            attr_cond = make_attr_cond(batch, attr_dim, device)
            semantic_plan_tokens = extract_qwen_semantic_tokens(qwen_provider, batch["instruction"], device)
            pred_out = model(
                reference,
                batch["mobility_tokens"],
                attr_cond,
                semantic_plan_tokens=semantic_plan_tokens,
                return_aux=True,
            )
            pred, aux = pred_out if isinstance(pred_out, tuple) else (pred_out, {})
            if pred.ndim == 3:
                pred = pred[:, None]
            logits = aux.get("candidate_logits") if isinstance(aux, dict) else None
            geometry = aux.get("reranker_geometry_features") if isinstance(aux, dict) else None
            token_gate = aux.get("reranker_token_gate") if isinstance(aux, dict) else None
            base_logits = aux.get("reranker_base_logits") if isinstance(aux, dict) else None
            delta_logits = aux.get("reranker_delta_logits") if isinstance(aux, dict) else None
            residual_alpha = aux.get("reranker_residual_alpha") if isinstance(aux, dict) else None
            mode_logit_bias = getattr(model.decoder, "mode_logit_bias", None)
            if logits is None or geometry is None:
                continue
            ade, fde = candidate_errors_m(pred, target, route_scale)
            selected_idx = logits.argmax(dim=1)
            oracle_idx = ade.argmin(dim=1)
            best_ade = ade.gather(1, oracle_idx.view(-1, 1)).squeeze(1)
            mode_is_oracle = oracle_idx == int(mode_index)
            mode_is_near = ade[:, mode_index] <= (best_ade + float(near_oracle_margin_m))
            failure_mask = mode_is_near & (selected_idx != int(mode_index))

            total_seen += int(pred.shape[0])
            total_mode_oracle += int(mode_is_oracle.sum().item())
            total_mode_near += int(mode_is_near.sum().item())

            for i in torch.nonzero(failure_mask, as_tuple=False).flatten().tolist():
                selected = int(selected_idx[i].item())
                oracle = int(oracle_idx[i].item())
                mode_values = geometry[i, mode_index].detach().cpu().numpy().astype(np.float64)
                selected_values = geometry[i, selected].detach().cpu().numpy().astype(np.float64)
                selected_base = _tensor_value(base_logits, i, selected)
                mode_base = _tensor_value(base_logits, i, mode_index)
                oracle_base = _tensor_value(base_logits, i, oracle)
                selected_delta = _tensor_value(delta_logits, i, selected)
                mode_delta = _tensor_value(delta_logits, i, mode_index)
                oracle_delta = _tensor_value(delta_logits, i, oracle)
                selected_bias = _tensor_value(mode_logit_bias, i, selected)
                mode_bias = _tensor_value(mode_logit_bias, i, mode_index)
                oracle_bias = _tensor_value(mode_logit_bias, i, oracle)
                alpha_value = _tensor_value(residual_alpha, i, mode_index)
                selected_pre_bias = (
                    None
                    if selected_base is None or selected_delta is None or alpha_value is None
                    else float(selected_base + alpha_value * selected_delta)
                )
                mode_pre_bias = (
                    None
                    if mode_base is None or mode_delta is None or alpha_value is None
                    else float(mode_base + alpha_value * mode_delta)
                )
                oracle_pre_bias = (
                    None
                    if oracle_base is None or oracle_delta is None or alpha_value is None
                    else float(oracle_base + alpha_value * oracle_delta)
                )
                row = {
                    "sample_id": int(batch["sample_id"][i].item()),
                    "dataset": str(batch.get("dataset", "")),
                    "instruction": str(batch["instruction"][i]),
                    "labels": batch["labels"][i],
                    "decoded_tokens": _decode_tokens(batch["mobility_tokens"][i]),
                    "mode_index": int(mode_index),
                    "selected_index": selected,
                    "oracle_index": oracle,
                    "mode_is_oracle": bool(mode_is_oracle[i].item()),
                    "mode_is_near_oracle": bool(mode_is_near[i].item()),
                    "selected_ade_m": float(ade[i, selected].item()),
                    "mode_ade_m": float(ade[i, mode_index].item()),
                    "oracle_ade_m": float(best_ade[i].item()),
                    "selected_fde_m": float(fde[i, selected].item()),
                    "mode_fde_m": float(fde[i, mode_index].item()),
                    "selected_logit": float(logits[i, selected].item()),
                    "mode_logit": float(logits[i, mode_index].item()),
                    "oracle_logit": float(logits[i, oracle].item()),
                    "selected_base_logit": selected_base,
                    "mode_base_logit": mode_base,
                    "oracle_base_logit": oracle_base,
                    "selected_delta_logit": selected_delta,
                    "mode_delta_logit": mode_delta,
                    "oracle_delta_logit": oracle_delta,
                    "residual_listwise_alpha": alpha_value,
                    "selected_pre_bias_logit": selected_pre_bias,
                    "mode_pre_bias_logit": mode_pre_bias,
                    "oracle_pre_bias_logit": oracle_pre_bias,
                    "selected_global_mode_bias": selected_bias,
                    "mode_global_mode_bias": mode_bias,
                    "oracle_global_mode_bias": oracle_bias,
                    "selected_prob": float(torch.softmax(logits[i], dim=0)[selected].item()),
                    "mode_prob": float(torch.softmax(logits[i], dim=0)[mode_index].item()),
                    "selected_token_gate": (
                        float(token_gate[i, selected].item()) if token_gate is not None else None
                    ),
                    "mode_token_gate": (
                        float(token_gate[i, mode_index].item()) if token_gate is not None else None
                    ),
                    "mode_rank_by_logit": int((logits[i] > logits[i, mode_index]).sum().item() + 1),
                    "mode_rank_by_ade": int((ade[i] < ade[i, mode_index]).sum().item() + 1),
                    "ade_delta_mode_minus_selected_m": float(ade[i, mode_index].item() - ade[i, selected].item()),
                    "logit_delta_mode_minus_selected": float(logits[i, mode_index].item() - logits[i, selected].item()),
                    "base_logit_delta_mode_minus_selected": _maybe_delta(mode_base, selected_base),
                    "delta_logit_delta_mode_minus_selected": _maybe_delta(mode_delta, selected_delta),
                    "pre_bias_logit_delta_mode_minus_selected": _maybe_delta(mode_pre_bias, selected_pre_bias),
                    "global_bias_delta_mode_minus_selected": _maybe_delta(mode_bias, selected_bias),
                    "geometry_mode": _feature_dict(mode_values),
                    "geometry_selected": _feature_dict(selected_values),
                    "geometry_delta_mode_minus_selected": _diff_dict(mode_values, selected_values),
                }
                rows.append(row)
                if len(rows) >= int(max_cases):
                    break
            if len(rows) >= int(max_cases):
                break

    summary = _case_summary(rows)
    summary.update(
        {
            "num_evaluated": int(total_seen),
            "num_candidates": int(num_candidates),
            "mode_index": int(mode_index),
            "near_oracle_margin_m": float(near_oracle_margin_m),
            "total_mode_oracle": int(total_mode_oracle),
            "total_mode_near_oracle": int(total_mode_near),
        }
    )
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export failure cases where a candidate mode is near-oracle but not selected.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "mode_failure_cases")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--max-cases", type=int, default=80)
    parser.add_argument("--mode-index", type=int, default=3)
    parser.add_argument("--near-oracle-margin-m", type=float, default=5.0)
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--token-calibration-path", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint, device, token_calibration_path=args.token_calibration_path)
    saved_args = checkpoint.get("args", {})
    attr_dim = int(saved_args.get("attr_dim", 6))
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
    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device) if args.qwen_planner_checkpoint else None
    rows, summary = evaluate(
        model,
        loader,
        device,
        attr_dim,
        mode_index=args.mode_index,
        near_oracle_margin_m=args.near_oracle_margin_m,
        max_cases=args.max_cases,
        qwen_provider=qwen_provider,
    )
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "input": str(jsonl_path),
            "target_key": args.target_key,
            "qwen_planner_checkpoint": str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path_out = args.output_dir / f"mode_{args.mode_index}_failure_cases.jsonl"
    with jsonl_path_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = args.output_dir / f"mode_{args.mode_index}_failure_cases.csv"
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        flat.update({f"mode_{k}": v for k, v in row["geometry_mode"].items()})
        flat.update({f"selected_{k}": v for k, v in row["geometry_selected"].items()})
        flat.update(row["geometry_delta_mode_minus_selected"])
        flat_rows.append(flat)
    if flat_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)
    summary_path = args.output_dir / "summary.json"
    summary.update({"cases_jsonl": str(jsonl_path_out), "cases_csv": str(csv_path)})
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "summary_path": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
