from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from langdirecttraj.data.instruction_dataset import InstructionTrajectoryDataset, collate_instruction_batch
from langdirecttraj.eval.eval_anchor_decoder import build_model, load_checkpoint, prepare_eval_batch, summarize
from langdirecttraj.eval.eval_multi_candidate_diagnostics import candidate_errors_m
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    move_batch,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _safe_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _ratio(num: int, den: int) -> float:
    return float(num) / float(max(den, 1))


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    attr_dim: int,
    near_oracle_margin_m: float,
    qwen_provider=None,
) -> Dict:
    num_candidates = int(getattr(model.decoder, "num_candidates", 1))
    selected_counts = np.zeros(num_candidates, dtype=np.int64)
    oracle_counts = np.zeros(num_candidates, dtype=np.int64)
    near_counts = np.zeros(num_candidates, dtype=np.int64)
    missed_oracle_counts = np.zeros(num_candidates, dtype=np.int64)
    per_mode_ade: List[List[float]] = [[] for _ in range(num_candidates)]
    per_mode_logits: List[List[float]] = [[] for _ in range(num_candidates)]
    selected_ade: List[float] = []
    oracle_ade: List[float] = []

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
            if isinstance(pred_out, tuple):
                pred, aux = pred_out
            else:
                pred, aux = pred_out, {}
            if pred.ndim == 3:
                pred = pred[:, None]
            logits = aux.get("candidate_logits") if isinstance(aux, dict) else None
            if logits is None:
                logits = torch.zeros(pred.shape[:2], dtype=pred.dtype, device=pred.device)
            ade, _ = candidate_errors_m(pred, target, route_scale)
            selected_idx = logits.argmax(dim=1)
            oracle_idx = ade.argmin(dim=1)
            batch_idx = torch.arange(pred.shape[0], device=pred.device)
            best_ade = ade[batch_idx, oracle_idx]
            near_mask = ade <= (best_ade[:, None] + float(near_oracle_margin_m))

            selected_np = selected_idx.detach().cpu().numpy()
            oracle_np = oracle_idx.detach().cpu().numpy()
            near_np = near_mask.detach().cpu().numpy()
            ade_np = ade.detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            for i in range(pred.shape[0]):
                selected_counts[int(selected_np[i])] += 1
                oracle_counts[int(oracle_np[i])] += 1
                if int(selected_np[i]) != int(oracle_np[i]):
                    missed_oracle_counts[int(oracle_np[i])] += 1
                for k in range(num_candidates):
                    if bool(near_np[i, k]):
                        near_counts[k] += 1
                    per_mode_ade[k].append(float(ade_np[i, k]))
                    per_mode_logits[k].append(float(logits_np[i, k]))
            selected_ade.extend(ade[batch_idx, selected_idx].detach().cpu().tolist())
            oracle_ade.extend(best_ade.detach().cpu().tolist())

    total = int(selected_counts.sum())
    per_mode = {}
    for k in range(num_candidates):
        selected_rate = _ratio(int(selected_counts[k]), total)
        oracle_rate = _ratio(int(oracle_counts[k]), total)
        near_oracle_rate = _ratio(int(near_counts[k]), total)
        per_mode[str(k)] = {
            "selected_count": int(selected_counts[k]),
            "oracle_count": int(oracle_counts[k]),
            "near_oracle_count": int(near_counts[k]),
            "missed_when_oracle_count": int(missed_oracle_counts[k]),
            "selected_rate": selected_rate,
            "oracle_rate": oracle_rate,
            "near_oracle_rate": near_oracle_rate,
            "coverage_gap": near_oracle_rate - selected_rate,
            "oracle_gap": oracle_rate - selected_rate,
            "missed_when_oracle_rate": _ratio(int(missed_oracle_counts[k]), int(oracle_counts[k])),
            "mean_ade_m": _safe_mean(per_mode_ade[k]),
            "mean_logit": _safe_mean(per_mode_logits[k]),
        }

    return {
        "num_samples": total,
        "num_candidates": num_candidates,
        "near_oracle_margin_m": float(near_oracle_margin_m),
        "selected_ade_m": summarize(selected_ade),
        "oracle_ade_m": summarize(oracle_ade),
        "selected_oracle_gap_m": float(np.mean(selected_ade) - np.mean(oracle_ade)) if selected_ade else float("nan"),
        "per_mode": per_mode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose scorer mode bias and missed near-oracle candidate modes.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "mode_bias_diagnostics")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--near-oracle-margin-m", type=float, default=5.0)
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
    metrics = evaluate(
        model,
        loader,
        device,
        attr_dim,
        near_oracle_margin_m=args.near_oracle_margin_m,
        qwen_provider=qwen_provider,
    )
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "input": str(jsonl_path),
            "target_key": args.target_key,
            "qwen_planner_checkpoint": str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "metrics.json"
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**metrics, "metrics_path": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
