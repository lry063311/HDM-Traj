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
from langdirecttraj.train.control_losses import MODE_ANCHOR_FEATURES, default_mode_anchor_targets, mode_geometry_descriptors
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    move_batch,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _ratio(num: int, den: int) -> float:
    return float(num) / float(max(den, 1))


def _safe_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


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
    per_mode: Dict[str, Dict[str, List[float]]] = {
        str(k): {
            "ade_m": [],
            "logit": [],
            **{name: [] for name in MODE_ANCHOR_FEATURES},
        }
        for k in range(num_candidates)
    }
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
            descriptors = mode_geometry_descriptors(pred, reference)

            selected_counts += np.bincount(
                selected_idx.detach().cpu().numpy(),
                minlength=num_candidates,
            )[:num_candidates]
            oracle_counts += np.bincount(
                oracle_idx.detach().cpu().numpy(),
                minlength=num_candidates,
            )[:num_candidates]
            near_counts += near_mask.detach().cpu().numpy().sum(axis=0).astype(np.int64)
            selected_ade.extend(ade[batch_idx, selected_idx].detach().cpu().tolist())
            oracle_ade.extend(best_ade.detach().cpu().tolist())

            ade_np = ade.detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            desc_np = descriptors.detach().cpu().numpy()
            for mode_idx in range(num_candidates):
                bucket = per_mode[str(mode_idx)]
                bucket["ade_m"].extend(ade_np[:, mode_idx].astype(float).tolist())
                bucket["logit"].extend(logits_np[:, mode_idx].astype(float).tolist())
                for feature_idx, feature_name in enumerate(MODE_ANCHOR_FEATURES):
                    bucket[feature_name].extend(desc_np[:, mode_idx, feature_idx].astype(float).tolist())

    total = int(selected_counts.sum())
    anchors, scales, feature_names = default_mode_anchor_targets(num_candidates, device=torch.device("cpu"))
    anchors_np = anchors.detach().cpu().numpy()
    scales_np = scales.detach().cpu().numpy()
    mode_rows = []
    mode_summary = {}
    for mode_idx in range(num_candidates):
        bucket = per_mode[str(mode_idx)]
        selected_rate = _ratio(int(selected_counts[mode_idx]), total)
        oracle_rate = _ratio(int(oracle_counts[mode_idx]), total)
        near_rate = _ratio(int(near_counts[mode_idx]), total)
        row = {
            "mode": int(mode_idx),
            "selected_count": int(selected_counts[mode_idx]),
            "oracle_count": int(oracle_counts[mode_idx]),
            "near_oracle_count": int(near_counts[mode_idx]),
            "selected_rate": selected_rate,
            "oracle_rate": oracle_rate,
            "near_oracle_rate": near_rate,
            "coverage_gap": near_rate - selected_rate,
            "oracle_gap": oracle_rate - selected_rate,
            "mean_ade_m": _safe_mean(bucket["ade_m"]),
            "mean_logit": _safe_mean(bucket["logit"]),
        }
        anchor_payload = {}
        for feature_idx, feature_name in enumerate(feature_names):
            mean_value = _safe_mean(bucket[feature_name])
            anchor_value = float(anchors_np[mode_idx, feature_idx])
            scale_value = float(scales_np[feature_idx])
            row[feature_name] = mean_value
            row[f"{feature_name}_anchor"] = anchor_value
            row[f"{feature_name}_anchor_z"] = (mean_value - anchor_value) / max(scale_value, 1e-6)
            anchor_payload[feature_name] = {
                "mean": mean_value,
                "default_anchor": anchor_value,
                "anchor_z": row[f"{feature_name}_anchor_z"],
            }
        mode_rows.append(row)
        mode_summary[str(mode_idx)] = {**row, "features": anchor_payload}

    return {
        "num_samples": total,
        "num_candidates": num_candidates,
        "near_oracle_margin_m": float(near_oracle_margin_m),
        "mode_priors": {
            "selected_rate": selected_counts.astype(float).tolist()
            if total <= 0
            else (selected_counts.astype(float) / float(total)).tolist(),
            "oracle_rate": oracle_counts.astype(float).tolist()
            if total <= 0
            else (oracle_counts.astype(float) / float(total)).tolist(),
            "near_oracle_rate": near_counts.astype(float).tolist()
            if total <= 0
            else (near_counts.astype(float) / float(total)).tolist(),
        },
        "selected_ade_m": summarize(selected_ade),
        "oracle_ade_m": summarize(oracle_ade),
        "selected_oracle_gap_m": float(np.mean(selected_ade) - np.mean(oracle_ade)) if selected_ade else float("nan"),
        "feature_names": feature_names,
        "default_anchor_scales": scales_np.astype(float).tolist(),
        "per_mode": mode_summary,
        "_rows": mode_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose geometry specialization of multi-candidate modes.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "mode_specialization")
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
    rows = metrics.pop("_rows")
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "input": str(jsonl_path),
            "target_key": args.target_key,
            "qwen_planner_checkpoint": str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    csv_path = args.output_dir / "per_mode_specialization.csv"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["mode"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({**metrics, "metrics_path": str(metrics_path), "csv_path": str(csv_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
