from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from langdirecttraj.data.instruction_dataset import InstructionTrajectoryDataset, collate_instruction_batch
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.eval.eval_anchor_decoder import build_model, load_checkpoint, prepare_eval_batch, summarize
from langdirecttraj.eval.instruction_geometry import contract_labels_from_tensors
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    move_batch,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def pairwise_candidate_distance(candidates: torch.Tensor, route_scales: torch.Tensor) -> torch.Tensor:
    """Mean pairwise trajectory distance in meters for [B, K, T, 2]."""
    if candidates.shape[1] < 2:
        return candidates.new_zeros(candidates.shape[0])
    diff = candidates[:, :, None] - candidates[:, None, :]
    dist = torch.linalg.norm(diff, dim=-1).mean(dim=-1) * route_scales.view(-1, 1, 1)
    k = candidates.shape[1]
    mask = torch.triu(torch.ones(k, k, dtype=torch.bool, device=candidates.device), diagonal=1)
    return dist[:, mask].mean(dim=1)


def endpoint_candidate_distance(candidates: torch.Tensor, route_scales: torch.Tensor) -> torch.Tensor:
    if candidates.shape[1] < 2:
        return candidates.new_zeros(candidates.shape[0])
    endpoints = candidates[:, :, -1, :]
    diff = endpoints[:, :, None] - endpoints[:, None, :]
    dist = torch.linalg.norm(diff, dim=-1) * route_scales.view(-1, 1, 1)
    k = candidates.shape[1]
    mask = torch.triu(torch.ones(k, k, dtype=torch.bool, device=candidates.device), diagonal=1)
    return dist[:, mask].mean(dim=1)


def candidate_errors_m(candidates: torch.Tensor, target: torch.Tensor, route_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    error = torch.linalg.norm(candidates - target[:, None], dim=-1) * route_scales.view(-1, 1, 1)
    ade = error.mean(dim=-1)
    fde = error[:, :, -1]
    return ade, fde


def summarize_index_counts(indices: List[int], num_candidates: int) -> Dict[str, int]:
    counts = {str(i): 0 for i in range(num_candidates)}
    for idx in indices:
        counts[str(int(idx))] += 1
    return counts


def _label_counts(labels: List[str], names: List[str]) -> Dict[str, int]:
    counts = {name: 0 for name in names}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    attr_dim: int,
    qwen_provider=None,
) -> Dict:
    selected_ade: List[float] = []
    selected_fde: List[float] = []
    oracle_ade: List[float] = []
    oracle_fde: List[float] = []
    first_ade: List[float] = []
    first_fde: List[float] = []
    diversity: List[float] = []
    endpoint_diversity: List[float] = []
    selected_indices: List[int] = []
    oracle_indices: List[int] = []
    oracle_margin: List[float] = []
    reranker_confidence: List[float] = []
    num_candidates = int(getattr(model.decoder, "num_candidates", 1))
    direction_names = list(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"].keys())
    direction_field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("direction")
    true_direction_labels: List[str] = []
    selected_direction_labels: List[str] = []
    candidate_direction_labels: List[List[str]] = [[] for _ in range(num_candidates)]
    candidate_has_true_by_label = {name: [] for name in direction_names}
    candidate_has_true_overall: List[float] = []

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
                selected_idx = torch.zeros(pred.shape[0], dtype=torch.long, device=device)
                probs = torch.ones(pred.shape[0], 1, dtype=pred.dtype, device=device)
            else:
                selected_idx = logits.argmax(dim=1)
                probs = torch.softmax(logits, dim=1)
            ade, fde = candidate_errors_m(pred, target, route_scale)
            oracle_idx = ade.argmin(dim=1)
            batch_idx = torch.arange(pred.shape[0], device=device)
            sorted_ade, _ = ade.sort(dim=1)
            margin = sorted_ade[:, 1] - sorted_ade[:, 0] if pred.shape[1] > 1 else torch.zeros_like(sorted_ade[:, 0])

            selected_ade.extend(ade[batch_idx, selected_idx].detach().cpu().tolist())
            selected_fde.extend(fde[batch_idx, selected_idx].detach().cpu().tolist())
            oracle_ade.extend(ade[batch_idx, oracle_idx].detach().cpu().tolist())
            oracle_fde.extend(fde[batch_idx, oracle_idx].detach().cpu().tolist())
            first_ade.extend(ade[:, 0].detach().cpu().tolist())
            first_fde.extend(fde[:, 0].detach().cpu().tolist())
            diversity.extend(pairwise_candidate_distance(pred, route_scale).detach().cpu().tolist())
            endpoint_diversity.extend(endpoint_candidate_distance(pred, route_scale).detach().cpu().tolist())
            selected_indices.extend(selected_idx.detach().cpu().tolist())
            oracle_indices.extend(oracle_idx.detach().cpu().tolist())
            oracle_margin.extend(margin.detach().cpu().tolist())
            reranker_confidence.extend(probs[batch_idx, selected_idx].detach().cpu().tolist())
            direction_ids = batch["mobility_tokens"][:, direction_field_idx].detach().cpu().numpy().tolist()
            true_labels = [
                DEFAULT_MOBILITY_VOCAB.id_to_token["direction"].get(int(token_id), "unknown")
                for token_id in direction_ids
            ]
            flat_pred = pred.reshape(pred.shape[0] * pred.shape[1], pred.shape[2], pred.shape[3])
            flat_reference = reference[:, None].expand(-1, pred.shape[1], -1, -1).reshape_as(flat_pred)
            flat_route_scale = route_scale[:, None].expand(-1, pred.shape[1]).reshape(-1)
            decoded_flat = [
                labels["direction"]
                for labels in contract_labels_from_tensors(
                    flat_pred.float(),
                    flat_reference.float(),
                    flat_route_scale.float(),
                    dataset="porto",
                )
            ]
            decoded = np.asarray(decoded_flat, dtype=object).reshape(pred.shape[0], pred.shape[1])
            selected_np = selected_idx.detach().cpu().numpy()
            for row_idx, true_label in enumerate(true_labels):
                true_direction_labels.append(true_label)
                selected_direction_labels.append(str(decoded[row_idx, selected_np[row_idx]]))
                row_labels = [str(decoded[row_idx, mode_idx]) for mode_idx in range(pred.shape[1])]
                for mode_idx, label in enumerate(row_labels):
                    candidate_direction_labels[mode_idx].append(label)
                has_true = true_label in row_labels
                candidate_has_true_by_label.setdefault(true_label, []).append(float(has_true))
                candidate_has_true_overall.append(float(has_true))

    selected_arr = np.asarray(selected_indices, dtype=np.int64)
    oracle_arr = np.asarray(oracle_indices, dtype=np.int64)
    reranker_accuracy = float(np.mean(selected_arr == oracle_arr)) if selected_arr.size else float("nan")
    return {
        "num_samples": int(len(selected_ade)),
        "num_candidates": int(num_candidates),
        "reranker_selected_ade_m": summarize(selected_ade),
        "reranker_selected_fde_m": summarize(selected_fde),
        "oracle_best_ade_m": summarize(oracle_ade),
        "oracle_best_fde_m": summarize(oracle_fde),
        "first_candidate_ade_m": summarize(first_ade),
        "first_candidate_fde_m": summarize(first_fde),
        "candidate_diversity_m": summarize(diversity),
        "candidate_endpoint_diversity_m": summarize(endpoint_diversity),
        "oracle_margin_ade_m": summarize(oracle_margin),
        "reranker_confidence": summarize(reranker_confidence),
        "reranker_accuracy": reranker_accuracy,
        "selected_index_counts": summarize_index_counts(selected_indices, num_candidates),
        "oracle_index_counts": summarize_index_counts(oracle_indices, num_candidates),
        "direction_candidate_coverage": {
            "true_counts": _label_counts(true_direction_labels, direction_names),
            "selected_counts": _label_counts(selected_direction_labels, direction_names),
            "mode_counts": {
                str(mode_idx): _label_counts(candidate_direction_labels[mode_idx], direction_names)
                for mode_idx in range(num_candidates)
            },
            "candidate_has_true_rate_by_label": {
                label: float(np.mean(values)) if values else float("nan")
                for label, values in candidate_has_true_by_label.items()
            },
            "candidate_has_true_overall": (
                float(np.mean(candidate_has_true_overall)) if candidate_has_true_overall else float("nan")
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose multi-candidate decoder quality and reranker behavior.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "multi_candidate_diagnostics")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--batch-size", type=int, default=16)
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
    jsonl_path = args.input or (REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_counterfactual_instruction_samples.filtered.jsonl")
    dataset = InstructionTrajectoryDataset(jsonl_path, max_samples=args.max_samples, target_key=args.target_key)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_instruction_batch,
    )
    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device) if args.qwen_planner_checkpoint else None
    metrics = evaluate(model, loader, device, attr_dim, qwen_provider=qwen_provider)
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
