#!/usr/bin/env python
"""Train a lightweight reranker over train-route memory candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langdirecttraj.eval.distribution_metrics import (  # noqa: E402
    evaluate_distribution,
    haversine_km,
    load_fixed_bounds,
)
from scripts.eval_coarse_road_cell_stage import (  # noqa: E402
    batch_refine_selection,
    build_coarse_prior,
    select_with_coarse_prior,
)
from scripts.eval_train_route_memory_prior import (  # noqa: E402
    CONTROLTRAJ_1000,
    build_memory_candidates,
    load_real,
    predict_train_only_target_lengths,
)
from scripts.eval_train_route_memory_prior import make_projection, od_features, od_frame_shapes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-real-npz", required=True, type=Path)
    parser.add_argument("--train-query-real-npz", required=True, type=Path)
    parser.add_argument("--val-real-npz", required=True, type=Path)
    parser.add_argument("--test-real-npz", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--memory-samples", type=int, default=50000)
    parser.add_argument("--train-query-samples", type=int, default=5000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--prior-samples", type=int, default=10000)
    parser.add_argument("--candidate-topm", type=int, default=64)
    parser.add_argument("--feature-mode", choices=["od", "od_len_angle"], default="od_len_angle")
    parser.add_argument("--retrieval-method", choices=["kdtree", "dense"], default="kdtree")
    parser.add_argument("--target-length-scale", type=float, default=1.15)
    parser.add_argument("--target-length-knn", type=int, default=16)
    parser.add_argument("--target-length-weight", type=float, default=10.0)
    parser.add_argument("--grid-num", type=int, default=32)
    parser.add_argument("--od-grid", type=int, default=6)
    parser.add_argument("--length-bins", type=int, default=32)
    parser.add_argument(
        "--bounds-json",
        type=Path,
        default=None,
        help="Optional fixed city bounds JSON, typically DiffTraj/data/<city>_od_only/<city>_global_stats.json.",
    )
    parser.add_argument(
        "--bounds-source",
        default=None,
        help="Stable provenance label written to priors and metrics.",
    )
    parser.add_argument(
        "--out-of-bounds-policy",
        choices=["discard", "clip", "overflow", "error"],
        default=None,
        help="Defaults to overflow with --bounds-json and discard for legacy inferred bounds.",
    )
    parser.add_argument("--min-group-count", type=int, default=3)
    parser.add_argument("--global-mix", type=float, default=0.1)
    parser.add_argument("--cell-weight", type=float, default=0.25)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--length-ratio-weight", type=float, default=2.0)
    parser.add_argument("--prototype-weight", type=float, default=0.5)
    parser.add_argument("--mode-index-weight", type=float, default=0.005)
    parser.add_argument("--batch-refine-cycles", type=int, default=5)
    parser.add_argument("--batch-density-refine-penalty", type=float, default=0.15)
    parser.add_argument("--batch-length-refine-penalty", type=float, default=6.0)
    parser.add_argument("--batch-refine-topk", type=int, default=64)
    parser.add_argument("--batch-density-hot-cells", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--eval-length-logit-weight", type=float, default=0.0)
    parser.add_argument("--teacher-ce-weight", type=float, default=1.0)
    parser.add_argument("--teacher-soft-weight", type=float, default=1.0)
    parser.add_argument("--teacher-score-weight", type=float, default=0.1)
    parser.add_argument("--teacher-temperature", type=float, default=0.35)
    parser.add_argument("--student-density-cost-weight", type=float, default=0.0)
    parser.add_argument("--student-transition-cost-weight", type=float, default=0.0)
    parser.add_argument("--student-length-cost-weight", type=float, default=0.0)
    parser.add_argument("--student-raw-length-cost-weight", type=float, default=0.0)
    parser.add_argument("--prior-score-feature", action="store_true")
    parser.add_argument("--eval-batch-refine", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


class RouteMemoryReranker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        n, k, d = features.shape
        return self.net(features.reshape(n * k, d)).reshape(n, k)


def standardize(train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True).clip(1e-4)
    return np.concatenate([mean, std], axis=0), [((arr - mean) / std).astype(np.float32) for arr in arrays]


def query_features(trajs: np.ndarray, proj, mode: str) -> np.ndarray:
    raw = od_features(trajs, proj).astype(np.float32)
    if mode == "od":
        return raw[:, :8]
    return raw


def make_candidate_features(
    query_real: np.ndarray,
    candidates: np.ndarray,
    memory_idx: np.ndarray,
    target_lengths: np.ndarray,
    proj,
    mode: str,
    query_stats: np.ndarray | None = None,
    prior_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n, k = candidates.shape[:2]
    q_raw = query_features(query_real, proj, mode)
    if query_stats is None:
        q_stats, [q] = standardize(q_raw, q_raw)
    else:
        mean, std = query_stats[:1], query_stats[1:]
        q_stats = query_stats
        q = ((q_raw - mean) / std).astype(np.float32)

    cand_lengths = np.empty((n, k), dtype=np.float32)
    normal_mean = np.empty((n, k), dtype=np.float32)
    normal_std = np.empty((n, k), dtype=np.float32)
    normal_abs_max = np.empty((n, k), dtype=np.float32)
    backtrack = np.empty((n, k), dtype=np.float32)
    chunk_size = 4096
    for start_i in range(0, n, chunk_size):
        end_i = min(start_i + chunk_size, n)
        cand_chunk = candidates[start_i:end_i]
        flat = cand_chunk.reshape((end_i - start_i) * k, candidates.shape[2], 2)
        shapes, _ = od_frame_shapes(flat, proj)
        shapes = shapes.reshape(end_i - start_i, k, candidates.shape[2], 2)
        normal = shapes[..., 1]
        s = shapes[..., 0]
        cand_lengths[start_i:end_i] = haversine_km(cand_chunk[:, :, :-1], cand_chunk[:, :, 1:]).sum(axis=2)
        normal_mean[start_i:end_i] = normal.mean(axis=2)
        normal_std[start_i:end_i] = normal.std(axis=2)
        normal_abs_max[start_i:end_i] = np.abs(normal).max(axis=2)
        backtrack[start_i:end_i] = np.maximum(0.0, -(s[:, :, 1:] - s[:, :, :-1])).mean(axis=2)
    chord = haversine_km(query_real[:, 0], query_real[:, -1]).clip(0.1)
    ratio = cand_lengths / chord[:, None]
    target = np.maximum(target_lengths[:, None], 0.1)
    length_abs = np.abs(cand_lengths - target) / target
    rank = np.arange(k, dtype=np.float32)[None, :] / max(k - 1, 1)
    feature_blocks = [
        rank.repeat(n, axis=0),
        cand_lengths / 20.0,
        ratio,
        length_abs,
        normal_mean,
        normal_std,
        normal_abs_max,
        backtrack,
        memory_idx.astype(np.float32) / max(float(memory_idx.max() + 1), 1.0),
    ]
    if prior_scores is not None:
        prior_scores = np.asarray(prior_scores, dtype=np.float32)
        score_mean = prior_scores.mean(axis=1, keepdims=True)
        score_std = prior_scores.std(axis=1, keepdims=True).clip(1e-4)
        score_z = (prior_scores - score_mean) / score_std
        score_order = np.argsort(np.argsort(prior_scores, axis=1), axis=1).astype(np.float32) / max(k - 1, 1)
        feature_blocks.extend([score_z, score_order])
    cand_feat = np.stack(
        feature_blocks,
        axis=2,
    ).astype(np.float32)
    features = np.concatenate([np.repeat(q[:, None, :], k, axis=1), cand_feat], axis=2)
    return features.astype(np.float32), q_stats


def compute_prior_scores(args, prior_train, query_real, candidates, prior) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    target_lengths = predict_train_only_target_lengths(
        prior_train,
        query_real,
        prior["bounds"],
        knn=int(args.target_length_knn),
        scale=float(args.target_length_scale),
        mode="ratio_median",
        hybrid_alpha=0.5,
    )
    selected, selector_stats, scores = select_with_coarse_prior(
        candidates,
        prior,
        min_group_count=int(args.min_group_count),
        global_mix=float(args.global_mix),
        cell_weight=float(args.cell_weight),
        transition_weight=float(args.transition_weight),
        length_ratio_weight=float(args.length_ratio_weight),
        prototype_weight=float(args.prototype_weight),
        target_lengths=target_lengths,
        target_length_weight=float(args.target_length_weight),
        mode_index_weight=float(args.mode_index_weight),
    )
    return selected.astype(np.int64), target_lengths, scores, selector_stats


def teacher_select(args, prior_train, query_real, candidates, prior) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    selected, target_lengths, scores, selector_stats = compute_prior_scores(
        args,
        prior_train,
        query_real,
        candidates,
        prior,
    )
    selected, refined_stats = batch_refine_selection(
        prior_train,
        candidates,
        selected,
        scores,
        grid_num=int(args.grid_num),
        length_bins=int(args.length_bins),
        cycles=int(args.batch_refine_cycles),
        density_penalty=float(args.batch_density_refine_penalty),
        length_penalty=float(args.batch_length_refine_penalty),
        topk=int(args.batch_refine_topk),
        hot_cells=int(args.batch_density_hot_cells),
        bounds=prior["bounds"],
        out_of_bounds_policy=str(prior.get("out_of_bounds_policy", "discard")),
    )
    return selected.astype(np.int64), target_lengths, selector_stats | refined_stats, scores


def train_reranker(
    args,
    features: np.ndarray,
    labels: np.ndarray,
    teacher_scores: np.ndarray,
    device: torch.device,
    teacher_component_scores: np.ndarray | None = None,
):
    torch.manual_seed(args.seed)
    model = RouteMemoryReranker(features.shape[2], int(args.hidden_dim)).to(device)
    has_components = teacher_component_scores is not None
    if has_components:
        teacher_component_scores = np.asarray(teacher_component_scores, dtype=np.float32)
        if teacher_component_scores.shape[:2] != features.shape[:2]:
            raise ValueError(
                "teacher_component_scores must have shape [N,K,C] aligned with features; "
                f"got {teacher_component_scores.shape}, expected first dims {features.shape[:2]}"
            )
        dataset = TensorDataset(
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(labels, dtype=torch.long),
            torch.as_tensor(teacher_scores, dtype=torch.float32),
            torch.as_tensor(teacher_component_scores, dtype=torch.float32),
        )
    else:
        dataset = TensorDataset(
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(labels, dtype=torch.long),
            torch.as_tensor(teacher_scores, dtype=torch.float32),
        )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        total_loss = 0.0
        total_acc = 0.0
        total_expected_density = 0.0
        total_expected_transition = 0.0
        total_expected_length = 0.0
        total_expected_raw_length = 0.0
        total = 0
        model.train()
        for batch_data in loader:
            if has_components:
                feat, label, score, component = batch_data
            else:
                feat, label, score = batch_data
                component = None
            feat = feat.to(device)
            label = label.to(device)
            score = score.to(device)
            logits = model(feat)
            ce = F.cross_entropy(logits, label)
            teacher_prob = F.softmax(-score / max(float(args.teacher_temperature), 1e-4), dim=1)
            soft = F.kl_div(F.log_softmax(logits, dim=1), teacher_prob, reduction="batchmean")
            centered_logits = logits - logits.mean(dim=1, keepdim=True)
            centered_teacher = (-score) - (-score).mean(dim=1, keepdim=True)
            centered_teacher = centered_teacher / centered_teacher.std(dim=1, keepdim=True).clamp_min(1e-4)
            centered_logits = centered_logits / centered_logits.std(dim=1, keepdim=True).clamp_min(1e-4)
            score_loss = F.mse_loss(centered_logits, centered_teacher)
            loss = (
                float(args.teacher_ce_weight) * ce
                + float(args.teacher_soft_weight) * soft
                + float(args.teacher_score_weight) * score_loss
            )
            if component is not None:
                component = component.to(device)
                probs = F.softmax(logits, dim=1)
                expected = (probs.unsqueeze(-1) * component).sum(dim=1)
                density_cost = expected[:, 0].mean()
                transition_cost = expected[:, 1].mean()
                length_cost = expected[:, 2].mean()
                raw_length_cost = expected[:, 3].mean() if expected.shape[1] > 3 else logits.new_tensor(0.0)
                loss = (
                    loss
                    + float(getattr(args, "student_density_cost_weight", 0.0)) * density_cost
                    + float(getattr(args, "student_transition_cost_weight", 0.0)) * transition_cost
                    + float(getattr(args, "student_length_cost_weight", 0.0)) * length_cost
                    + float(getattr(args, "student_raw_length_cost_weight", 0.0)) * raw_length_cost
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            batch = feat.shape[0]
            total += batch
            total_loss += float(loss.item()) * batch
            total_acc += float((logits.argmax(dim=1) == label).float().mean().item()) * batch
            if component is not None:
                total_expected_density += float(density_cost.item()) * batch
                total_expected_transition += float(transition_cost.item()) * batch
                total_expected_length += float(length_cost.item()) * batch
                total_expected_raw_length += float(raw_length_cost.item()) * batch
        if epoch == 1 or epoch == int(args.epochs) or epoch % max(1, int(args.epochs) // 8) == 0:
            row = {"epoch": epoch, "loss": total_loss / max(total, 1), "teacher_acc": total_acc / max(total, 1)}
            if has_components:
                row.update(
                    {
                        "expected_density_cost": total_expected_density / max(total, 1),
                        "expected_transition_cost": total_expected_transition / max(total, 1),
                        "expected_length_cost": total_expected_length / max(total, 1),
                        "expected_raw_length_cost": total_expected_raw_length / max(total, 1),
                    }
                )
            print(json.dumps(row, ensure_ascii=False), flush=True)
            history.append(row)
    return model, history


@torch.no_grad()
def predict(model, features: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    chunks = []
    for start in range(0, features.shape[0], batch_size):
        feat = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
        chunks.append(model(feat).cpu().numpy())
    logits = np.concatenate(chunks, axis=0)
    return logits.argmax(axis=1).astype(np.int64), logits


def evaluate_split(args, name, model, memory, prior_train, real, prior, proj, q_stats, device):
    candidates, memory_idx = build_memory_candidates(
        memory,
        real,
        int(args.candidate_topm),
        args.feature_mode,
        args.retrieval_method,
    )
    prior_selected, target_lengths, prior_scores, prior_stats = compute_prior_scores(
        args,
        prior_train,
        real,
        candidates,
        prior,
    )
    features, _ = make_candidate_features(
        real,
        candidates,
        memory_idx,
        target_lengths,
        proj,
        args.feature_mode,
        q_stats,
        prior_scores if args.prior_score_feature else None,
    )
    selected, logits = predict(model, features, device, int(args.batch_size))
    if float(args.eval_length_logit_weight) > 0:
        cand_lengths = haversine_km(candidates[:, :, :-1], candidates[:, :, 1:]).sum(axis=2)
        target = np.maximum(target_lengths[:, None], 0.1)
        length_penalty = np.abs(cand_lengths - target) / target
        adjusted_logits = logits - float(args.eval_length_logit_weight) * length_penalty
        selected = adjusted_logits.argmax(axis=1).astype(np.int64)
    else:
        adjusted_logits = logits
    if args.eval_batch_refine:
        selected, refined_stats = batch_refine_selection(
            prior_train,
            candidates,
            selected,
            -adjusted_logits,
            grid_num=int(args.grid_num),
            length_bins=int(args.length_bins),
            cycles=int(args.batch_refine_cycles),
            density_penalty=float(args.batch_density_refine_penalty),
            length_penalty=float(args.batch_length_refine_penalty),
            topk=int(args.batch_refine_topk),
            hot_cells=int(args.batch_density_hot_cells),
            bounds=prior["bounds"],
            out_of_bounds_policy=str(prior.get("out_of_bounds_policy", "discard")),
        )
    else:
        refined_stats = {}
    generated = candidates[np.arange(len(real)), selected]
    metrics = evaluate_distribution(
        real,
        generated,
        grid_num=int(args.grid_num),
        length_bins=int(args.length_bins),
        bounds=prior["bounds"],
        bounds_source=prior["bounds_source"],
        out_of_bounds_policy=str(prior.get("out_of_bounds_policy", "discard")),
    )
    metrics.update(
        {
            "selector": "trainable_route_memory_reranker",
            "num_samples": int(len(real)),
            "memory_samples": int(len(memory)),
            "prior_samples": int(len(prior_train)),
            "candidate_topm": int(args.candidate_topm),
            "eval_length_logit_weight": float(args.eval_length_logit_weight),
            "prior_score_feature": bool(args.prior_score_feature),
            "eval_batch_refine": bool(args.eval_batch_refine),
            "gt_avg_length_km": float(haversine_km(real[:, :-1], real[:, 1:]).sum(axis=1).mean()),
            "gen_avg_length_km": float(haversine_km(generated[:, :-1], generated[:, 1:]).sum(axis=1).mean()),
            "selected_memory_unique": int(np.unique(memory_idx[np.arange(len(real)), selected]).size),
            "prior_argmin_match_rate": float((selected == prior_selected).mean()),
            "refined_stats": refined_stats,
            "beats_controltraj_density": bool(metrics["Density Error"] < CONTROLTRAJ_1000["Density Error"]),
            "beats_controltraj_route": bool(metrics["Route Transition JSD"] < CONTROLTRAJ_1000["Route Transition JSD"]),
            "beats_controltraj_length": bool(metrics["Length Error"] < CONTROLTRAJ_1000["Length Error"]),
        }
    )
    metrics["beats_controltraj_all"] = bool(
        metrics["beats_controltraj_density"] and metrics["beats_controltraj_route"] and metrics["beats_controltraj_length"]
    )
    out = args.output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "reranker_trajectories.npz",
        real_trajectories=real.astype(np.float32),
        generated_trajectories=generated.astype(np.float32),
        selected_local=selected.astype(np.int64),
        logits=logits.astype(np.float32),
        adjusted_logits=adjusted_logits.astype(np.float32),
    )
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}]", json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main() -> None:
    args = parse_args()
    fixed_bounds = load_fixed_bounds(args.bounds_json) if args.bounds_json else None
    out_of_bounds_policy = args.out_of_bounds_policy or ("overflow" if fixed_bounds is not None else "discard")
    bounds_source = args.bounds_source or (str(args.bounds_json.resolve()) if args.bounds_json else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    memory = load_real(args.memory_real_npz, int(args.memory_samples))
    train_query = load_real(args.train_query_real_npz, int(args.train_query_samples))
    prior_train = memory[: int(args.prior_samples)]
    val = load_real(args.val_real_npz, int(args.eval_samples))
    test = load_real(args.test_real_npz, int(args.eval_samples)) if args.test_real_npz else None
    proj = make_projection(memory)
    prior = build_coarse_prior(
        prior_train,
        grid_num=int(args.grid_num),
        od_grid=int(args.od_grid),
        length_bins=int(args.length_bins),
        bounds=fixed_bounds,
        bounds_source=bounds_source,
        out_of_bounds_policy=out_of_bounds_policy,
    )
    train_candidates, train_memory_idx = build_memory_candidates(
        memory,
        train_query,
        int(args.candidate_topm),
        args.feature_mode,
        args.retrieval_method,
    )
    teacher, target_lengths, teacher_stats, teacher_scores = teacher_select(
        args,
        prior_train,
        train_query,
        train_candidates,
        prior,
    )
    features, q_stats = make_candidate_features(
        train_query,
        train_candidates,
        train_memory_idx,
        target_lengths,
        proj,
        args.feature_mode,
        None,
        teacher_scores if args.prior_score_feature else None,
    )
    model, history = train_reranker(args, features, teacher, teacher_scores, device)
    torch.save(
        {
            "model_state": model.state_dict(),
            "query_feature_stats": q_stats,
            "args": vars(args),
        },
        args.output_dir / "model.pt",
    )
    (args.output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "teacher_stats.json").write_text(
        json.dumps({"teacher_stats": teacher_stats, "teacher_hist": np.bincount(teacher, minlength=int(args.candidate_topm)).tolist()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    evaluate_split(args, "val1000", model, memory, prior_train, val, prior, proj, q_stats, device)
    if test is not None:
        evaluate_split(args, "test1000", model, memory, prior_train, test, prior, proj, q_stats, device)


if __name__ == "__main__":
    main()
