#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
for path in (REPO_ROOT, WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from runtime_benchmark_common import RuntimeProtocol, module_parameter_counts, run_runtime_protocol
from langdirecttraj.eval.distribution_metrics import load_fixed_bounds
from scripts.eval_coarse_road_cell_stage import build_coarse_prior
from scripts.eval_train_route_memory_prior import (
    load_real,
    make_projection,
    od_features,
    od_frame_shapes,
    reconstruct_candidates_from_shapes,
    select_features,
)
from scripts.train_route_memory_reranker import compute_prior_scores, make_candidate_features, predict
from scripts.train_semantic_route_memory_reranker import (
    add_semantic_features,
    build_records,
    load_real_window,
    load_reranker_checkpoint,
    onehot_labels,
    select_candidates,
    table1_cost_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified end-to-end runtime benchmark for HDM-Traj Mode A.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--memory-real-npz", required=True, type=Path)
    parser.add_argument("--test-real-npz", required=True, type=Path)
    parser.add_argument("--test-input", required=True, type=Path)
    parser.add_argument("--bounds-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-samples", type=int, default=50000)
    parser.add_argument("--prior-samples", type=int, default=10000)
    parser.add_argument("--latency-samples", type=int, default=1000)
    parser.add_argument("--throughput-samples", type=int, default=9984)
    parser.add_argument("--throughput-batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--throughput-warmup-batches", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    for path in (
        cli.checkpoint,
        cli.memory_real_npz,
        cli.test_real_npz,
        cli.test_input,
        cli.bounds_json,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available() and cli.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the unified benchmark")
    device = torch.device(cli.device)
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    args = SimpleNamespace(**saved_args)
    expected = {
        "candidate_topm": 4,
        "feature_mode": "od",
        "retrieval_method": "kdtree",
        "grid_num": 16,
        "out_of_bounds_policy": "overflow",
        "disable_semantic_costs": True,
        "semantic_weight": 0.0,
        "selection_policy": "weighted_sum",
    }
    mismatch = {key: (getattr(args, key, None), value) for key, value in expected.items() if getattr(args, key, None) != value}
    if mismatch:
        raise ValueError(f"HDM checkpoint does not match the final Mode-A protocol: {mismatch}")
    model, q_stats = load_reranker_checkpoint(cli.checkpoint, device)
    memory = load_real(cli.memory_real_npz, cli.memory_samples)
    prior_train = memory[: cli.prior_samples]
    protocol = RuntimeProtocol(
        latency_samples=cli.latency_samples,
        throughput_samples=cli.throughput_samples,
        throughput_batch_size=cli.throughput_batch_size,
        repeats=cli.repeats,
        latency_warmup=cli.latency_warmup,
        throughput_warmup_batches=cli.throughput_warmup_batches,
    )
    sample_count = max(protocol.latency_samples, protocol.throughput_samples)
    queries = load_real_window(cli.test_real_npz, sample_count)
    sample_ids, records = build_records(cli.test_input, cli.test_real_npz, sample_count)
    del sample_ids

    projection = make_projection(memory)
    memory_shapes, raw_memory_features = od_frame_shapes(memory, projection)
    memory_features = select_features(raw_memory_features, args.feature_mode)
    feature_mean = memory_features.mean(axis=0, keepdims=True)
    feature_std = memory_features.std(axis=0, keepdims=True).clip(1e-4)
    memory_features = ((memory_features - feature_mean) / feature_std).astype(np.float32)
    index = cKDTree(memory_features)

    bounds = load_fixed_bounds(cli.bounds_json)
    prior = build_coarse_prior(
        prior_train,
        grid_num=int(args.grid_num),
        od_grid=int(args.od_grid),
        length_bins=int(args.length_bins),
        bounds=bounds,
        bounds_source=str(getattr(args, "bounds_source", cli.bounds_json)),
        out_of_bounds_policy=str(args.out_of_bounds_policy),
    )

    def retrieve(query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = od_features(query, projection).astype(np.float32)
        features = select_features(raw, args.feature_mode)
        features = ((features - feature_mean) / feature_std).astype(np.float32)
        _, memory_idx = index.query(features, k=int(args.candidate_topm))
        memory_idx = np.asarray(memory_idx, dtype=np.int64)
        if memory_idx.ndim == 1:
            memory_idx = memory_idx[:, None]
        candidates = reconstruct_candidates_from_shapes(query, memory_shapes[memory_idx], projection)
        return candidates, memory_idx

    def infer(start: int, count: int):
        real = np.asarray(queries[start : start + count], dtype=np.float32)
        batch_records = records[start : start + count]
        candidates, memory_idx = retrieve(real)
        _, target_lengths, prior_scores, _ = compute_prior_scores(
            args, prior_train, real, candidates, prior
        )
        features, _ = make_candidate_features(
            real,
            candidates,
            memory_idx,
            target_lengths,
            projection,
            args.feature_mode,
            q_stats,
            prior_scores if bool(args.prior_score_feature) else None,
        )
        semantic_cost = np.zeros(candidates.shape[:2], dtype=np.float32)
        semantic_fields = np.zeros((*candidates.shape[:2], 5), dtype=np.float32)
        if bool(getattr(args, "table1_cost_feature", False)):
            cost_features = table1_cost_features(
                args, prior_train, prior, candidates, target_lengths, semantic_cost
            )
            features = np.concatenate([features, cost_features], axis=2).astype(np.float32)
        features = add_semantic_features(features, onehot_labels(batch_records), semantic_fields)
        _, logits = predict(model, features, device, max(1, count))
        selected, _, _ = select_candidates(
            args,
            logits,
            prior_scores,
            semantic_cost,
            candidates,
            target_lengths,
        )
        return candidates[np.arange(count), selected]

    total, trainable = module_parameter_counts([model])
    run_runtime_protocol(
        method="HDM-Traj Mode A",
        infer=infer,
        device=device,
        output=cli.output,
        total_parameters=total,
        trainable_parameters=trainable,
        aux_storage_bytes=int(memory.nbytes),
        protocol=protocol,
        metadata={
            "checkpoint": str(cli.checkpoint.resolve()),
            "dataset": "porto",
            "split": "test",
            "steps_or_k": "K=4",
            "candidate_topm": 4,
            "route_memory_samples": int(len(memory)),
            "prior_samples": int(len(prior_train)),
            "route_memory_storage_definition": "retained_float32_route_bank_only",
            "inference_boundary": "OD_query_to_retrieval_to_prior_features_to_reranking_to_argmax_to_resampling",
            "semantic_weight": 0.0,
            "postprocessing": False,
        },
    )


if __name__ == "__main__":
    main()
