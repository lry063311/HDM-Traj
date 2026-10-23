#!/usr/bin/env python
"""Evaluate a lightweight train-route memory selector without OSM K-path caches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - progress is optional.
    tqdm = None

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


CONTROLTRAJ_UNIFIED_FULL = {
    "Density Error": 0.004439255705884607,
    "Route Transition JSD": 0.03114247146600556,
    "Length Error": 0.00405751974397496,
}
CONTROLTRAJ_1000 = CONTROLTRAJ_UNIFIED_FULL


def progress(iterable, *, desc: str, total: int | None = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, mininterval=5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-real-npz", required=True, type=Path)
    parser.add_argument("--eval-real-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=0, help="Deprecated alias for --eval-samples; 0 means full split.")
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=0)
    parser.add_argument("--prior-samples", type=int, default=0)
    parser.add_argument("--candidate-topm", type=int, default=64)
    parser.add_argument("--retrieval-method", choices=["kdtree", "dense"], default="kdtree")
    parser.add_argument("--feature-mode", choices=["od", "od_len_angle"], default="od_len_angle")
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
    parser.add_argument("--target-length-weight", type=float, default=10.0)
    parser.add_argument("--target-length-knn", type=int, default=16)
    parser.add_argument("--target-length-scale", type=float, default=1.1)
    parser.add_argument(
        "--target-length-mode",
        choices=["ratio_median", "length_median", "hybrid"],
        default="ratio_median",
    )
    parser.add_argument("--target-length-hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--prototype-weight", type=float, default=0.5)
    parser.add_argument("--mode-index-weight", type=float, default=0.0)
    parser.add_argument("--batch-refine-cycles", type=int, default=5)
    parser.add_argument("--batch-density-refine-penalty", type=float, default=0.15)
    parser.add_argument("--batch-length-refine-penalty", type=float, default=6.0)
    parser.add_argument("--batch-refine-topk", type=int, default=64)
    parser.add_argument("--batch-density-hot-cells", type=int, default=64)
    parser.add_argument("--batch-length-target", choices=["train", "predicted"], default="predicted")
    parser.add_argument("--controltraj-density-threshold", type=float, default=CONTROLTRAJ_UNIFIED_FULL["Density Error"])
    parser.add_argument("--controltraj-route-threshold", type=float, default=CONTROLTRAJ_UNIFIED_FULL["Route Transition JSD"])
    parser.add_argument("--controltraj-length-threshold", type=float, default=CONTROLTRAJ_UNIFIED_FULL["Length Error"])
    return parser.parse_args()


def load_real(path: Path, n: int) -> np.ndarray:
    data = np.load(path)
    traj = np.asarray(data["real_trajectories"], dtype=np.float32)
    if n > 0:
        traj = traj[:n]
    return traj


def make_projection(trajs: np.ndarray) -> dict[str, float]:
    lat0 = float(np.mean(trajs[..., 1]))
    return {
        "lat0": lat0,
        "cos_lat0": float(max(np.cos(np.deg2rad(lat0)), 1e-6)),
        "lon0": float(np.mean(trajs[..., 0])),
        "lat_origin": lat0,
    }


def lonlat_to_xy(points: np.ndarray, proj: dict[str, float]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    x = (points[..., 0] - float(proj["lon0"])) * 111.320 * float(proj["cos_lat0"])
    y = (points[..., 1] - float(proj["lat_origin"])) * 110.574
    return np.stack([x, y], axis=-1).astype(np.float32)


def xy_to_lonlat(points: np.ndarray, proj: dict[str, float]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    lon = points[..., 0] / (111.320 * float(proj["cos_lat0"])) + float(proj["lon0"])
    lat = points[..., 1] / 110.574 + float(proj["lat_origin"])
    return np.stack([lon, lat], axis=-1).astype(np.float32)


def od_features(trajs: np.ndarray, proj: dict[str, float]) -> np.ndarray:
    xy = lonlat_to_xy(trajs, proj)
    start = xy[:, 0]
    end = xy[:, -1]
    vec = end - start
    chord = np.linalg.norm(vec, axis=1).clip(1e-3)
    angle = np.arctan2(vec[:, 1], vec[:, 0])
    length = haversine_km(trajs[:, :-1], trajs[:, 1:]).sum(axis=1).clip(1e-3)
    ratio = length / np.maximum(chord, 1e-3)
    return np.stack(
        [
            start[:, 0],
            start[:, 1],
            end[:, 0],
            end[:, 1],
            vec[:, 0],
            vec[:, 1],
            np.log(chord),
            angle,
            np.log(length),
            ratio,
        ],
        axis=1,
    ).astype(np.float32)


def od_frame_shapes(trajs: np.ndarray, proj: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    xy = lonlat_to_xy(trajs, proj)
    start = xy[:, 0]
    end = xy[:, -1]
    vec = end - start
    chord = np.linalg.norm(vec, axis=1).clip(1e-3)
    tangent = vec / chord[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    rel = xy - start[:, None, :]
    along = (rel * tangent[:, None, :]).sum(axis=2) / chord[:, None]
    lateral = (rel * normal[:, None, :]).sum(axis=2) / chord[:, None]
    shapes = np.stack([along, lateral], axis=2).astype(np.float32)
    shapes[:, 0, :] = np.array([0.0, 0.0], dtype=np.float32)
    shapes[:, -1, :] = np.array([1.0, 0.0], dtype=np.float32)
    return shapes, od_features(trajs, proj)


def reconstruct_candidates_from_shapes(eval_real: np.ndarray, candidate_shapes: np.ndarray, proj: dict[str, float]) -> np.ndarray:
    eval_xy = lonlat_to_xy(eval_real, proj)
    start = eval_xy[:, 0]
    end = eval_xy[:, -1]
    vec = end - start
    chord = np.linalg.norm(vec, axis=1).clip(1e-3)
    tangent = vec / chord[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    shapes = np.asarray(candidate_shapes, dtype=np.float32)
    xy = (
        start[:, None, None, :]
        + shapes[..., 0, None] * chord[:, None, None, None] * tangent[:, None, None, :]
        + shapes[..., 1, None] * chord[:, None, None, None] * normal[:, None, None, :]
    )
    out = xy_to_lonlat(xy, proj)
    out[:, :, 0] = eval_real[:, None, 0]
    out[:, :, -1] = eval_real[:, None, -1]
    return out.astype(np.float32)


def select_features(raw: np.ndarray, mode: str) -> np.ndarray:
    if mode == "od":
        return raw[:, :8]
    return raw


def standardize(train: np.ndarray, eval_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True).clip(1e-4)
    return ((train - mean) / std).astype(np.float32), ((eval_ - mean) / std).astype(np.float32)


def build_memory_candidates(
    train: np.ndarray,
    eval_real: np.ndarray,
    topm: int,
    feature_mode: str,
    retrieval_method: str,
    *,
    cache_path: Path | None = None,
    reuse_cache: bool = False,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    topm = max(1, min(int(topm), train.shape[0]))
    n = int(eval_real.shape[0])
    target_len = int(eval_real.shape[1])

    cache_parts_dir: Path | None = None
    manifest_path: Path | None = None
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_parts_dir = Path(str(cache_path) + ".parts")
        manifest_path = cache_parts_dir / "manifest.json"
        if bool(reuse_cache) and cache_path.exists():
            payload = np.load(cache_path)
            candidates = np.asarray(payload["candidates"], dtype=np.float32)
            idx = np.asarray(payload["memory_idx"], dtype=np.int64)
            if candidates.shape[:3] != (n, topm, target_len):
                raise ValueError(f"Candidate cache shape {candidates.shape} does not match {(n, topm, target_len, 2)}")
            return candidates, idx
        if bool(reuse_cache) and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "n": n,
                "topm": topm,
                "target_len": target_len,
                "feature_mode": str(feature_mode),
                "retrieval_method": str(retrieval_method),
            }
            mismatch = {k: (manifest.get(k), v) for k, v in expected.items() if manifest.get(k) != v}
            if mismatch:
                raise ValueError(f"Candidate cache manifest mismatch: {mismatch}")
            candidates_path = cache_parts_dir / "candidates.float32.dat"
            idx_path = cache_parts_dir / "memory_idx.int64.dat"
            completed_path = cache_parts_dir / "completed.bool.dat"
            if candidates_path.exists() and idx_path.exists() and completed_path.exists():
                completed = np.memmap(completed_path, mode="r", dtype=np.bool_, shape=(n,))
                if bool(np.asarray(completed).all()):
                    candidates = np.memmap(candidates_path, mode="r", dtype=np.float32, shape=(n, topm, target_len, 2))
                    idx = np.memmap(idx_path, mode="r", dtype=np.int64, shape=(n, topm))
                    return candidates, idx

    proj = make_projection(train)
    train_shapes, train_features_raw = od_frame_shapes(train, proj)
    eval_features_raw = od_features(eval_real, proj).astype(np.float32)
    train_features = select_features(train_features_raw, feature_mode)
    eval_features = select_features(eval_features_raw, feature_mode)
    train_features, eval_features = standardize(train_features, eval_features)

    def query_topm(features_chunk: np.ndarray) -> np.ndarray:
        if retrieval_method == "dense":
            distances = ((features_chunk[:, None, :] - train_features[None, :, :]) ** 2).sum(axis=2)
            local_idx = np.argpartition(distances, topm - 1, axis=1)[:, :topm]
            row_order = np.argsort(np.take_along_axis(distances, local_idx, axis=1), axis=1)
            return np.take_along_axis(local_idx, row_order, axis=1).astype(np.int64)
        tree = query_topm.tree
        _, local_idx = tree.query(features_chunk, k=topm)
        local_idx = np.asarray(local_idx, dtype=np.int64)
        if local_idx.ndim == 1:
            local_idx = local_idx[:, None]
        return local_idx

    query_topm.tree = cKDTree(train_features) if retrieval_method != "dense" else None

    if cache_path is not None and cache_parts_dir is not None and manifest_path is not None:
        cache_parts_dir.mkdir(parents=True, exist_ok=True)
        candidates_path = cache_parts_dir / "candidates.float32.dat"
        idx_path = cache_parts_dir / "memory_idx.int64.dat"
        completed_path = cache_parts_dir / "completed.bool.dat"
        completed_exists = completed_path.exists()
        manifest = {
            "n": n,
            "topm": topm,
            "target_len": target_len,
            "feature_mode": str(feature_mode),
            "retrieval_method": str(retrieval_method),
            "dtype": "float32",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        candidates = np.memmap(candidates_path, mode="w+" if not candidates_path.exists() else "r+", dtype=np.float32, shape=(n, topm, target_len, 2))
        idx = np.memmap(idx_path, mode="w+" if not idx_path.exists() else "r+", dtype=np.int64, shape=(n, topm))
        completed = np.memmap(completed_path, mode="w+" if not completed_path.exists() else "r+", dtype=np.bool_, shape=(n,))
        if not completed_exists:
            completed[:] = False
            completed.flush()
        chunk_size = max(1, int(chunk_size))
        ranges = list(range(0, n, chunk_size))
        for start in progress(ranges, desc="memory candidates chunks", total=len(ranges)):
            end = min(n, start + chunk_size)
            if bool(np.asarray(completed[start:end]).all()):
                continue
            local_idx = query_topm(eval_features[start:end])
            candidate_shapes = train_shapes[local_idx]
            candidates[start:end] = reconstruct_candidates_from_shapes(eval_real[start:end], candidate_shapes, proj).astype(np.float32)
            idx[start:end] = local_idx.astype(np.int64)
            completed[start:end] = True
            candidates.flush()
            idx.flush()
            completed.flush()
        if not bool(np.asarray(completed).all()):
            raise RuntimeError(f"Incomplete candidate cache at {cache_parts_dir}")
        return candidates, idx

    if retrieval_method == "dense":
        distances = ((eval_features[:, None, :] - train_features[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(distances, topm - 1, axis=1)[:, :topm]
        row_order = np.argsort(np.take_along_axis(distances, idx, axis=1), axis=1)
        idx = np.take_along_axis(idx, row_order, axis=1)
    else:
        tree = cKDTree(train_features)
        _, idx = tree.query(eval_features, k=topm)
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[:, None]
    candidate_shapes = train_shapes[idx]
    candidates = reconstruct_candidates_from_shapes(eval_real, candidate_shapes, proj)
    return candidates, idx.astype(np.int64)


def predict_train_only_target_lengths(
    train: np.ndarray,
    eval_trajs: np.ndarray,
    bounds: dict[str, list[float]],
    *,
    knn: int,
    scale: float,
    mode: str,
    hybrid_alpha: float,
) -> np.ndarray:
    lon_min, lon_max = bounds["lon_bound"]
    lat_min, lat_max = bounds["lat_bound"]

    def features(trajs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chord = haversine_km(trajs[:, 0], trajs[:, -1]).clip(0.1)
        denom_lon = max(float(lon_max - lon_min), 1e-9)
        denom_lat = max(float(lat_max - lat_min), 1e-9)
        vec = np.stack(
            [
                (trajs[:, 0, 0] - lon_min) / denom_lon,
                (trajs[:, 0, 1] - lat_min) / denom_lat,
                (trajs[:, -1, 0] - lon_min) / denom_lon,
                (trajs[:, -1, 1] - lat_min) / denom_lat,
                np.log(chord) / 5.0,
            ],
            axis=1,
        )
        return vec.astype(np.float32), chord.astype(np.float32)

    train_features, train_chord = features(train)
    eval_features, eval_chord = features(eval_trajs)
    train_lengths = haversine_km(train[:, :-1], train[:, 1:]).sum(axis=1).astype(np.float32)
    train_ratios = train_lengths / train_chord.clip(0.1)
    k = max(1, min(int(knn), train_features.shape[0]))
    # Full val/test can be 100k+ queries. Building the dense
    # (num_eval, num_train, dim) tensor at once can exceed local RAM, so keep
    # the exact KNN calculation but stream it over eval chunks.
    chunk_size = 1024
    neighbor_idx = np.empty((eval_features.shape[0], k), dtype=np.int64)
    for start in range(0, eval_features.shape[0], chunk_size):
        end = min(start + chunk_size, eval_features.shape[0])
        distances = ((eval_features[start:end, None, :] - train_features[None, :, :]) ** 2).sum(axis=2)
        neighbor_idx[start:end] = np.argpartition(distances, k - 1, axis=1)[:, :k]
    ratio_target = np.median(train_ratios[neighbor_idx], axis=1) * eval_chord
    length_target = np.median(train_lengths[neighbor_idx], axis=1)
    if mode == "length_median":
        target = length_target
    elif mode == "hybrid":
        alpha = float(np.clip(hybrid_alpha, 0.0, 1.0))
        target = (1.0 - alpha) * ratio_target + alpha * length_target
    else:
        target = ratio_target
    return (target * float(scale)).astype(np.float32)


def main() -> None:
    args = parse_args()
    fixed_bounds = load_fixed_bounds(args.bounds_json) if args.bounds_json else None
    out_of_bounds_policy = args.out_of_bounds_policy or ("overflow" if fixed_bounds is not None else "discard")
    bounds_source = args.bounds_source or (str(args.bounds_json.resolve()) if args.bounds_json else None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_samples = int(args.eval_samples) if int(args.eval_samples) > 0 else int(args.num_samples)
    train_samples = int(args.train_samples)
    train = load_real(args.train_real_npz, train_samples)
    real = load_real(args.eval_real_npz, eval_samples)
    candidates, memory_idx = build_memory_candidates(
        train,
        real,
        int(args.candidate_topm),
        args.feature_mode,
        args.retrieval_method,
    )
    prior_train = train[: int(args.prior_samples)] if int(args.prior_samples) > 0 else train
    prior = build_coarse_prior(
        prior_train,
        grid_num=int(args.grid_num),
        od_grid=int(args.od_grid),
        length_bins=int(args.length_bins),
        bounds=fixed_bounds,
        bounds_source=bounds_source,
        out_of_bounds_policy=out_of_bounds_policy,
    )
    target_lengths = None
    if float(args.target_length_weight) > 0:
        target_lengths = predict_train_only_target_lengths(
            prior_train,
            real,
            prior["bounds"],
            knn=int(args.target_length_knn),
            scale=float(args.target_length_scale),
            mode=args.target_length_mode,
            hybrid_alpha=float(args.target_length_hybrid_alpha),
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
    refined_stats = {}
    if int(args.batch_refine_cycles) > 0:
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
            target_lengths=target_lengths if args.batch_length_target == "predicted" else None,
            bounds=prior["bounds"],
            out_of_bounds_policy=out_of_bounds_policy,
        )
    generated = candidates[np.arange(len(real)), selected]
    metrics = evaluate_distribution(
        real,
        generated,
        grid_num=int(args.grid_num),
        length_bins=int(args.length_bins),
        bounds=prior["bounds"],
        bounds_source=prior["bounds_source"],
        out_of_bounds_policy=out_of_bounds_policy,
    )
    metrics.update(
        {
            "selector": "train_route_memory_prior",
            "num_samples": int(len(real)),
            "train_samples": int(len(train)),
            "prior_samples": int(len(prior_train)),
            "candidate_topm": int(candidates.shape[1]),
            "feature_mode": args.feature_mode,
            "retrieval_method": args.retrieval_method,
            "target_length_mode": args.target_length_mode,
            "target_length_hybrid_alpha": float(args.target_length_hybrid_alpha),
            "batch_length_target": args.batch_length_target,
            "gt_avg_length_km": float(haversine_km(real[:, :-1], real[:, 1:]).sum(axis=1).mean()),
            "gen_avg_length_km": float(haversine_km(generated[:, :-1], generated[:, 1:]).sum(axis=1).mean()),
            "selected_memory_unique": int(np.unique(memory_idx[np.arange(len(real)), selected]).size),
            "selector_stats": selector_stats | refined_stats,
            "controltraj_thresholds": {
                "Density Error": float(args.controltraj_density_threshold),
                "Route Transition JSD": float(args.controltraj_route_threshold),
                "Length Error": float(args.controltraj_length_threshold),
            },
            "beats_controltraj_density": bool(metrics["Density Error"] < float(args.controltraj_density_threshold)),
            "beats_controltraj_route": bool(metrics["Route Transition JSD"] < float(args.controltraj_route_threshold)),
            "beats_controltraj_length": bool(metrics["Length Error"] < float(args.controltraj_length_threshold)),
        }
    )
    metrics["beats_controltraj_all"] = bool(
        metrics["beats_controltraj_density"]
        and metrics["beats_controltraj_route"]
        and metrics["beats_controltraj_length"]
    )
    np.savez_compressed(
        args.output_dir / "memory_prior_trajectories.npz",
        real_trajectories=real.astype(np.float32),
        generated_trajectories=generated.astype(np.float32),
        selected_local=selected.astype(np.int64),
        selected_memory_index=memory_idx[np.arange(len(real)), selected].astype(np.int64),
        memory_indices=memory_idx.astype(np.int64),
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
