#!/usr/bin/env python
"""Evaluate an OD-conditioned coarse road-cell route stage over OSM candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
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
    infer_bounds,
    js_divergence,
    load_fixed_bounds,
    to_grid,
    to_grid_ids,
)


def load_real(path: Path, key: str = "real_trajectories") -> tuple[np.ndarray, np.ndarray | None]:
    payload = np.load(path)
    if key not in payload.files:
        for candidate in ("real_trajectories", "real", "trajectories"):
            if candidate in payload.files:
                key = candidate
                break
        else:
            raise KeyError(f"{path} missing trajectory key {key!r}")
    sample_ids = None
    for id_key in ("sample_id", "sample_ids"):
        if id_key in payload.files:
            sample_ids = np.asarray(payload[id_key], dtype=np.int64)
            break
    return np.asarray(payload[key], dtype=np.float32), sample_ids


def _resample_polyline(points: np.ndarray, target_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == target_points:
        return points
    if len(points) == 0:
        return np.zeros((target_points, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, target_points, axis=0).astype(np.float32)
    seg = np.diff(points, axis=0)
    dist = np.sqrt((seg * seg).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(dist)])
    if float(cum[-1]) <= 1e-8:
        return np.repeat(points[:1], target_points, axis=0).astype(np.float32)
    target = np.linspace(0.0, float(cum[-1]), target_points)
    out = np.empty((target_points, 2), dtype=np.float32)
    out[:, 0] = np.interp(target, cum, points[:, 0])
    out[:, 1] = np.interp(target, cum, points[:, 1])
    return out


def reconstruct_candidates(
    tokens: np.ndarray,
    mask: np.ndarray,
    success: np.ndarray,
    fallback: np.ndarray,
    *,
    trajectory_points: int,
) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    success = np.asarray(success, dtype=bool)
    fallback = np.asarray(fallback, dtype=np.float32)
    n, k = tokens.shape[:2]
    candidates = np.empty((n, k, trajectory_points, 2), dtype=np.float32)
    for i in progress(range(n), desc="reconstruct candidates", total=n):
        fallback_i = _resample_polyline(fallback[i], trajectory_points)
        for j in range(k):
            valid = mask[i, j]
            path = tokens[i, j][valid] if valid.ndim > 0 else tokens[i, j]
            if not bool(success[i, j]) or len(path) < 2:
                candidates[i, j] = fallback_i
            else:
                candidates[i, j] = _resample_polyline(path, trajectory_points)
        candidates[i, :, 0] = fallback_i[0]
        candidates[i, :, -1] = fallback_i[-1]
    return candidates


def progress(iterable, *, desc: str, total: int | None = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, mininterval=5.0)


CONTROLTRAJ_VAL1000 = {
    "Density Error": 0.010734664117162583,
    "Route Transition JSD": 0.07013653251827912,
    "Length Error": 0.008120642734329995,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-real-npz", required=True, type=Path)
    parser.add_argument("--train-real-key", default="real_trajectories")
    parser.add_argument("--eval-real-npz", required=True, type=Path)
    parser.add_argument("--eval-real-key", default="real_trajectories")
    parser.add_argument("--road-corridor-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--trajectory-points", type=int, default=128)
    parser.add_argument("--grid-num", type=int, default=32)
    parser.add_argument("--od-grid", type=int, default=6)
    parser.add_argument("--length-bins", type=int, default=32)
    parser.add_argument(
        "--bounds-json",
        type=Path,
        default=None,
        help="Preprocessing statistics JSON containing fixed lon/lat bounds.",
    )
    parser.add_argument(
        "--bounds-source",
        default=None,
        help="Stable provenance label written to priors and metrics.",
    )
    parser.add_argument(
        "--out-of-bounds-policy",
        choices=("discard", "clip", "overflow", "error"),
        default=None,
        help="Defaults to overflow with --bounds-json and discard for legacy inferred bounds.",
    )
    parser.add_argument("--min-group-count", type=int, default=3)
    parser.add_argument("--global-mix", type=float, default=0.35)
    parser.add_argument("--cell-weight", type=float, default=0.25)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--length-ratio-weight", type=float, default=1.0)
    parser.add_argument("--target-length-weight", type=float, default=0.0)
    parser.add_argument("--target-length-knn", type=int, default=16)
    parser.add_argument("--target-length-scale", type=float, default=1.0)
    parser.add_argument("--prototype-weight", type=float, default=0.0)
    parser.add_argument("--mode-index-weight", type=float, default=0.0)
    parser.add_argument("--batch-refine-cycles", type=int, default=0)
    parser.add_argument("--batch-density-refine-penalty", type=float, default=0.0)
    parser.add_argument("--batch-length-refine-penalty", type=float, default=0.0)
    parser.add_argument("--batch-refine-topk", type=int, default=24)
    parser.add_argument("--batch-density-hot-cells", type=int, default=64)
    parser.add_argument("--save-trajectories", action="store_true")
    return parser.parse_args()


def _od_group(
    traj: np.ndarray,
    bounds: dict[str, list[float]],
    od_grid: int,
    out_of_bounds_policy: str = "discard",
) -> int:
    ids = to_grid_ids(
        np.stack([traj[0], traj[-1]], axis=0),
        bounds,
        od_grid,
        out_of_bounds_policy=out_of_bounds_policy,
    )
    if ids.shape[0] != 2:
        return 0
    od_num_cells = int(od_grid) * int(od_grid) + (1 if out_of_bounds_policy == "overflow" else 0)
    return int(ids[0]) * od_num_cells + int(ids[1])


def _changed_ids(
    traj: np.ndarray,
    bounds: dict[str, list[float]],
    grid_num: int,
    out_of_bounds_policy: str = "discard",
) -> np.ndarray:
    ids = to_grid_ids(traj, bounds, grid_num, out_of_bounds_policy=out_of_bounds_policy)
    if ids.shape[0] < 2:
        return ids
    keep = np.ones(ids.shape[0], dtype=bool)
    keep[1:] = ids[1:] != ids[:-1]
    return ids[keep]


def _transition_ids(changed_ids: np.ndarray, num_cells: int) -> np.ndarray:
    if changed_ids.shape[0] < 2:
        return np.zeros(0, dtype=np.int64)
    return changed_ids[:-1] * int(num_cells) + changed_ids[1:]


def _length_ratio(traj: np.ndarray) -> float:
    length = float(haversine_km(traj[:-1], traj[1:]).sum())
    chord = float(haversine_km(traj[0], traj[-1]))
    return length / max(chord, 0.1)


def _route_set(
    traj: np.ndarray,
    bounds: dict[str, list[float]],
    grid_num: int,
    out_of_bounds_policy: str = "discard",
) -> set[int]:
    num_cells = int(grid_num) * int(grid_num) + (1 if out_of_bounds_policy == "overflow" else 0)
    return set(
        _transition_ids(
            _changed_ids(traj, bounds, grid_num, out_of_bounds_policy),
            num_cells,
        ).tolist()
    )


def build_coarse_prior(
    train: np.ndarray,
    *,
    grid_num: int,
    od_grid: int,
    length_bins: int,
    bounds: dict[str, list[float]] | None = None,
    bounds_source: str | None = None,
    out_of_bounds_policy: str = "discard",
) -> dict:
    if bounds is None:
        bounds = infer_bounds(train, train)
        bounds_source = bounds_source or "train_split_extrema_legacy"
    else:
        bounds = {
            "lon_bound": [float(bounds["lon_bound"][0]), float(bounds["lon_bound"][1])],
            "lat_bound": [float(bounds["lat_bound"][0]), float(bounds["lat_bound"][1])],
        }
        bounds_source = bounds_source or "caller_supplied_fixed_bounds"
    num_cells = int(grid_num) * int(grid_num) + (1 if out_of_bounds_policy == "overflow" else 0)
    od_num_cells = int(od_grid) * int(od_grid) + (1 if out_of_bounds_policy == "overflow" else 0)
    global_cell = np.zeros(num_cells, dtype=np.float64)
    global_transition = np.zeros(num_cells * num_cells, dtype=np.float64)
    ratios = np.asarray([_length_ratio(traj) for traj in train], dtype=np.float64)
    ratio_max = max(float(np.percentile(ratios, 99)), 1.25)
    ratio_edges = np.linspace(1.0, ratio_max, int(length_bins) + 1)
    global_ratio_hist = np.zeros(int(length_bins), dtype=np.float64)
    group_cell: dict[int, np.ndarray] = {}
    group_transition: dict[int, np.ndarray] = {}
    group_ratio_hist: dict[int, np.ndarray] = {}
    group_routes: dict[int, list[set[int]]] = defaultdict(list)
    group_count: dict[int, int] = defaultdict(int)

    for traj, ratio in zip(train, ratios):
        group = _od_group(traj, bounds, od_grid, out_of_bounds_policy)
        group_count[group] += 1
        if group not in group_cell:
            group_cell[group] = np.zeros(num_cells, dtype=np.float64)
            group_transition[group] = np.zeros(num_cells * num_cells, dtype=np.float64)
            group_ratio_hist[group] = np.zeros(int(length_bins), dtype=np.float64)
        ids = _changed_ids(traj, bounds, grid_num, out_of_bounds_policy)
        transitions = _transition_ids(ids, num_cells)
        if ids.size:
            np.add.at(global_cell, ids, 1.0)
            np.add.at(group_cell[group], ids, 1.0)
        if transitions.size:
            np.add.at(global_transition, transitions, 1.0)
            np.add.at(group_transition[group], transitions, 1.0)
        bin_id = int(np.searchsorted(ratio_edges, np.clip(ratio, ratio_edges[0], ratio_edges[-1]), side="right") - 1)
        bin_id = int(np.clip(bin_id, 0, int(length_bins) - 1))
        global_ratio_hist[bin_id] += 1.0
        group_ratio_hist[group][bin_id] += 1.0
        group_routes[group].append(set(transitions.tolist()))

    def smooth_prob(counts: np.ndarray) -> np.ndarray:
        return (counts + 1.0) / float(counts.sum() + counts.size)

    return {
        "bounds": bounds,
        "bounds_source": bounds_source,
        "out_of_bounds_policy": out_of_bounds_policy,
        "grid_num": int(grid_num),
        "num_cells": int(num_cells),
        "od_grid": int(od_grid),
        "od_num_cells": int(od_num_cells),
        "length_bins": int(length_bins),
        "ratio_edges": ratio_edges,
        "global_cell_nll": -np.log(smooth_prob(global_cell)),
        "global_transition_nll": -np.log(smooth_prob(global_transition)),
        "global_ratio_nll": -np.log(smooth_prob(global_ratio_hist)),
        "group_cell_nll": {key: -np.log(smooth_prob(value)) for key, value in group_cell.items()},
        "group_transition_nll": {key: -np.log(smooth_prob(value)) for key, value in group_transition.items()},
        "group_ratio_nll": {key: -np.log(smooth_prob(value)) for key, value in group_ratio_hist.items()},
        "group_count": dict(group_count),
        "group_routes": dict(group_routes),
    }


def predict_target_lengths(
    train: np.ndarray,
    eval_trajs: np.ndarray,
    bounds: dict[str, list[float]],
    *,
    knn: int,
    scale: float,
) -> np.ndarray:
    lon_min, lon_max = bounds["lon_bound"]
    lat_min, lat_max = bounds["lat_bound"]

    def features(trajs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chord = haversine_km(trajs[:, 0], trajs[:, -1]).clip(0.1)
        denom_lon = max(float(lon_max - lon_min), 1e-9)
        denom_lat = max(float(lat_max - lat_min), 1e-9)
        feat = np.stack(
            [
                (trajs[:, 0, 0] - lon_min) / denom_lon,
                (trajs[:, 0, 1] - lat_min) / denom_lat,
                (trajs[:, -1, 0] - lon_min) / denom_lon,
                (trajs[:, -1, 1] - lat_min) / denom_lat,
                np.log(chord) / 5.0,
            ],
            axis=1,
        )
        return feat.astype(np.float32), chord.astype(np.float32)

    train_features, train_chord = features(train)
    eval_features, eval_chord = features(eval_trajs)
    train_lengths = haversine_km(train[:, :-1], train[:, 1:]).sum(axis=1)
    train_ratios = train_lengths / train_chord.clip(0.1)
    k = max(1, min(int(knn), train_features.shape[0]))
    distances = ((eval_features[:, None, :] - train_features[None, :, :]) ** 2).sum(axis=2)
    neighbor_idx = np.argpartition(distances, k - 1, axis=1)[:, :k]
    target_ratio = np.median(train_ratios[neighbor_idx], axis=1)
    return (target_ratio * eval_chord * float(scale)).astype(np.float32)


def _mean_lookup(values: np.ndarray, ids: np.ndarray) -> float:
    if ids.size == 0:
        return float(values.mean())
    return float(values[ids].mean())


def _prototype_distance(route: set[int], prototypes: list[set[int]]) -> float:
    if not prototypes:
        return 1.0
    if not route:
        return 1.0
    best = 1.0
    for proto in prototypes:
        union = len(route | proto)
        if union == 0:
            dist = 0.0
        else:
            dist = 1.0 - len(route & proto) / union
        if dist < best:
            best = dist
    return float(best)


def select_with_coarse_prior(
    candidates: np.ndarray,
    prior: dict,
    *,
    min_group_count: int,
    global_mix: float,
    cell_weight: float,
    transition_weight: float,
    length_ratio_weight: float,
    prototype_weight: float,
    target_lengths: np.ndarray | None = None,
    target_length_weight: float = 0.0,
    mode_index_weight: float,
) -> tuple[np.ndarray, dict, np.ndarray]:
    n, k_paths = candidates.shape[:2]
    grid_num = int(prior["grid_num"])
    out_of_bounds_policy = str(prior.get("out_of_bounds_policy", "discard"))
    num_cells = int(prior.get("num_cells", grid_num * grid_num))
    selected = np.zeros(n, dtype=np.int64)
    group_used = 0
    scores = np.zeros((n, k_paths), dtype=np.float64)
    for i in progress(range(n), desc="coarse prior score", total=n):
        group = _od_group(
            candidates[i, 0],
            prior["bounds"],
            int(prior["od_grid"]),
            out_of_bounds_policy,
        )
        use_group = int(prior["group_count"].get(group, 0)) >= int(min_group_count)
        if use_group:
            group_used += 1
        cell_nll = prior["group_cell_nll"].get(group, prior["global_cell_nll"]) if use_group else prior["global_cell_nll"]
        transition_nll = (
            prior["group_transition_nll"].get(group, prior["global_transition_nll"])
            if use_group
            else prior["global_transition_nll"]
        )
        ratio_nll = prior["group_ratio_nll"].get(group, prior["global_ratio_nll"]) if use_group else prior["global_ratio_nll"]
        prototypes = prior["group_routes"].get(group, []) if use_group else []
        mix = float(np.clip(global_mix, 0.0, 1.0)) if use_group else 1.0
        for k in range(k_paths):
            traj = candidates[i, k]
            ids = _changed_ids(traj, prior["bounds"], grid_num, out_of_bounds_policy)
            transitions = _transition_ids(ids, num_cells)
            ratio = _length_ratio(traj)
            bin_id = int(
                np.searchsorted(
                    prior["ratio_edges"],
                    np.clip(ratio, prior["ratio_edges"][0], prior["ratio_edges"][-1]),
                    side="right",
                )
                - 1
            )
            bin_id = int(np.clip(bin_id, 0, int(prior["length_bins"]) - 1))
            group_score = (
                float(cell_weight) * _mean_lookup(cell_nll, ids)
                + float(transition_weight) * _mean_lookup(transition_nll, transitions)
                + float(length_ratio_weight) * float(ratio_nll[bin_id])
            )
            global_score = (
                float(cell_weight) * _mean_lookup(prior["global_cell_nll"], ids)
                + float(transition_weight) * _mean_lookup(prior["global_transition_nll"], transitions)
                + float(length_ratio_weight) * float(prior["global_ratio_nll"][bin_id])
            )
            prototype_score = _prototype_distance(set(transitions.tolist()), prototypes)
            target_length_score = 0.0
            if target_lengths is not None and float(target_length_weight) > 0:
                length = float(haversine_km(traj[:-1], traj[1:]).sum())
                target_length = max(float(target_lengths[i]), 1.0)
                target_length_score = abs(length - target_length) / target_length
            scores[i, k] = (
                (1.0 - mix) * group_score
                + mix * global_score
                + float(prototype_weight) * prototype_score
                + float(target_length_weight) * target_length_score
                + float(mode_index_weight) * float(k)
            )
        selected[i] = int(np.argmin(scores[i]))
    return selected, {
        "group_used_rate": float(group_used / max(n, 1)),
        "selected_score_mean": float(scores[np.arange(n), selected].mean()),
        "target_length_mean_km": float(np.mean(target_lengths)) if target_lengths is not None else 0.0,
    }, scores


def batch_refine_selection(
    train: np.ndarray,
    candidates: np.ndarray,
    selected: np.ndarray,
    scores: np.ndarray,
    *,
    grid_num: int,
    length_bins: int,
    cycles: int,
    density_penalty: float,
    length_penalty: float,
    topk: int,
    hot_cells: int,
    target_lengths: np.ndarray | None = None,
    frozen_mask: np.ndarray | None = None,
    bounds: dict[str, list[float]] | None = None,
    out_of_bounds_policy: str = "discard",
) -> tuple[np.ndarray, dict]:
    if int(cycles) <= 0:
        return selected, {
            "batch_refine_cycles": 0,
            "batch_density_changes": 0,
            "batch_length_changes": 0,
        }
    n, k_paths = candidates.shape[:2]
    selected = selected.copy()
    frozen = (
        np.asarray(frozen_mask, dtype=bool).reshape(n)
        if frozen_mask is not None
        else np.zeros(n, dtype=bool)
    )
    start_scores = scores[np.arange(n), selected].copy()
    topk = max(1, min(int(topk), k_paths))
    if bounds is None:
        bounds = infer_bounds(train, candidates.reshape(n * k_paths, candidates.shape[2], 2))
    num_cells = int(grid_num) * int(grid_num) + (1 if out_of_bounds_policy == "overflow" else 0)
    train_density = to_grid(train, bounds, int(grid_num), out_of_bounds_policy)
    candidate_density = np.zeros((n, k_paths, num_cells), dtype=np.float32)
    for i in progress(range(n), desc="batch refine density prep", total=n):
        for k in range(k_paths):
            candidate_density[i, k] = to_grid(
                candidates[i, k : k + 1],
                bounds,
                int(grid_num),
                out_of_bounds_policy,
            )

    train_lengths = haversine_km(train[:, :-1], train[:, 1:]).sum(axis=1)
    reference_lengths = np.asarray(target_lengths, dtype=np.float32) if target_lengths is not None else train_lengths
    max_len = float(np.ceil(np.percentile(np.concatenate([train_lengths, reference_lengths]), 99) / 10.0) * 10.0)
    max_len = max(max_len, 1.0)
    length_edges = np.linspace(0.0, max_len, int(length_bins) + 1)
    target_length_hist, _ = np.histogram(np.clip(reference_lengths, 0.0, max_len), bins=length_edges)
    target_length_prob = target_length_hist.astype(np.float64) / max(float(target_length_hist.sum()), 1.0)
    candidate_lengths = haversine_km(candidates[:, :, :-1], candidates[:, :, 1:]).sum(axis=2)
    candidate_length_bins = np.clip(
        np.searchsorted(length_edges, np.clip(candidate_lengths, 0.0, max_len), side="right") - 1,
        0,
        int(length_bins) - 1,
    ).astype(np.int64)

    density_changes = 0
    length_changes = 0

    def density_step() -> int:
        nonlocal selected
        current_density = candidate_density[np.arange(n), selected].sum(axis=0)
        current_js = js_divergence(train_density, current_density)
        changes = 0
        for _pass in range(2):
            overfull = current_density / max(float(current_density.sum()), 1.0) - train_density / max(
                float(train_density.sum()), 1.0
            )
            hot = np.argsort(overfull)[-max(1, int(hot_cells)) :]
            pressure = candidate_density[np.arange(n), selected][:, hot].sum(axis=1)
            changed = False
            for sample_idx in np.argsort(pressure)[::-1]:
                if bool(frozen[sample_idx]):
                    continue
                old_k = int(selected[sample_idx])
                for candidate_k in np.argsort(scores[sample_idx])[:topk]:
                    candidate_k = int(candidate_k)
                    if candidate_k == old_k:
                        continue
                    if scores[sample_idx, candidate_k] - start_scores[sample_idx] > float(density_penalty):
                        continue
                    proposal = (
                        current_density
                        - candidate_density[sample_idx, old_k]
                        + candidate_density[sample_idx, candidate_k]
                    )
                    proposal_js = js_divergence(train_density, proposal)
                    if proposal_js + 1e-10 < current_js:
                        current_density = proposal
                        current_js = proposal_js
                        selected[sample_idx] = candidate_k
                        changes += 1
                        changed = True
                        break
            if not changed:
                break
        return changes

    def length_step() -> int:
        nonlocal selected
        counts = np.bincount(candidate_length_bins[np.arange(n), selected], minlength=int(length_bins)).astype(np.float64)
        current_js = js_divergence(target_length_hist, counts)
        changes = 0
        for _pass in range(3):
            overfull = counts / max(float(counts.sum()), 1.0) - target_length_prob
            current_bins = candidate_length_bins[np.arange(n), selected]
            order = np.argsort(overfull[current_bins])[::-1]
            changed = False
            for sample_idx in order:
                if bool(frozen[sample_idx]):
                    continue
                old_k = int(selected[sample_idx])
                old_bin = int(candidate_length_bins[sample_idx, old_k])
                for candidate_k in np.argsort(scores[sample_idx])[:topk]:
                    candidate_k = int(candidate_k)
                    if candidate_k == old_k:
                        continue
                    if scores[sample_idx, candidate_k] - start_scores[sample_idx] > float(length_penalty):
                        continue
                    new_bin = int(candidate_length_bins[sample_idx, candidate_k])
                    if overfull[new_bin] >= overfull[old_bin]:
                        continue
                    proposal = counts.copy()
                    proposal[old_bin] -= 1.0
                    proposal[new_bin] += 1.0
                    proposal_js = js_divergence(target_length_hist, proposal)
                    if proposal_js + 1e-10 < current_js:
                        counts = proposal
                        current_js = proposal_js
                        selected[sample_idx] = candidate_k
                        changes += 1
                        changed = True
                        break
            if not changed:
                break
        return changes

    for _cycle in progress(range(int(cycles)), desc="batch refine cycles", total=int(cycles)):
        if float(density_penalty) > 0:
            density_changes += density_step()
        if float(length_penalty) > 0:
            length_changes += length_step()
    if float(density_penalty) > 0:
        density_changes += density_step()

    return selected, {
        "batch_refine_cycles": int(cycles),
        "batch_density_changes": int(density_changes),
        "batch_length_changes": int(length_changes),
        "batch_density_refine_penalty": float(density_penalty),
        "batch_length_refine_penalty": float(length_penalty),
        "batch_refine_topk": int(topk),
        "batch_density_hot_cells": int(hot_cells),
        "batch_length_target": "predicted" if target_lengths is not None else "train",
        "batch_refine_frozen_count": int(frozen.sum()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bounds = load_fixed_bounds(args.bounds_json) if args.bounds_json else None
    out_of_bounds_policy = args.out_of_bounds_policy or ("overflow" if bounds is not None else "discard")
    bounds_source = args.bounds_source or (str(args.bounds_json.resolve()) if args.bounds_json else None)
    train, _ = load_real(args.train_real_npz, args.train_real_key)
    real, sample_ids = load_real(args.eval_real_npz, args.eval_real_key)
    real = real[: int(args.num_samples)]
    if sample_ids is not None:
        sample_ids = sample_ids[: len(real)]
    cache = np.load(args.road_corridor_cache)
    tokens = np.asarray(cache["road_corridor_tokens"], dtype=np.float32)[: len(real)]
    mask = np.asarray(cache["road_corridor_mask"], dtype=bool)[: len(real)]
    success = np.asarray(cache["road_path_success"], dtype=bool)[: len(real)]
    candidates = reconstruct_candidates(
        tokens,
        mask,
        success,
        real,
        trajectory_points=int(args.trajectory_points),
    )
    prior = build_coarse_prior(
        train,
        grid_num=int(args.grid_num),
        od_grid=int(args.od_grid),
        length_bins=int(args.length_bins),
        bounds=bounds,
        bounds_source=bounds_source,
        out_of_bounds_policy=out_of_bounds_policy,
    )
    target_lengths = None
    if float(args.target_length_weight) > 0:
        target_lengths = predict_target_lengths(
            train,
            real,
            prior["bounds"],
            knn=int(args.target_length_knn),
            scale=float(args.target_length_scale),
        )
    selected, selector_stats, candidate_scores = select_with_coarse_prior(
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
            train,
            candidates,
            selected,
            candidate_scores,
            grid_num=int(args.grid_num),
            length_bins=int(args.length_bins),
            cycles=int(args.batch_refine_cycles),
            density_penalty=float(args.batch_density_refine_penalty),
            length_penalty=float(args.batch_length_refine_penalty),
            topk=int(args.batch_refine_topk),
            hot_cells=int(args.batch_density_hot_cells),
            bounds=prior["bounds"],
            out_of_bounds_policy=out_of_bounds_policy,
        )
        selector_stats.update(refined_stats)
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
            "selector": "coarse_road_cell_stage",
            "train_real_npz": str(args.train_real_npz),
            "eval_real_npz": str(args.eval_real_npz),
            "road_corridor_cache": str(args.road_corridor_cache),
            "selected_k_hist": np.bincount(selected, minlength=candidates.shape[1]).astype(int).tolist(),
            "selector_stats": selector_stats,
            "params": {
                "grid_num": int(args.grid_num),
                "od_grid": int(args.od_grid),
                "length_bins": int(args.length_bins),
                "min_group_count": int(args.min_group_count),
                "global_mix": float(args.global_mix),
                "cell_weight": float(args.cell_weight),
                "transition_weight": float(args.transition_weight),
                "length_ratio_weight": float(args.length_ratio_weight),
                "target_length_weight": float(args.target_length_weight),
                "target_length_knn": int(args.target_length_knn),
                "target_length_scale": float(args.target_length_scale),
                "prototype_weight": float(args.prototype_weight),
                "mode_index_weight": float(args.mode_index_weight),
                "batch_refine_cycles": int(args.batch_refine_cycles),
                "batch_density_refine_penalty": float(args.batch_density_refine_penalty),
                "batch_length_refine_penalty": float(args.batch_length_refine_penalty),
                "batch_refine_topk": int(args.batch_refine_topk),
                "batch_density_hot_cells": int(args.batch_density_hot_cells),
            },
            "beats_controltraj_density": bool(metrics["Density Error"] < CONTROLTRAJ_VAL1000["Density Error"]),
            "beats_controltraj_route": bool(metrics["Route Transition JSD"] < CONTROLTRAJ_VAL1000["Route Transition JSD"]),
            "beats_controltraj_length": bool(metrics["Length Error"] < CONTROLTRAJ_VAL1000["Length Error"]),
        }
    )
    metrics["beats_controltraj_all"] = bool(
        metrics["beats_controltraj_density"]
        and metrics["beats_controltraj_route"]
        and metrics["beats_controltraj_length"]
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_trajectories:
        payload = {
            "real_trajectories": real.astype(np.float32),
            "trajectories": generated.astype(np.float32),
            "selected_k": selected.astype(np.int64),
        }
        if sample_ids is not None:
            payload["sample_ids"] = sample_ids
        np.savez_compressed(args.output_dir / "coarse_road_cell_stage_trajectories.npz", **payload)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
