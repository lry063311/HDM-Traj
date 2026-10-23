"""Evaluate downstream utility of generated trajectory sets.

The script keeps the downstream learner deliberately lightweight and shared
across all sources. This makes the result a data-utility test rather than
another large-model benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langdirecttraj.eval.distribution_metrics import haversine_km, js_divergence


def parse_source(value: str) -> Tuple[str, Path, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("source must be name:path or name:path:key")
    name, rest = value.split(":", 1)
    if not name or not rest:
        raise argparse.ArgumentTypeError("source must be name:path or name:path:key")
    if ":" not in rest:
        return name, Path(rest), ""
    path_text, maybe_key = rest.rsplit(":", 1)
    if ("\\" in maybe_key) or ("/" in maybe_key) or maybe_key.endswith(".npz"):
        return name, Path(rest), ""
    return name, Path(path_text), maybe_key


def load_npz_array(path: Path, key: str = "") -> np.ndarray:
    payload = np.load(path)
    candidates = [key] if key else []
    candidates += [
        "real_trajectories",
        "generated_trajectories",
        "generated",
        "trajectories",
        "references",
    ]
    for candidate in candidates:
        if candidate and candidate in payload:
            arr = np.asarray(payload[candidate], dtype=np.float32)
            if arr.ndim != 3 or arr.shape[-1] != 2:
                raise ValueError(f"{path}:{candidate} has invalid trajectory shape {arr.shape}")
            return arr
    raise KeyError(f"{path} does not contain a supported trajectory key; keys={list(payload.keys())}")


def maybe_limit(arr: np.ndarray, max_samples: int) -> np.ndarray:
    if max_samples and max_samples > 0:
        return arr[: int(max_samples)]
    return arr


def infer_bounds(*arrays: np.ndarray) -> Dict[str, List[float]]:
    pts = np.concatenate([arr.reshape(-1, 2) for arr in arrays if len(arr)], axis=0)
    return {
        "lon_bound": [float(np.nanmin(pts[:, 0])), float(np.nanmax(pts[:, 0]))],
        "lat_bound": [float(np.nanmin(pts[:, 1])), float(np.nanmax(pts[:, 1]))],
    }


def to_cell_ids(trajs: np.ndarray, bounds: Dict[str, List[float]], grid_size: int) -> np.ndarray:
    lon_min, lon_max = bounds["lon_bound"]
    lat_min, lat_max = bounds["lat_bound"]
    lon_scale = max(float(lon_max - lon_min), 1e-9)
    lat_scale = max(float(lat_max - lat_min), 1e-9)
    gx = np.floor((trajs[..., 0] - lon_min) / lon_scale * grid_size).astype(np.int64)
    gy = np.floor((trajs[..., 1] - lat_min) / lat_scale * grid_size).astype(np.int64)
    gx = np.clip(gx, 0, grid_size - 1)
    gy = np.clip(gy, 0, grid_size - 1)
    return gy * grid_size + gx


def changed_sequence(ids: np.ndarray) -> List[int]:
    if ids.size == 0:
        return []
    seq = [int(ids[0])]
    for value in ids[1:]:
        value = int(value)
        if value != seq[-1]:
            seq.append(value)
    return seq


def iter_transitions(cell_ids: np.ndarray) -> Iterable[Tuple[int, int]]:
    for row in cell_ids:
        seq = changed_sequence(row)
        for a, b in zip(seq[:-1], seq[1:]):
            yield int(a), int(b)


def train_transition_model(cell_ids: np.ndarray) -> Tuple[Dict[int, Counter], Counter]:
    rows: Dict[int, Counter] = defaultdict(Counter)
    global_next: Counter = Counter()
    for a, b in iter_transitions(cell_ids):
        rows[a][b] += 1
        global_next[b] += 1
    return rows, global_next


def transition_distribution(transitions: Iterable[Tuple[int, int]]) -> Dict[int, float]:
    counts: Counter = Counter()
    total = 0
    for a, b in transitions:
        counts[int(a) << 16 | int(b)] += 1
        total += 1
    if total <= 0:
        return {}
    return {k: float(v) / float(total) for k, v in counts.items()}


def jsd_sparse(p: Dict[int, float], q: Dict[int, float]) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    pv = np.asarray([p.get(k, 0.0) for k in keys], dtype=np.float64)
    qv = np.asarray([q.get(k, 0.0) for k in keys], dtype=np.float64)
    pv = pv / max(float(pv.sum()), 1e-12)
    qv = qv / max(float(qv.sum()), 1e-12)
    return float(js_divergence(pv, qv))


def eval_transition_forecasting(
    train_trajs: np.ndarray,
    test_trajs: np.ndarray,
    bounds: Dict[str, List[float]],
    grid_size: int,
    alpha: float,
) -> Dict[str, float]:
    num_cells = int(grid_size) * int(grid_size)
    train_ids = to_cell_ids(train_trajs, bounds, grid_size)
    test_ids = to_cell_ids(test_trajs, bounds, grid_size)
    rows, global_next = train_transition_model(train_ids)
    global_best = int(global_next.most_common(1)[0][0]) if global_next else 0
    true_dist = transition_distribution(iter_transitions(test_ids))
    predicted_pairs: List[Tuple[int, int]] = []
    nll = 0.0
    total = 0
    top1 = 0
    top5 = 0
    for a, b in iter_transitions(test_ids):
        counts = rows.get(a)
        if counts:
            row_sum = float(sum(counts.values()))
            denom = row_sum + float(alpha) * float(num_cells)
            prob = (float(counts.get(b, 0)) + float(alpha)) / denom
            top = [int(k) for k, _v in counts.most_common(5)]
            pred = top[0] if top else global_best
        else:
            prob = 1.0 / float(num_cells)
            top = [int(k) for k, _v in global_next.most_common(5)]
            pred = top[0] if top else global_best
        nll -= math.log(max(prob, 1e-12))
        top1 += int(pred == b)
        top5 += int(b in top)
        predicted_pairs.append((int(a), int(pred)))
        total += 1
    pred_dist = transition_distribution(predicted_pairs)
    return {
        "num_test_transitions": int(total),
        "next_cell_accuracy": float(top1 / max(total, 1)),
        "top5_accuracy": float(top5 / max(total, 1)),
        "transition_nll": float(nll / max(total, 1)),
        "predicted_transition_jsd": jsd_sparse(true_dist, pred_dist),
        "train_contexts": int(len(rows)),
    }


def trajectory_lengths_km(trajs: np.ndarray) -> np.ndarray:
    return haversine_km(trajs[:, :-1], trajs[:, 1:]).sum(axis=1).astype(np.float32)


def od_features(trajs: np.ndarray, bounds: Dict[str, List[float]]) -> np.ndarray:
    start = trajs[:, 0]
    end = trajs[:, -1]
    lon_min, lon_max = bounds["lon_bound"]
    lat_min, lat_max = bounds["lat_bound"]
    lon_scale = max(float(lon_max - lon_min), 1e-9)
    lat_scale = max(float(lat_max - lat_min), 1e-9)
    dx = (end[:, 0] - start[:, 0]) / lon_scale
    dy = (end[:, 1] - start[:, 1]) / lat_scale
    chord = haversine_km(start, end).astype(np.float32)
    heading = np.arctan2(dy, dx).astype(np.float32)
    features = np.stack(
        [
            (start[:, 0] - lon_min) / lon_scale,
            (start[:, 1] - lat_min) / lat_scale,
            (end[:, 0] - lon_min) / lon_scale,
            (end[:, 1] - lat_min) / lat_scale,
            dx,
            dy,
            np.log1p(chord),
            np.sin(heading),
            np.cos(heading),
        ],
        axis=1,
    )
    return features.astype(np.float32)


def trajectory_mid_cells(trajs: np.ndarray, bounds: Dict[str, List[float]], grid_size: int, fraction: float = 0.5) -> np.ndarray:
    idx = int(round((trajs.shape[1] - 1) * float(fraction)))
    idx = max(0, min(trajs.shape[1] - 1, idx))
    return to_cell_ids(trajs[:, idx : idx + 1], bounds, grid_size)[:, 0].astype(np.int64)


def corridor_context_ids(
    trajs: np.ndarray,
    bounds: Dict[str, List[float]],
    context_grid: int,
    chord_edges: np.ndarray,
    heading_bins: int,
) -> np.ndarray:
    endpoints = trajs[:, [0, -1], :]
    endpoint_cells = to_cell_ids(endpoints, bounds, context_grid)
    origin = endpoint_cells[:, 0].astype(np.int64)
    dest = endpoint_cells[:, 1].astype(np.int64)
    start = trajs[:, 0]
    end = trajs[:, -1]
    lon_min, lon_max = bounds["lon_bound"]
    lat_min, lat_max = bounds["lat_bound"]
    lon_scale = max(float(lon_max - lon_min), 1e-9)
    lat_scale = max(float(lat_max - lat_min), 1e-9)
    dx = (end[:, 0] - start[:, 0]) / lon_scale
    dy = (end[:, 1] - start[:, 1]) / lat_scale
    chord = haversine_km(start, end).astype(np.float32)
    length_bin = np.searchsorted(chord_edges, chord, side="right").astype(np.int64)
    heading = np.arctan2(dy, dx)
    heading_norm = (heading + np.pi) / (2.0 * np.pi)
    heading_bin = np.floor(heading_norm * int(heading_bins)).astype(np.int64)
    heading_bin = np.clip(heading_bin, 0, int(heading_bins) - 1)
    num_context_cells = int(context_grid) * int(context_grid)
    return (
        (((origin * num_context_cells + dest) * (len(chord_edges) + 1) + length_bin) * int(heading_bins))
        + heading_bin
    ).astype(np.int64)


def train_corridor_model(contexts: np.ndarray, mid_cells: np.ndarray) -> Tuple[Dict[int, Counter], Counter]:
    rows: Dict[int, Counter] = defaultdict(Counter)
    global_mid: Counter = Counter()
    for ctx, mid in zip(contexts.tolist(), mid_cells.tolist()):
        rows[int(ctx)][int(mid)] += 1
        global_mid[int(mid)] += 1
    return rows, global_mid


def cell_distribution(cells: Iterable[int]) -> Dict[int, float]:
    counts: Counter = Counter()
    total = 0
    for cell in cells:
        counts[int(cell)] += 1
        total += 1
    if total <= 0:
        return {}
    return {k: float(v) / float(total) for k, v in counts.items()}


def eval_midroute_corridor_forecasting(
    train_trajs: np.ndarray,
    test_trajs: np.ndarray,
    real_train_trajs: np.ndarray,
    bounds: Dict[str, List[float]],
    grid_size: int,
    context_grid: int,
    chord_bins: int,
    heading_bins: int,
    alpha: float,
) -> Dict[str, float]:
    real_chords = haversine_km(real_train_trajs[:, 0], real_train_trajs[:, -1]).astype(np.float32)
    quantiles = np.linspace(0.0, 1.0, int(chord_bins) + 1)[1:-1]
    chord_edges = np.quantile(real_chords, quantiles).astype(np.float32)
    train_contexts = corridor_context_ids(train_trajs, bounds, int(context_grid), chord_edges, int(heading_bins))
    test_contexts = corridor_context_ids(test_trajs, bounds, int(context_grid), chord_edges, int(heading_bins))
    train_mid = trajectory_mid_cells(train_trajs, bounds, int(grid_size), fraction=0.5)
    test_mid = trajectory_mid_cells(test_trajs, bounds, int(grid_size), fraction=0.5)
    rows, global_mid = train_corridor_model(train_contexts, train_mid)
    global_top = [int(k) for k, _v in global_mid.most_common(5)]
    global_best = global_top[0] if global_top else 0
    num_cells = int(grid_size) * int(grid_size)
    nll = 0.0
    top1 = 0
    top5 = 0
    predicted_cells: List[int] = []
    for ctx, mid in zip(test_contexts.tolist(), test_mid.tolist()):
        counts = rows.get(int(ctx))
        if counts:
            row_sum = float(sum(counts.values()))
            denom = row_sum + float(alpha) * float(num_cells)
            prob = (float(counts.get(int(mid), 0)) + float(alpha)) / denom
            top = [int(k) for k, _v in counts.most_common(5)]
            pred = top[0] if top else global_best
        else:
            prob = 1.0 / float(num_cells)
            top = global_top
            pred = global_best
        nll -= math.log(max(prob, 1e-12))
        top1 += int(int(mid) == int(pred))
        top5 += int(int(mid) in top)
        predicted_cells.append(int(pred))
    total = int(len(test_mid))
    return {
        "num_test_samples": int(total),
        "mid_cell_accuracy": float(top1 / max(total, 1)),
        "mid_top5_accuracy": float(top5 / max(total, 1)),
        "mid_corridor_nll": float(nll / max(total, 1)),
        "predicted_mid_jsd": jsd_sparse(cell_distribution(test_mid.tolist()), cell_distribution(predicted_cells)),
        "train_contexts": int(len(rows)),
        "context_grid": int(context_grid),
        "chord_bins": int(chord_bins),
        "heading_bins": int(heading_bins),
    }


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True).clip(1e-6)
    xs = (x - mean) / std
    x_aug = np.concatenate([np.ones((xs.shape[0], 1), dtype=np.float32), xs], axis=1).astype(np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    reg = float(ridge) * np.eye(x_aug.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y64)
    return coef.astype(np.float64), mean.astype(np.float32), std.astype(np.float32)


def predict_ridge(x: np.ndarray, coef: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    xs = (x - mean) / std
    x_aug = np.concatenate([np.ones((xs.shape[0], 1), dtype=np.float32), xs], axis=1).astype(np.float64)
    return (x_aug @ coef).astype(np.float32)


def macro_f1_from_bins(true_bins: np.ndarray, pred_bins: np.ndarray, num_bins: int) -> float:
    scores = []
    for label in range(num_bins):
        tp = int(((true_bins == label) & (pred_bins == label)).sum())
        fp = int(((true_bins != label) & (pred_bins == label)).sum())
        fn = int(((true_bins == label) & (pred_bins != label)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall <= 0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def eval_length_prediction(
    train_trajs: np.ndarray,
    test_trajs: np.ndarray,
    real_train_trajs: np.ndarray,
    bounds: Dict[str, List[float]],
    ridge: float,
    bins: int,
) -> Dict[str, float]:
    x_train = od_features(train_trajs, bounds)
    y_train = trajectory_lengths_km(train_trajs)
    x_test = od_features(test_trajs, bounds)
    y_test = trajectory_lengths_km(test_trajs)
    coef, mean, std = fit_ridge(x_train, y_train, ridge)
    pred = np.clip(predict_ridge(x_test, coef, mean, std), 0.0, None)
    mae = np.abs(pred - y_test).mean()
    rmse = np.sqrt(((pred - y_test) ** 2).mean())
    real_lengths = trajectory_lengths_km(real_train_trajs)
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)[1:-1]
    edges = np.quantile(real_lengths, quantiles).astype(np.float32)
    true_bins = np.searchsorted(edges, y_test, side="right")
    pred_bins = np.searchsorted(edges, pred, side="right")
    return {
        "length_mae_km": float(mae),
        "length_rmse_km": float(rmse),
        "length_bin_macro_f1": macro_f1_from_bins(true_bins, pred_bins, int(bins)),
        "length_bin_count": int(bins),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-train", type=Path, required=True)
    parser.add_argument("--real-train-key", default="real_trajectories")
    parser.add_argument("--real-test", type=Path, required=True)
    parser.add_argument("--real-test-key", default="real_trajectories")
    parser.add_argument(
        "--train-source",
        action="append",
        type=parse_source,
        required=True,
        help="Training source as name:path or name:path:key. Repeat for real/controltraj/hdm/od-chord.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--length-bins", type=int, default=4)
    parser.add_argument("--corridor-context-grid", type=int, default=16)
    parser.add_argument("--corridor-chord-bins", type=int, default=4)
    parser.add_argument("--corridor-heading-bins", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_train = maybe_limit(load_npz_array(args.real_train, args.real_train_key), int(args.max_train_samples))
    real_test = maybe_limit(load_npz_array(args.real_test, args.real_test_key), int(args.max_test_samples))
    bounds = infer_bounds(real_train, real_test)
    results = {
        "protocol": {
            "tasks": [
                "route_cell_transition_forecasting",
                "mid_route_corridor_forecasting",
                "od_level_length_prediction_supplement",
            ],
            "grid_size": int(args.grid_size),
            "transition_alpha": float(args.alpha),
            "corridor_context_grid": int(args.corridor_context_grid),
            "corridor_chord_bins": int(args.corridor_chord_bins),
            "corridor_heading_bins": int(args.corridor_heading_bins),
            "ridge": float(args.ridge),
            "length_bins": int(args.length_bins),
            "bounds": bounds,
            "real_train": str(args.real_train),
            "real_test": str(args.real_test),
            "num_real_train": int(len(real_train)),
            "num_real_test": int(len(real_test)),
        },
        "sources": {},
    }
    for name, path, key in args.train_source:
        train = maybe_limit(load_npz_array(path, key), int(args.max_train_samples))
        n = min(len(train), len(real_train))
        train = train[:n]
        source_result = {
            "source_path": str(path),
            "source_key": key or "auto",
            "num_train": int(len(train)),
            "transition_forecasting": eval_transition_forecasting(
                train,
                real_test,
                bounds,
                int(args.grid_size),
                float(args.alpha),
            ),
            "midroute_corridor_forecasting": eval_midroute_corridor_forecasting(
                train,
                real_test,
                real_train,
                bounds,
                int(args.grid_size),
                int(args.corridor_context_grid),
                int(args.corridor_chord_bins),
                int(args.corridor_heading_bins),
                float(args.alpha),
            ),
            "length_prediction": eval_length_prediction(
                train,
                real_test,
                real_train,
                bounds,
                float(args.ridge),
                int(args.length_bins),
            ),
        }
        results["sources"][name] = source_result
        print(json.dumps({name: source_result}, ensure_ascii=False, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sources": list(results["sources"].keys())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
