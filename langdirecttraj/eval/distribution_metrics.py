from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OutOfBoundsPolicy = Literal["discard", "clip", "overflow", "error"]


def load_array(path: Path, key: str) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npz":
        payload = np.load(path)
        return np.asarray(payload[key], dtype=np.float32)
    return np.asarray(np.load(path), dtype=np.float32)


def haversine_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lon1 = np.deg2rad(a[..., 0])
    lat1 = np.deg2rad(a[..., 1])
    lon2 = np.deg2rad(b[..., 0])
    lat2 = np.deg2rad(b[..., 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64) / max(float(p.sum()), 1e-12)
    q = q.astype(np.float64) / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    kl_pm = np.sum(p[mask_p] * np.log(p[mask_p] / np.clip(m[mask_p], 1e-12, None)))
    kl_qm = np.sum(q[mask_q] * np.log(q[mask_q] / np.clip(m[mask_q], 1e-12, None)))
    return float(0.5 * (kl_pm + kl_qm))


def infer_bounds(real: np.ndarray, generated: np.ndarray, padding_ratio: float = 0.0) -> Dict[str, List[float]]:
    pts = np.concatenate([real.reshape(-1, 2), generated.reshape(-1, 2)], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    lon_min, lat_min = pts.min(axis=0)
    lon_max, lat_max = pts.max(axis=0)
    lon_pad = (lon_max - lon_min) * padding_ratio
    lat_pad = (lat_max - lat_min) * padding_ratio
    return {
        "lon_bound": [float(lon_min - lon_pad), float(lon_max + lon_pad)],
        "lat_bound": [float(lat_min - lat_pad), float(lat_max + lat_pad)],
    }


def load_fixed_bounds(path: Path) -> Dict[str, List[float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates = [payload]
    for key in ("bounds", "stats"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)

    for candidate in candidates:
        lon_bound = candidate.get("lon_bound", candidate.get("porto_lon_bound"))
        lat_bound = candidate.get("lat_bound", candidate.get("porto_lat_bound"))
        if lon_bound is None or lat_bound is None:
            continue
        bounds = {
            "lon_bound": [float(lon_bound[0]), float(lon_bound[1])],
            "lat_bound": [float(lat_bound[0]), float(lat_bound[1])],
        }
        if not (bounds["lon_bound"][0] < bounds["lon_bound"][1]):
            raise ValueError(f"Invalid longitude bounds in {path}: {bounds['lon_bound']}")
        if not (bounds["lat_bound"][0] < bounds["lat_bound"][1]):
            raise ValueError(f"Invalid latitude bounds in {path}: {bounds['lat_bound']}")
        return bounds
    raise KeyError(
        f"Could not find lon_bound/lat_bound (or Porto-prefixed equivalents) in {path}"
    )


def point_bounds_stats(points: np.ndarray, bounds: Dict[str, List[float]]) -> Dict[str, float | int]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    finite = np.isfinite(pts).all(axis=1)
    lon_min, lon_max = bounds["lon_bound"]
    lat_min, lat_max = bounds["lat_bound"]
    inside = (
        finite
        & (pts[:, 0] >= lon_min)
        & (pts[:, 0] <= lon_max)
        & (pts[:, 1] >= lat_min)
        & (pts[:, 1] <= lat_max)
    )
    finite_count = int(finite.sum())
    outside_count = int((finite & ~inside).sum())
    return {
        "total_point_count": int(len(pts)),
        "finite_point_count": finite_count,
        "nonfinite_point_count": int((~finite).sum()),
        "out_of_bounds_point_count": outside_count,
        "out_of_bounds_point_rate": float(outside_count / max(finite_count, 1)),
    }


def to_grid(
    points: np.ndarray,
    bounds: Dict[str, List[float]],
    grid_num: int,
    out_of_bounds_policy: OutOfBoundsPolicy = "discard",
) -> np.ndarray:
    lon_min, lon_max = bounds["lon_bound"]
    lat_min, lat_max = bounds["lat_bound"]
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    in_bounds = (
        (pts[:, 0] >= lon_min)
        & (pts[:, 0] <= lon_max)
        & (pts[:, 1] >= lat_min)
        & (pts[:, 1] <= lat_max)
    )
    if out_of_bounds_policy == "error" and not bool(np.all(in_bounds)):
        raise ValueError(f"Found {int((~in_bounds).sum())} points outside fixed evaluation bounds")
    if out_of_bounds_policy == "discard":
        pts = pts[in_bounds]
        in_bounds = np.ones(len(pts), dtype=bool)
    elif out_of_bounds_policy == "clip":
        pts = pts.copy()
        pts[:, 0] = np.clip(pts[:, 0], lon_min, lon_max)
        pts[:, 1] = np.clip(pts[:, 1], lat_min, lat_max)
        in_bounds = np.ones(len(pts), dtype=bool)
    elif out_of_bounds_policy not in {"overflow", "error"}:
        raise ValueError(f"Unknown out-of-bounds policy: {out_of_bounds_policy}")

    hist_size = grid_num * grid_num + (1 if out_of_bounds_policy == "overflow" else 0)
    hist = np.zeros(hist_size, dtype=np.float64)
    if len(pts) == 0:
        return hist
    col = ((pts[:, 0] - lon_min) / max(lon_max - lon_min, 1e-12) * grid_num).astype(int)
    row = ((pts[:, 1] - lat_min) / max(lat_max - lat_min, 1e-12) * grid_num).astype(int)
    ids = np.clip(row, 0, grid_num - 1) * grid_num + np.clip(col, 0, grid_num - 1)
    if out_of_bounds_policy == "overflow":
        ids = ids.astype(np.int64, copy=False)
        ids[~in_bounds] = grid_num * grid_num
    np.add.at(hist, ids, 1)
    return hist


def to_grid_ids(
    points: np.ndarray,
    bounds: Dict[str, List[float]],
    grid_num: int,
    out_of_bounds_policy: OutOfBoundsPolicy = "discard",
) -> np.ndarray:
    lon_min, lon_max = bounds["lon_bound"]
    lat_min, lat_max = bounds["lat_bound"]
    pts = np.asarray(points, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return np.zeros(0, dtype=np.int64)
    in_bounds = (
        (pts[:, 0] >= lon_min)
        & (pts[:, 0] <= lon_max)
        & (pts[:, 1] >= lat_min)
        & (pts[:, 1] <= lat_max)
    )
    if out_of_bounds_policy == "error" and not bool(np.all(in_bounds)):
        raise ValueError(f"Found {int((~in_bounds).sum())} points outside fixed evaluation bounds")
    if out_of_bounds_policy == "discard":
        pts = pts[in_bounds]
        in_bounds = np.ones(len(pts), dtype=bool)
    elif out_of_bounds_policy == "clip":
        pts = pts.copy()
        pts[:, 0] = np.clip(pts[:, 0], lon_min, lon_max)
        pts[:, 1] = np.clip(pts[:, 1], lat_min, lat_max)
        in_bounds = np.ones(len(pts), dtype=bool)
    elif out_of_bounds_policy not in {"overflow", "error"}:
        raise ValueError(f"Unknown out-of-bounds policy: {out_of_bounds_policy}")
    if len(pts) == 0:
        return np.zeros(0, dtype=np.int64)

    col = ((pts[:, 0] - lon_min) / max(lon_max - lon_min, 1e-12) * grid_num).astype(np.int64)
    row = ((pts[:, 1] - lat_min) / max(lat_max - lat_min, 1e-12) * grid_num).astype(np.int64)
    col = np.clip(col, 0, grid_num - 1)
    row = np.clip(row, 0, grid_num - 1)
    ids = row * grid_num + col
    if out_of_bounds_policy == "overflow":
        ids[~in_bounds] = grid_num * grid_num
    return ids


def endpoint_marginal_distribution(
    trajs: np.ndarray,
    bounds: Dict[str, List[float]],
    grid_num: int,
    out_of_bounds_policy: OutOfBoundsPolicy = "discard",
) -> np.ndarray:
    return to_grid(
        np.stack([trajs[:, 0], trajs[:, -1]], axis=1),
        bounds,
        grid_num,
        out_of_bounds_policy=out_of_bounds_policy,
    )


def od_pair_distribution(
    trajs: np.ndarray,
    bounds: Dict[str, List[float]],
    grid_num: int,
    out_of_bounds_policy: OutOfBoundsPolicy = "discard",
) -> np.ndarray:
    num_cells = grid_num * grid_num + (1 if out_of_bounds_policy == "overflow" else 0)
    hist = np.zeros(num_cells * num_cells, dtype=np.float64)
    for traj in np.asarray(trajs, dtype=np.float32):
        ids = to_grid_ids(
            np.stack([traj[0], traj[-1]], axis=0),
            bounds,
            grid_num,
            out_of_bounds_policy=out_of_bounds_policy,
        )
        if len(ids) != 2:
            continue
        hist[int(ids[0]) * num_cells + int(ids[1])] += 1.0
    return hist


def transition_distribution(
    trajs: np.ndarray,
    bounds: Dict[str, List[float]],
    grid_num: int,
    out_of_bounds_policy: OutOfBoundsPolicy = "discard",
) -> np.ndarray:
    """Grid transition distribution over full trajectories.

    Endpoint-marginal JS only compares endpoint occupancy. In OD-conditioned
    generation that can be exactly zero because OD is an input condition. This
    transition distribution measures the generated route's movement pattern
    between origin and destination, so it remains informative for OD-chord
    conditioned macro trajectory generation.
    """

    num_cells = grid_num * grid_num + (1 if out_of_bounds_policy == "overflow" else 0)
    pair_chunks = []
    for traj in np.asarray(trajs, dtype=np.float32):
        ids = to_grid_ids(
            traj,
            bounds,
            grid_num,
            out_of_bounds_policy=out_of_bounds_policy,
        )
        if len(ids) < 2:
            continue
        keep = np.ones(len(ids), dtype=bool)
        keep[1:] = ids[1:] != ids[:-1]
        ids = ids[keep]
        if len(ids) < 2:
            continue
        pair_chunks.append(ids[:-1] * num_cells + ids[1:])
    if not pair_chunks:
        return np.zeros(num_cells * num_cells, dtype=np.float64)
    pair_ids = np.concatenate(pair_chunks, axis=0)
    return np.bincount(pair_ids, minlength=num_cells * num_cells).astype(np.float64)


def length_distribution(trajs: np.ndarray, bins: int, max_len_km: float | None = None) -> np.ndarray:
    lengths = haversine_km(trajs[:, :-1], trajs[:, 1:]).sum(axis=1)
    if max_len_km is None:
        max_len_km = float(np.percentile(lengths, 99))
    hist, _ = np.histogram(np.clip(lengths, 0.0, max_len_km), bins=np.linspace(0.0, max_len_km, bins + 1))
    return hist.astype(np.float64)


def endpoint_error_metrics(real: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
    start_error_km = haversine_km(real[:, 0], generated[:, 0])
    end_error_km = haversine_km(real[:, -1], generated[:, -1])
    mean_endpoint_error_m = 500.0 * float(np.mean(start_error_km + end_error_km))
    max_endpoint_error_m = 1000.0 * np.maximum(start_error_km, end_error_km)
    return {
        "Endpoint Error (m)": mean_endpoint_error_m,
        "Endpoint Max Error P95 (m)": float(np.percentile(max_endpoint_error_m, 95)),
        "OD Adherence <=1m": float(np.mean(max_endpoint_error_m <= 1.0)),
    }


def evaluate_distribution(
    real: np.ndarray,
    generated: np.ndarray,
    grid_num: int = 32,
    length_bins: int = 32,
    bounds: Dict[str, List[float]] | None = None,
    bounds_source: str | None = None,
    out_of_bounds_policy: OutOfBoundsPolicy = "discard",
) -> Dict:
    n = min(len(real), len(generated))
    real = np.asarray(real[:n], dtype=np.float32)
    generated = np.asarray(generated[:n], dtype=np.float32)
    if bounds is None:
        bounds = infer_bounds(real, generated)
        bounds_source = bounds_source or "real_generated_union_legacy"
    else:
        bounds = {
            "lon_bound": [float(bounds["lon_bound"][0]), float(bounds["lon_bound"][1])],
            "lat_bound": [float(bounds["lat_bound"][0]), float(bounds["lat_bound"][1])],
        }
        bounds_source = bounds_source or "caller_supplied_fixed_bounds"
    real_density = to_grid(real, bounds, grid_num, out_of_bounds_policy)
    gen_density = to_grid(generated, bounds, grid_num, out_of_bounds_policy)
    real_route_transition = transition_distribution(real, bounds, grid_num, out_of_bounds_policy)
    gen_route_transition = transition_distribution(generated, bounds, grid_num, out_of_bounds_policy)
    real_od_pair = od_pair_distribution(real, bounds, grid_num, out_of_bounds_policy)
    gen_od_pair = od_pair_distribution(generated, bounds, grid_num, out_of_bounds_policy)
    real_endpoint_marginal = endpoint_marginal_distribution(real, bounds, grid_num, out_of_bounds_policy)
    gen_endpoint_marginal = endpoint_marginal_distribution(generated, bounds, grid_num, out_of_bounds_policy)
    real_lengths = haversine_km(real[:, :-1], real[:, 1:]).sum(axis=1)
    max_len = float(np.ceil(np.percentile(real_lengths, 99) / 10.0) * 10.0)
    metrics = {
        "num_compared": int(n),
        "Density Error": js_divergence(real_density, gen_density),
        "Endpoint Marginal JSD": js_divergence(real_endpoint_marginal, gen_endpoint_marginal),
        "Endpoint Marginal Protocol": "endpoint_marginal_js",
        "Route Transition JSD": js_divergence(real_route_transition, gen_route_transition),
        "Route Transition Protocol": "grid_transition_js",
        "OD Pair JSD": js_divergence(real_od_pair, gen_od_pair),
        "Length Error": js_divergence(
            length_distribution(real, length_bins, max_len),
            length_distribution(generated, length_bins, max_len),
        ),
        "gt_avg_length_km": float(real_lengths.mean()),
        "gen_avg_length_km": float(haversine_km(generated[:, :-1], generated[:, 1:]).sum(axis=1).mean()),
        "bounds": bounds,
        "bounds_source": bounds_source,
        "grid_num": int(grid_num),
        "length_bins": int(length_bins),
        "out_of_bounds_policy": out_of_bounds_policy,
        "real_bounds_stats": point_bounds_stats(real, bounds),
        "generated_bounds_stats": point_bounds_stats(generated, bounds),
    }
    metrics.update(endpoint_error_metrics(real, generated))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ControlTraj/DirectTraj-style continuous distribution metrics.")
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--real-key", default="trajectories")
    parser.add_argument("--generated-key", default="trajectories")
    parser.add_argument("--grid-num", type=int, default=32)
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
        help="Stable provenance label written to the metrics JSON.",
    )
    parser.add_argument(
        "--out-of-bounds-policy",
        choices=("discard", "clip", "overflow", "error"),
        default=None,
        help="Defaults to overflow with --bounds-json and discard for legacy inferred bounds.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bounds = load_fixed_bounds(args.bounds_json) if args.bounds_json else None
    out_of_bounds_policy = args.out_of_bounds_policy or ("overflow" if bounds is not None else "discard")
    metrics = evaluate_distribution(
        load_array(args.real, args.real_key),
        load_array(args.generated, args.generated_key),
        grid_num=args.grid_num,
        length_bins=args.length_bins,
        bounds=bounds,
        bounds_source=args.bounds_source or (str(args.bounds_json.resolve()) if args.bounds_json else None),
        out_of_bounds_policy=out_of_bounds_policy,
    )
    metrics.update({"real": str(args.real), "generated": str(args.generated)})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
