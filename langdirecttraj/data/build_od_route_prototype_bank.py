from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _iter_records(path: Path, max_samples: int | None) -> Iterable[dict]:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if max_samples is not None and count >= int(max_samples):
                break


def _resample(points: np.ndarray, target_len: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected trajectory [T, 2], got {points.shape}")
    if len(points) == int(target_len):
        return points
    if len(points) < 2:
        return np.repeat(points[:1], int(target_len), axis=0)
    src = np.linspace(0.0, 1.0, len(points), dtype=np.float32)
    dst = np.linspace(0.0, 1.0, int(target_len), dtype=np.float32)
    out = np.stack([np.interp(dst, src, points[:, dim]) for dim in range(2)], axis=-1)
    return out.astype(np.float32)


def _od_local_offsets(traj: np.ndarray, points: int) -> np.ndarray | None:
    traj = _resample(traj, int(points))
    start = traj[0]
    end = traj[-1]
    chord = end - start
    length = float(np.linalg.norm(chord))
    if not math.isfinite(length) or length < 1e-8:
        return None
    tangent = chord / length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    t = np.linspace(0.0, 1.0, int(points), dtype=np.float32)[:, None]
    reference = start[None, :] + t * chord[None, :]
    delta = traj - reference
    local = np.stack([delta @ tangent, delta @ normal], axis=-1) / max(length, 1e-8)
    local[0] = 0.0
    local[-1] = 0.0
    return local.astype(np.float32)


def _trajectory_features(local: np.ndarray) -> np.ndarray:
    tangent = local[:, 0]
    normal = local[:, 1]
    diffs = np.diff(local, axis=0)
    second = np.diff(diffs, axis=0) if len(diffs) >= 2 else np.zeros((0, 2), dtype=np.float32)
    return np.asarray(
        [
            float(np.mean(normal)),
            float(np.mean(np.abs(normal))),
            float(np.max(np.abs(normal))),
            float(np.mean(tangent)),
            float(np.mean(np.abs(tangent))),
            float(np.mean(np.linalg.norm(second, axis=-1))) if len(second) else 0.0,
        ],
        dtype=np.float32,
    )


def _kmeans(data: np.ndarray, k: int, seed: int, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    n = data.shape[0]
    k = min(int(k), n)
    flat = data.reshape(n, -1)
    centers = np.empty((k, flat.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, n))
    centers[0] = flat[first]
    dist = np.linalg.norm(flat - centers[0], axis=1)
    for idx in range(1, k):
        pick = int(np.argmax(dist))
        centers[idx] = flat[pick]
        dist = np.minimum(dist, np.linalg.norm(flat - centers[idx], axis=1))
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max(1, int(iterations))):
        dists = np.linalg.norm(flat[:, None, :] - centers[None, :, :], axis=-1)
        labels = dists.argmin(axis=1)
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                centers[idx] = flat[mask].mean(axis=0)
    return centers.reshape(k, *data.shape[1:]).astype(np.float32), labels


def _sort_prototypes(prototypes: np.ndarray) -> np.ndarray:
    features = np.stack([_trajectory_features(proto) for proto in prototypes], axis=0)
    signed = features[:, 0]
    abs_mean = features[:, 1]
    curvature = features[:, 5]
    order: list[int] = []

    def add(idx: int) -> None:
        if idx not in order:
            order.append(int(idx))

    add(int(np.argmin(abs_mean)))
    add(int(np.argmax(signed)))
    add(int(np.argmin(signed)))
    for idx in np.argsort(-(abs_mean + curvature)):
        add(int(idx))
    return prototypes[order]


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    span = float(values.max() - values.min())
    if not math.isfinite(span) or span < 1e-8:
        return np.full_like(values, 0.5, dtype=np.float32)
    return ((values - values.min()) / span).astype(np.float32)


def _inv_rank01(values: np.ndarray) -> np.ndarray:
    return (1.0 - _rank01(values)).astype(np.float32)


def _row_normalize(table: np.ndarray) -> np.ndarray:
    table = np.asarray(table, dtype=np.float32)
    if table.ndim != 2:
        raise ValueError(f"Expected table [K, D], got {table.shape}")
    table = np.nan_to_num(table, nan=0.0, posinf=1.0, neginf=0.0)
    table = np.clip(table, 0.0, 1.0)
    row_max = table.max(axis=1, keepdims=True)
    return np.where(row_max > 1e-8, table / row_max, table).astype(np.float32)


def _build_role_tables(features: np.ndarray) -> dict[str, list]:
    """Infer decoder mode-role priors from OD-local prototype geometry.

    The roles are distribution-level prototype attributes, not per-sample
    reference paths. They align prototype_bank modes with the same role tables
    used by decoder priors and training-time positive assignment.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 6:
        raise ValueError(f"Expected prototype features [K, 6], got {features.shape}")

    signed_normal = features[:, 0]
    abs_normal = features[:, 1]
    max_abs_normal = features[:, 2]
    signed_tangent = features[:, 3]
    abs_tangent = features[:, 4]
    curvature = features[:, 5]

    ref_offset = 0.65 * abs_normal + 0.35 * max_abs_normal
    route_complexity = 0.45 * abs_normal + 0.25 * max_abs_normal + 0.30 * curvature
    direction_strength = np.abs(signed_normal) + 0.5 * max_abs_normal

    close_ref = _inv_rank01(ref_offset)
    smooth = _inv_rank01(0.55 * curvature + 0.45 * abs_normal)
    loose_ref = _rank01(ref_offset)
    winding = _rank01(0.65 * curvature + 0.35 * max_abs_normal)
    fast_progress = _rank01(signed_tangent + 0.25 * abs_tangent)
    slow_progress = _rank01(-signed_tangent + 0.25 * abs_tangent)
    mode_role_table = _row_normalize(
        np.stack([close_ref, smooth, loose_ref, winding, fast_progress, slow_progress], axis=1)
    )

    endpoint_scale = 0.24 + 0.34 * smooth + 0.10 * fast_progress
    mid_scale = 0.36 + 0.34 * loose_ref + 0.26 * winding
    score_scale = 0.70 + 0.30 * _inv_rank01(route_complexity)
    mode_role_budget_table = np.stack([endpoint_scale, mid_scale, score_scale], axis=1).astype(np.float32)

    tangent_scale_bias = 0.45 * (fast_progress - slow_progress)
    normal_scale_bias = 0.65 * loose_ref + 0.45 * winding - 0.45 * close_ref
    gate_bias = 0.25 * winding + 0.18 * loose_ref - 0.20 * smooth
    mode_role_control_table = np.stack([tangent_scale_bias, normal_scale_bias, gate_bias], axis=1).astype(np.float32)

    straight = _inv_rank01(direction_strength + 0.5 * curvature)
    # Under the endpoint-preserving OD-local contract, a left-turn arc has a
    # negative middle normal offset and positive signed heading change. Keep
    # this aligned with direction_control_loss_alignment tests.
    left_turn = _rank01(np.maximum(-signed_normal, 0.0))
    right_turn = _rank01(np.maximum(signed_normal, 0.0))
    u_turn = _rank01(0.55 * max_abs_normal + 0.45 * curvature)
    multi_turn = _rank01(0.60 * curvature + 0.25 * abs_normal + 0.15 * abs_tangent)
    direction_mode_role_table = _row_normalize(
        np.stack([straight, left_turn, right_turn, u_turn, multi_turn], axis=1)
    )

    direct = _inv_rank01(route_complexity)
    left_bias = left_turn
    right_bias = right_turn
    detour = _rank01(0.55 * ref_offset + 0.30 * abs_tangent + 0.15 * curvature)
    route_transition_mode_table = _row_normalize(
        np.stack([direct, left_bias, right_bias, u_turn, multi_turn, detour], axis=1)
    )

    return {
        "mode_role_table": mode_role_table.tolist(),
        "mode_role_budget_table": mode_role_budget_table.tolist(),
        "mode_role_control_table": mode_role_control_table.tolist(),
        "direction_mode_role_table": direction_mode_role_table.tolist(),
        "route_transition_mode_table": route_transition_mode_table.tolist(),
        "mode_role_table_columns": [
            "close_ref",
            "smooth",
            "loose_ref",
            "winding",
            "fast_progress",
            "slow_progress",
        ],
        "mode_role_budget_columns": ["endpoint_scale", "mid_scale", "score_scale"],
        "mode_role_control_columns": ["tangent_scale_bias", "normal_scale_bias", "gate_bias"],
        "direction_mode_role_columns": ["straight", "left_turn", "right_turn", "u_turn", "multi_turn"],
        "route_transition_mode_columns": ["direct", "left_bias", "right_bias", "u_turn", "multi_turn", "detour"],
    }


def build_bank(args: argparse.Namespace) -> dict:
    locals_: list[np.ndarray] = []
    records_seen = 0
    skipped = 0
    skipped_outlier = 0
    for record in _iter_records(args.input, args.max_samples):
        records_seen += 1
        if args.target_key not in record:
            skipped += 1
            continue
        local = _od_local_offsets(np.asarray(record[args.target_key], dtype=np.float32), args.points)
        if local is None:
            skipped += 1
            continue
        if float(np.max(np.abs(local))) > float(args.max_normalized_offset):
            skipped_outlier += 1
            continue
        locals_.append(local)
    if not locals_:
        raise RuntimeError(f"No valid trajectories found in {args.input} with key {args.target_key}")
    data = np.stack(locals_, axis=0).astype(np.float32)
    prototypes, labels = _kmeans(data, args.num_prototypes, args.seed, args.kmeans_iters)
    prototypes = _sort_prototypes(prototypes)
    if prototypes.shape[0] < int(args.num_prototypes):
        pad = np.repeat(prototypes[-1:], int(args.num_prototypes) - prototypes.shape[0], axis=0)
        prototypes = np.concatenate([prototypes, pad], axis=0)
    features = np.stack([_trajectory_features(proto) for proto in prototypes], axis=0)
    role_tables = _build_role_tables(features)
    return {
        "protocol": "porto_od_chord_v1_route_prototype_bank",
        "input": str(args.input),
        "target_key": args.target_key,
        "num_records_seen": int(records_seen),
        "num_valid": int(data.shape[0]),
        "num_skipped": int(skipped),
        "num_skipped_outlier": int(skipped_outlier),
        "max_normalized_offset": float(args.max_normalized_offset),
        "num_prototypes": int(args.num_prototypes),
        "points": int(args.points),
        "normalization": "OD-local tangent/normal offsets divided by OD chord length",
        "leakage_guard": (
            "Built only as a training-set distribution artifact. At inference the current sample "
            "provides only OD chord geometry; no per-sample GT middle trajectory is used as condition."
        ),
        "cluster_counts": [int((labels == idx).sum()) for idx in range(min(args.num_prototypes, labels.max() + 1))],
        "prototype_features": [
            {
                "signed_normal_mean": float(row[0]),
                "abs_normal_mean": float(row[1]),
                "max_abs_normal": float(row[2]),
                "signed_tangent_mean": float(row[3]),
                "abs_tangent_mean": float(row[4]),
                "curvature_proxy": float(row[5]),
            }
            for row in features
        ],
        **role_tables,
        "prototypes": prototypes[: int(args.num_prototypes)].tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build non-leaking OD-local route prototype bank.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-key", default="pseudo_trajectory")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "route_prototypes" / "porto_od_chord_v1_route_prototypes.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-prototypes", type=int, default=6)
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument(
        "--max-normalized-offset",
        type=float,
        default=3.0,
        help=(
            "Drop trajectories whose OD-local tangent/normal offset exceeds this value after "
            "division by chord length. This prevents tiny-OD outliers from dominating prototypes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_bank(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in payload.items() if key != "prototypes"}
    summary["output"] = str(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
