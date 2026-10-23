from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from langdirecttraj.data.export_baseline_dataset import tokenize_trajectory
from langdirecttraj.data.instruction_dataset import InstructionTrajectoryDataset, validate_official_record_protocol


REPO_ROOT = Path(__file__).resolve().parents[2]


def _transition_key(src: int, dst: int) -> str:
    return f"{int(src)}:{int(dst)}"


def iter_transitions(tokens: Iterable[int]) -> Iterable[tuple[int, int]]:
    seq = [int(x) for x in tokens]
    if len(seq) < 2:
        return
    for src, dst in zip(seq[:-1], seq[1:]):
        if src == dst:
            continue
        yield int(src), int(dst)


def _haversine_m(a: np.ndarray, b: np.ndarray) -> float:
    lon1, lat1 = math.radians(float(a[0])), math.radians(float(a[1]))
    lon2, lat2 = math.radians(float(b[0])), math.radians(float(b[1]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6_371_008.8 * 2.0 * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def _od_bearing_sector(traj: np.ndarray, bins: int) -> int:
    start, end = traj[0], traj[-1]
    lat = math.radians(float((start[1] + end[1]) * 0.5))
    dx = (float(end[0]) - float(start[0])) * math.cos(lat)
    dy = float(end[1]) - float(start[1])
    angle = math.atan2(dy, dx)
    sector = int(math.floor(((angle + math.pi) / (2.0 * math.pi)) * bins))
    return int(np.clip(sector, 0, bins - 1))


def _distance_bin(distance_m: float, thresholds: List[float]) -> int:
    out = 0
    for threshold in thresholds:
        if distance_m > float(threshold):
            out += 1
    return int(out)


def _nll_payload_from_counts(counts: Counter[int] | Counter[str], alpha: float) -> tuple[float, Dict]:
    num_items = max(len(counts), 1)
    total = float(sum(counts.values()))
    denom = total + alpha * float(num_items + 1)
    default_nll = -math.log(max(alpha / max(denom, 1e-12), 1e-12))
    values = {
        str(key): {
            "count": int(count),
            "nll": float(-math.log(max((float(count) + alpha) / max(denom, 1e-12), 1e-12))),
        }
        for key, count in sorted(counts.items(), key=lambda item: str(item[0]))
    }
    return float(default_nll), values


def infer_bounds_for_key(records: List[Dict], target_key: str, padding_ratio: float = 0.02) -> Dict[str, List[float]]:
    pts = []
    for index, record in enumerate(records):
        if target_key not in record:
            raise KeyError(
                f"Record {index} is missing target_key={target_key!r}; "
                f"available keys={sorted(record.keys())}"
            )
        pts.append(np.asarray(record[target_key], dtype=np.float32))
    all_pts = np.concatenate(pts, axis=0)
    lon_min, lat_min = all_pts.min(axis=0)
    lon_max, lat_max = all_pts.max(axis=0)
    lon_pad = max(float(lon_max - lon_min) * padding_ratio, 1e-6)
    lat_pad = max(float(lat_max - lat_min) * padding_ratio, 1e-6)
    return {
        "lon_bound": [float(lon_min - lon_pad), float(lon_max + lon_pad)],
        "lat_bound": [float(lat_min - lat_pad), float(lat_max + lat_pad)],
    }


def build_global_transition_prior(args: argparse.Namespace) -> Dict:
    dataset = InstructionTrajectoryDataset(
        args.input,
        max_samples=args.max_samples,
        target_key=args.target_key,
    )
    records = [dataset.get_record(index) for index in range(len(dataset))]
    for row_index, record in enumerate(records):
        validate_official_record_protocol(record, row_index=row_index)

    bounds = infer_bounds_for_key(records, args.target_key, padding_ratio=args.grid_padding_ratio)
    counts: Counter[str] = Counter()
    cell_counts: Counter[int] = Counter()
    from_counts: Counter[int] = Counter()
    token_lengths: List[int] = []
    encoded_records = []
    chord_distances = []
    for record in records:
        traj = np.asarray(record[args.target_key], dtype=np.float32)
        tokens = tokenize_trajectory(
            traj,
            bounds,
            args.grid_size,
        )
        if tokens:
            cell_counts.update(int(token) for token in tokens)
        encoded_records.append((traj, tokens))
        chord_distances.append(_haversine_m(traj[0], traj[-1]))
        token_lengths.append(len(tokens))
        for src, dst in iter_transitions(tokens):
            counts[_transition_key(src, dst)] += 1
            from_counts[int(src)] += 1

    num_cells = int(args.grid_size) * int(args.grid_size)
    alpha = float(args.smoothing)
    default_nll, transitions = _nll_payload_from_counts(counts, alpha)
    cell_default_nll, cells = _nll_payload_from_counts(cell_counts, alpha)

    distance_bins = max(1, int(args.od_distance_bins))
    if distance_bins > 1:
        quantiles = [100.0 * i / distance_bins for i in range(1, distance_bins)]
        distance_thresholds = [
            float(np.percentile(np.asarray(chord_distances, dtype=np.float32), q))
            for q in quantiles
        ]
    else:
        distance_thresholds = []
    bearing_bins = max(1, int(args.od_bearing_bins))
    grouped_cell_counts: Dict[str, Counter[int]] = {}
    grouped_length_ratios: Dict[str, List[float]] = {}
    length_ratios: List[float] = []
    for traj, tokens in encoded_records:
        sector = _od_bearing_sector(traj, bearing_bins)
        chord_m = _haversine_m(traj[0], traj[-1])
        dbin = _distance_bin(chord_m, distance_thresholds)
        group_key = f"s{sector}_d{dbin}"
        grouped_cell_counts.setdefault(group_key, Counter())
        grouped_cell_counts[group_key].update(int(token) for token in tokens)
        step_m = [
            _haversine_m(a, b)
            for a, b in zip(traj[:-1], traj[1:])
        ]
        route_m = float(sum(step_m))
        ratio = route_m / max(float(chord_m), 100.0)
        if math.isfinite(ratio) and ratio > 0.0:
            grouped_length_ratios.setdefault(group_key, []).append(float(ratio))
            length_ratios.append(float(ratio))

    cell_groups = {}
    length_groups = {}
    global_length_ratios = np.asarray(length_ratios, dtype=np.float32)
    if len(global_length_ratios) == 0:
        global_length_ratios = np.asarray([1.0], dtype=np.float32)
    global_length_stats = {
        "count": int(len(global_length_ratios)),
        "p10": float(np.percentile(global_length_ratios, 10)),
        "p50": float(np.percentile(global_length_ratios, 50)),
        "p90": float(np.percentile(global_length_ratios, 90)),
    }
    for sector in range(bearing_bins):
        for dbin in range(distance_bins):
            group_key = f"s{sector}_d{dbin}"
            group_counts = grouped_cell_counts.get(group_key, Counter())
            if sum(group_counts.values()) < int(args.od_min_group_cell_count):
                group_counts = cell_counts
            group_default, group_cells = _nll_payload_from_counts(group_counts, alpha)
            cell_groups[group_key] = {
                "default_nll": group_default,
                "num_cells_observed": int(len(group_counts)),
                "total_cell_visits": int(sum(group_counts.values())),
                "cells": group_cells,
            }
            group_ratios = np.asarray(grouped_length_ratios.get(group_key, []), dtype=np.float32)
            if len(group_ratios) < int(args.od_min_group_length_count):
                group_ratios = global_length_ratios
            length_groups[group_key] = {
                "count": int(len(group_ratios)),
                "p10": float(np.percentile(group_ratios, 10)),
                "p50": float(np.percentile(group_ratios, 50)),
                "p90": float(np.percentile(group_ratios, 90)),
            }

    lengths = np.asarray(token_lengths, dtype=np.float32)
    total = float(sum(counts.values()))
    payload = {
        "dataset": args.dataset,
        "input": str(args.input),
        "target_key": args.target_key,
        "num_records": int(len(records)),
        "grid_size": int(args.grid_size),
        "num_cells": int(num_cells),
        "bounds": bounds,
        "smoothing": float(alpha),
        "total_transitions": int(total),
        "num_unique_transitions": int(len(counts)),
        "default_nll": float(default_nll),
        "transitions": transitions,
        "cell_default_nll": float(cell_default_nll),
        "cells": cells,
        "od_cell_prior": {
            "bearing_bins": int(bearing_bins),
            "distance_bins": int(distance_bins),
            "distance_thresholds_m": distance_thresholds,
            "min_group_cell_count": int(args.od_min_group_cell_count),
            "groups": cell_groups,
        },
        "od_length_prior": {
            "bearing_bins": int(bearing_bins),
            "distance_bins": int(distance_bins),
            "distance_thresholds_m": distance_thresholds,
            "min_group_length_count": int(args.od_min_group_length_count),
            "global": global_length_stats,
            "groups": length_groups,
        },
        "token_length": {
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "p50": float(np.percentile(lengths, 50)) if len(lengths) else 0.0,
            "p90": float(np.percentile(lengths, 90)) if len(lengths) else 0.0,
        },
        "protocol": {
            "name": "porto_od_chord_v1",
            "condition_source": "origin_destination_chord",
            "prior_source": "training_set_global_transition_and_od_cell_statistics",
            "leakage_guard": (
                "This file stores only aggregate transition and OD-bucket cell statistics from the training split. "
                "It is not a per-sample reference and is not used as a model condition."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a non-leaking global grid-transition prior.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-key", default="trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--grid-padding-ratio", type=float, default=0.02)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--od-bearing-bins", type=int, default=8)
    parser.add_argument("--od-distance-bins", type=int, default=3)
    parser.add_argument("--od-min-group-cell-count", type=int, default=32)
    parser.add_argument("--od-min-group-length-count", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_global_transition_prior(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
