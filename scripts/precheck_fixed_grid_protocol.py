#!/usr/bin/env python
"""Precheck the fixed-grid HDM/Table-2 evaluation protocol before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langdirecttraj.eval.distribution_metrics import (  # noqa: E402
    evaluate_distribution,
    load_fixed_bounds,
    to_grid,
)
from scripts.eval_coarse_road_cell_stage import build_coarse_prior  # noqa: E402


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def default_workspace_path(*parts: str) -> Path:
    return first_existing(REPO_ROOT.joinpath(*parts), REPO_ROOT.parent.joinpath(*parts))


def default_real_path(city: str, split: str) -> Path:
    if city == "porto":
        return default_workspace_path("outputs", "road_full", "real", f"{city}_{split}_real_trajectories.npz")
    return default_workspace_path("outputs", "road_full", city, "real", f"{city}_{split}_real_trajectories.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="porto")
    parser.add_argument("--grid-num", type=int, default=16)
    parser.add_argument("--od-grid", type=int, default=6)
    parser.add_argument("--length-bins", type=int, default=32)
    parser.add_argument("--out-of-bounds-policy", default="overflow", choices=["discard", "clip", "overflow", "error"])
    parser.add_argument("--max-train", type=int, default=2048)
    parser.add_argument("--bounds-json", type=Path, default=None)
    parser.add_argument("--train-real", type=Path, default=None)
    parser.add_argument("--val-real", type=Path, default=None)
    parser.add_argument("--test-real", type=Path, default=None)
    return parser.parse_args()


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    payload = np.load(path)
    traj = np.asarray(payload["real_trajectories"], dtype=np.float32)
    ids = None
    for key in ("sample_id", "sample_ids"):
        if key in payload.files:
            ids = np.asarray(payload[key], dtype=np.int64)
            break
    return traj, ids


def id_overlap(a: np.ndarray | None, b: np.ndarray | None) -> int | None:
    if a is None or b is None:
        return None
    return int(np.intersect1d(a, b, assume_unique=False).size)


def main() -> None:
    args = parse_args()
    city = str(args.city)
    bounds_json = args.bounds_json or default_workspace_path(
        "DiffTraj",
        "data",
        f"{city}_od_only",
        f"{city}_global_stats.json",
    )
    train_path = args.train_real or default_real_path(city, "train")
    val_path = args.val_real or default_real_path(city, "val")
    test_path = args.test_real or default_real_path(city, "test")

    missing = [str(path) for path in (bounds_json, train_path, val_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required protocol inputs:\n" + "\n".join(missing))

    bounds = load_fixed_bounds(bounds_json)
    train, train_ids = load_npz(train_path)
    val, val_ids = load_npz(val_path)
    test, test_ids = load_npz(test_path)

    prior_train = train[: min(int(args.max_train), len(train))]
    prior = build_coarse_prior(
        prior_train,
        grid_num=int(args.grid_num),
        od_grid=int(args.od_grid),
        length_bins=int(args.length_bins),
        bounds=bounds,
        bounds_source=str(bounds_json.resolve()),
        out_of_bounds_policy=str(args.out_of_bounds_policy),
    )
    density = to_grid(prior_train, prior["bounds"], int(args.grid_num), str(args.out_of_bounds_policy))
    sanity_metrics = evaluate_distribution(
        val[: min(16, len(val))],
        val[: min(16, len(val))],
        grid_num=int(args.grid_num),
        length_bins=int(args.length_bins),
        bounds=prior["bounds"],
        bounds_source=prior["bounds_source"],
        out_of_bounds_policy=str(args.out_of_bounds_policy),
    )

    expected_cells = int(args.grid_num) * int(args.grid_num) + (1 if args.out_of_bounds_policy == "overflow" else 0)
    expected_od_cells = int(args.od_grid) * int(args.od_grid) + (1 if args.out_of_bounds_policy == "overflow" else 0)
    checks = {
        "bounds_json_exists": bounds_json.exists(),
        "fixed_bounds_applied": prior["bounds"] == bounds,
        "num_cells_ok": int(prior["num_cells"]) == expected_cells and int(density.shape[0]) == expected_cells,
        "od_cells_ok": int(prior["od_num_cells"]) == expected_od_cells,
        "self_density_zero": abs(float(sanity_metrics["Density Error"])) < 1e-12,
        "self_route_zero": abs(float(sanity_metrics["Route Transition JSD"])) < 1e-12,
        "self_endpoint_error_zero": abs(float(sanity_metrics["Endpoint Error (m)"])) < 1e-6,
        "train_val_id_overlap": id_overlap(train_ids, val_ids),
        "train_test_id_overlap": id_overlap(train_ids, test_ids),
        "val_test_id_overlap": id_overlap(val_ids, test_ids),
    }
    disjoint_ok = all(
        value in (0, None)
        for key, value in checks.items()
        if key.endswith("_id_overlap")
    )
    passed = all(value is True for key, value in checks.items() if not key.endswith("_id_overlap")) and disjoint_ok
    report = {
        "passed": bool(passed),
        "city": city,
        "grid_num": int(args.grid_num),
        "length_bins": int(args.length_bins),
        "out_of_bounds_policy": str(args.out_of_bounds_policy),
        "bounds_source": str(bounds_json.resolve()),
        "bounds": bounds,
        "train_shape": list(train.shape),
        "val_shape": list(val.shape),
        "test_shape": list(test.shape),
        "prior_num_cells": int(prior["num_cells"]),
        "prior_od_num_cells": int(prior["od_num_cells"]),
        "checks": checks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
