from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from langdirecttraj.eval.distribution_metrics import evaluate_distribution


def _read_jsonl(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_samples > 0 and len(records) >= max_samples:
                break
    return records


def _as_traj_array(records: Iterable[Dict[str, Any]], key: str) -> np.ndarray:
    trajs = []
    for idx, record in enumerate(records):
        if key not in record:
            raise KeyError(f"Record {idx} has no trajectory key '{key}'")
        traj = np.asarray(record[key], dtype=np.float32)
        if traj.ndim != 2 or traj.shape[-1] != 2:
            raise ValueError(f"Record {idx} key '{key}' must have shape [T, 2], got {traj.shape}")
        trajs.append(traj)
    if not trajs:
        raise ValueError("No trajectories loaded")
    return np.stack(trajs, axis=0).astype(np.float32)


def _route_jsd(metrics: Dict[str, Any]) -> float:
    return float(metrics["Route Transition JSD"])


def _summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _gt_split_trials(
    trajs: np.ndarray,
    *,
    trials: int,
    sample_size: int,
    seed: int,
    grid_num: int,
    length_bins: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = int(len(trajs))
    half = min(int(sample_size), n // 2)
    if half <= 0:
        raise ValueError("Need at least two trajectories for GT split calibration")
    values: List[float] = []
    density_values: List[float] = []
    length_values: List[float] = []
    for _ in range(max(1, int(trials))):
        perm = rng.permutation(n)
        a = trajs[perm[:half]]
        b = trajs[perm[half : half + half]]
        metrics = evaluate_distribution(a, b, grid_num=grid_num, length_bins=length_bins)
        values.append(_route_jsd(metrics))
        density_values.append(float(metrics["Density Error"]))
        length_values.append(float(metrics["Length Error"]))
    return {
        "num_trials": int(max(1, int(trials))),
        "sample_size_per_side": int(half),
        "Route Transition JSD": _summarize(values),
        "Density Error": _summarize(density_values),
        "Length Error": _summarize(length_values),
    }


def _load_model_route_metrics(paths: List[Path], od_chord_route_jsd: float | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        candidates: List[tuple[str, Dict[str, Any]]] = []
        if "table1" in payload:
            candidates.append(("table1", payload["table1"]))
        for key, value in payload.items():
            if isinstance(value, dict) and "table1" in value:
                candidates.append((key, value["table1"]))
        for section, metrics in candidates:
            if "Route Transition JSD" not in metrics:
                continue
            route_jsd = float(metrics["Route Transition JSD"])
            row = {
                "path": str(path),
                "section": section,
                "Density Error": float(metrics.get("Density Error", float("nan"))),
                "Route Transition JSD": route_jsd,
                "Length Error": float(metrics.get("Length Error", float("nan"))),
                "num_compared": int(metrics.get("num_compared", 0)),
            }
            if od_chord_route_jsd is not None and od_chord_route_jsd > 0:
                row["route_transition_vs_od_chord_ratio"] = float(route_jsd / od_chord_route_jsd)
            rows.append(row)
    rows.sort(key=lambda item: float(item["Route Transition JSD"]))
    return rows


def evaluate_scale(args: argparse.Namespace) -> Dict[str, Any]:
    records = _read_jsonl(args.input, args.max_samples)
    real = _as_traj_array(records, args.trajectory_key)
    reference = _as_traj_array(records, args.reference_key)
    n = min(len(real), len(reference))
    real = real[:n]
    reference = reference[:n]

    identity = evaluate_distribution(real, real, grid_num=args.grid_num, length_bins=args.length_bins)
    od_chord = evaluate_distribution(real, reference, grid_num=args.grid_num, length_bins=args.length_bins)
    gt_split = _gt_split_trials(
        real,
        trials=args.trials,
        sample_size=args.split_size,
        seed=args.seed,
        grid_num=args.grid_num,
        length_bins=args.length_bins,
    )
    od_chord_route_jsd = float(od_chord["Route Transition JSD"])
    gt_split_route_mean = float(gt_split["Route Transition JSD"]["mean"])
    model_rows = _load_model_route_metrics(args.model_metrics, od_chord_route_jsd=od_chord_route_jsd)

    if gt_split_route_mean <= od_chord_route_jsd:
        target_good = gt_split_route_mean + 0.35 * max(od_chord_route_jsd - gt_split_route_mean, 0.0)
        target_acceptable = gt_split_route_mean + 0.55 * max(od_chord_route_jsd - gt_split_route_mean, 0.0)
        scale_note = (
            "GT split is below OD-chord, so it can be treated as a sample-level lower reference "
            "for this run."
        )
    else:
        target_good = 1.10 * od_chord_route_jsd
        target_acceptable = 1.35 * od_chord_route_jsd
        scale_note = (
            "GT split is above OD-chord because this is a paired OD-conditioned metric and "
            "random GT subsets have different OD/route mixtures. Do not treat GT split as a "
            "lower bound here; judge route-transition quality mainly relative to OD-chord and "
            "same-sample model branches."
        )

    return {
        "input": str(args.input),
        "protocol_expected": "porto_od_chord_v1",
        "num_records": int(n),
        "grid_num": int(args.grid_num),
        "length_bins": int(args.length_bins),
        "trajectory_key": args.trajectory_key,
        "reference_key": args.reference_key,
        "identity_sanity": {
            "Density Error": float(identity["Density Error"]),
            "Route Transition JSD": float(identity["Route Transition JSD"]),
            "Length Error": float(identity["Length Error"]),
        },
        "gt_split_sample_variance": gt_split,
        "od_chord_weak_baseline": {
            "Density Error": float(od_chord["Density Error"]),
            "Route Transition JSD": od_chord_route_jsd,
            "Length Error": float(od_chord["Length Error"]),
        },
        "working_thresholds": {
            "good_route_transition_jsd": float(target_good),
            "acceptable_route_transition_jsd": float(target_acceptable),
            "interpretation": scale_note,
        },
        "model_metrics": model_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate Route Transition JSD against GT-split noise and OD-chord baseline."
    )
    parser.add_argument("--input", type=Path, required=True, help="porto_od_chord_v1 jsonl file.")
    parser.add_argument("--trajectory-key", default="trajectory")
    parser.add_argument("--reference-key", default="reference")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--split-size", type=int, default=32)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--grid-num", type=int, default=32)
    parser.add_argument("--length-bins", type=int, default=32)
    parser.add_argument("--model-metrics", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_scale(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
