from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from langdirecttraj.instructions import mine_instruction_record


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent


def _read_stats(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dataset_dir(dataset: str, data_root: Path | None) -> Path:
    if data_root is not None:
        return data_root / dataset
    candidates = [
        REPO_ROOT / "data" / dataset,
        WORKSPACE_ROOT / "AnchorTraj" / "data" / dataset,
        WORKSPACE_ROOT / "DirectTraj" / "data" / dataset,
    ]
    for candidate in candidates:
        if (candidate / f"{dataset}_trajs.npy").exists():
            return candidate
    searched = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find dataset '{dataset}'. Searched:\n{searched}")


def _select_indices(total: int, max_samples: int, seed: int, stride: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= total:
        if stride > 1:
            return np.arange(0, total, stride, dtype=np.int64)
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=max_samples, replace=False)).astype(np.int64)


def _counter_to_dict(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items())}


def _summarize(records: Iterable[Dict]) -> Dict:
    label_counts = defaultdict(Counter)
    length_values: List[float] = []
    ref_values: List[float] = []
    count = 0
    for record in records:
        count += 1
        for key, value in record["labels"].items():
            label_counts[key][value] += 1
        length_values.append(float(record["metrics"]["length_m"]))
        ref_dist = record["metrics"].get("mean_reference_distance_m")
        if ref_dist is not None and np.isfinite(ref_dist):
            ref_values.append(float(ref_dist))

    summary = {
        "num_samples": count,
        "label_counts": {key: _counter_to_dict(counter) for key, counter in label_counts.items()},
    }
    if length_values:
        arr = np.asarray(length_values, dtype=np.float32)
        summary["length_m"] = {
            "mean": float(arr.mean()),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }
    if ref_values:
        arr = np.asarray(ref_values, dtype=np.float32)
        summary["mean_reference_distance_m"] = {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
        }
    return summary


def export_instruction_dataset(args: argparse.Namespace) -> Dict:
    dataset = args.dataset.lower()
    dataset_dir = _dataset_dir(dataset, args.data_root)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_path = dataset_dir / f"{dataset}_trajs.npy"
    road_path = dataset_dir / f"{dataset}_roads.npy"
    stats_path = dataset_dir / f"{dataset}_global_stats.json"
    trajs = np.load(traj_path, mmap_mode="r")
    roads = np.load(road_path, mmap_mode="r") if road_path.exists() else None
    stats = _read_stats(stats_path)

    total = len(trajs)
    indices = _select_indices(total, args.max_samples, args.seed, args.stride)
    jsonl_path = output_dir / f"{dataset}_instruction_samples.jsonl"
    summary_path = output_dir / f"{dataset}_instruction_summary.json"

    records_for_summary = []
    with jsonl_path.open("w", encoding="utf-8") as f:
        for index in indices:
            road = roads[int(index)] if roads is not None else None
            record = mine_instruction_record(
                sample_id=int(index),
                dataset=dataset,
                traj=trajs[int(index)],
                road=road,
                stats=stats,
                include_coordinates=not args.no_coordinates,
            ).to_dict()
            records_for_summary.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = _summarize(records_for_summary)
    summary.update(
        {
            "dataset": dataset,
            "source_dir": str(dataset_dir),
            "output_jsonl": str(jsonl_path),
            "seed": int(args.seed),
            "max_samples": int(args.max_samples),
            "stride": int(args.stride),
            "coordinates_included": not bool(args.no_coordinates),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export rule-mined instruction-trajectory pairs.")
    parser.add_argument("--dataset", default="porto", choices=["porto", "beijing", "sanfrancisco"])
    parser.add_argument("--data-root", type=Path, default=None, help="Optional root containing {dataset}/{dataset}_*.npy")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "instruction_data")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-coordinates", action="store_true", help="Only export labels and metrics.")
    return parser.parse_args()


def main() -> None:
    summary = export_instruction_dataset(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
