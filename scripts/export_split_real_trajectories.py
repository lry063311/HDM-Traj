#!/usr/bin/env python
"""Export split real trajectories from compact Porto instruction metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    meta_path = dataset_root / f"porto_instruction_meta.{args.split}.jsonl"
    base_path = dataset_root / "porto_base_lonlat.npy"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    max_samples = int(args.max_samples)
    rows: list[int] = []
    sample_ids: list[int] = []
    gt_lengths: list[float] = []
    with meta_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if max_samples > 0 and len(rows) >= max_samples:
                break
            record = json.loads(line)
            compact_refs = record.get("compact_refs", {})
            trajectory_ref = compact_refs.get("trajectory", {})
            row = trajectory_ref.get("row", record.get("base_index"))
            if row is None:
                raise ValueError(f"Missing compact trajectory row at line {line_idx}")
            rows.append(int(row))
            sample_ids.append(int(record.get("sample_id", line_idx)))
            gt_lengths.append(float(record.get("metrics", {}).get("length_m", np.nan)))

    base = np.load(base_path, mmap_mode="r")
    row_array = np.asarray(rows, dtype=np.int64)
    trajectories = np.asarray(base[row_array], dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        real_trajectories=trajectories,
        sample_id=np.asarray(sample_ids, dtype=np.int64),
        base_index=row_array,
        gt_length_m=np.asarray(gt_lengths, dtype=np.float32),
    )
    summary = {
        "split": args.split,
        "num_samples": int(trajectories.shape[0]),
        "trajectory_shape": list(trajectories.shape),
        "output": str(args.output),
        "gt_avg_length_km": float(np.nanmean(gt_lengths) / 1000.0),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
