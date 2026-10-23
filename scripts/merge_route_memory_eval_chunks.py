#!/usr/bin/env python
"""Merge chunked route-memory eval exports into one downstream source npz."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True, help="Glob of semantic_reranker_trajectories.npz chunk files.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in glob.glob(args.input_glob)]
    if not paths:
        raise FileNotFoundError(f"No chunk files matched: {args.input_glob}")

    rows = []
    total = 0
    traj_shape = None
    for path in paths:
        payload = np.load(path)
        ids = np.asarray(payload["sample_id"], dtype=np.int64)
        gen = payload["generated_trajectories"]
        if traj_shape is None:
            traj_shape = tuple(gen.shape[1:])
        elif tuple(gen.shape[1:]) != traj_shape:
            raise ValueError(f"Chunk shape mismatch at {path}: {gen.shape[1:]} != {traj_shape}")
        rows.append((path, len(ids), int(ids.min()), int(ids.max())))
        total += len(ids)

    rows.sort(key=lambda x: x[2])
    tmp_dir = args.tmp_dir or (args.output.parent / "_tmp_merge_route_chunks")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ids_path = tmp_dir / "sample_id.int64.npy"
    gen_path = tmp_dir / "generated_trajectories.float32.npy"
    ids_out = open_memmap(ids_path, mode="w+", dtype=np.int64, shape=(total,))
    gen_out = open_memmap(gen_path, mode="w+", dtype=np.float32, shape=(total, *traj_shape))

    cursor = 0
    for path, n, _lo, _hi in rows:
        payload = np.load(path)
        ids = np.asarray(payload["sample_id"], dtype=np.int64)
        gen = np.asarray(payload["generated_trajectories"], dtype=np.float32)
        ids_out[cursor : cursor + n] = ids
        gen_out[cursor : cursor + n] = gen
        cursor += n
        ids_out.flush()
        gen_out.flush()

    order = np.argsort(np.asarray(ids_out))
    sorted_ids = np.asarray(ids_out)[order]
    if len(np.unique(sorted_ids)) != len(sorted_ids):
        raise ValueError("Duplicate sample_id values found across chunks.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        sample_id=sorted_ids.astype(np.int64),
        generated_trajectories=np.asarray(gen_out)[order].astype(np.float32),
    )
    print(
        {
            "chunks": len(rows),
            "num_samples": int(total),
            "output": str(args.output),
            "first_sample_id": int(sorted_ids[0]),
            "last_sample_id": int(sorted_ids[-1]),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
