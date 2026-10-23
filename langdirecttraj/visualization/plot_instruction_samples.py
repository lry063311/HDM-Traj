from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def _load_records(path: Path, limit: int) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def _plot_record(ax, record: dict) -> None:
    traj = np.asarray(record.get("trajectory", []), dtype=np.float32)
    ref = np.asarray(record.get("reference", []), dtype=np.float32)
    if len(ref) > 1:
        ax.plot(ref[:, 0], ref[:, 1], color="#9aa0a6", linewidth=1.2, label="reference")
    if len(traj) > 1:
        ax.plot(traj[:, 0], traj[:, 1], color="#1f77b4", linewidth=1.6, label="trajectory")
        ax.scatter(traj[0, 0], traj[0, 1], s=16, color="#2ca02c", zorder=3)
        ax.scatter(traj[-1, 0], traj[-1, 1], s=16, color="#d62728", zorder=3)
    labels = record.get("labels", {})
    title = (
        f"{record.get('sample_id')} | {labels.get('distance')} | "
        f"{labels.get('direction')} | {labels.get('curvature')}"
    )
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=6)


def plot_instruction_samples(args: argparse.Namespace) -> Path:
    records = _load_records(args.input, args.num)
    if not records:
        raise ValueError(f"No records found in {args.input}")
    cols = min(args.cols, len(records))
    rows = int(math.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, record in zip(axes.ravel(), records):
        ax.axis("on")
        _plot_record(ax, record)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    return args.output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot instruction-trajectory JSONL samples.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--cols", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    output = plot_instruction_samples(parse_args())
    print(output)


if __name__ == "__main__":
    main()
