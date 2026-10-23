from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from langdirecttraj.eval.distribution_metrics import (  # noqa: E402
    js_divergence,
    load_fixed_bounds,
    to_grid_ids,
)


REAL_PATH = REPO_ROOT / "outputs/road_full/real/porto_test_real_trajectories.npz"
BOUNDS_PATH = WORKSPACE_ROOT / "DiffTraj/data/porto_od_only/porto_global_stats.json"
CONTROL_ID_MANIFEST = (
    REPO_ROOT / "data/proto/porto_od_chord_compact_v1/instruction_data/porto_split_manifest.json"
)
INPUTS = {
    "HDM-Traj": {
        "path": REPO_ROOT
        / "outputs/paper_full/porto_fixed16_k_sweep_seed2026/final_test_k4/test_full"
        / "semantic_reranker_trajectories.npz",
        "key": "generated_trajectories",
    },
    "DiffTraj": {
        "path": WORKSPACE_ROOT
        / "DiffTraj/difftraj_outputs/unsmoothed_sigma0/porto"
        / "porto_difftraj_sigma0_predictions.npz",
        "key": "generated_trajectories",
    },
    "Traj-Transformer": {
        "path": WORKSPACE_ROOT
        / "Traj-Transformer-master/outputs/porto/eval"
        / "porto_traj_transformer_predictions.npz",
        "key": "generated_trajectories",
    },
    "ControlTraj": {
        "path": WORKSPACE_ROOT
        / "ControlTraj/controltraj_outputs/ControlTraj_porto"
        / "controltraj_porto_unet150_langdirect_eval"
        / "controltraj_porto_unet150_predictions.npz",
        "key": "generated",
    },
}
GRIDS = (16, 32, 64)
NUM_SHARDS = 10
OUT = REPO_ROOT / "outputs/paper_full/spatial_scale_robustness/porto_16_32_64"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_npz(path: Path, key: str) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        values = np.asarray(payload[key], dtype=np.float32)
        sample_ids = (
            np.asarray(payload["sample_id"], dtype=np.int64)
            if "sample_id" in payload.files
            else None
        )
    return values, sample_ids


def control_test_ids() -> np.ndarray:
    payload = json.loads(CONTROL_ID_MANIFEST.read_text(encoding="utf-8"))
    return np.asarray(
        sorted(
            int(sample_id)
            for sample_id, split in payload["sample_splits"].items()
            if split == "test"
        ),
        dtype=np.int64,
    )


def grid_ids(
    trajectories: np.ndarray,
    bounds: dict[str, list[float]],
    grid_num: int,
) -> np.ndarray:
    if not np.isfinite(trajectories).all():
        raise ValueError("Spatial-scale audit requires finite trajectories")
    ids = to_grid_ids(
        trajectories.reshape(-1, 2),
        bounds,
        grid_num,
        out_of_bounds_policy="overflow",
    )
    return ids.reshape(trajectories.shape[:2])


def count_density(ids: np.ndarray, num_cells: int) -> np.ndarray:
    return np.bincount(ids.reshape(-1), minlength=num_cells).astype(np.float64)


def count_transitions(ids: np.ndarray, num_cells: int) -> np.ndarray:
    starts = ids[:, :-1]
    ends = ids[:, 1:]
    changed = starts != ends
    pairs = starts[changed] * num_cells + ends[changed]
    return np.bincount(pairs, minlength=num_cells * num_cells).astype(np.float64)


def metric_pair(
    real_ids: np.ndarray,
    generated_ids: np.ndarray,
    num_cells: int,
) -> dict[str, float]:
    metrics = {
        "Density Error": js_divergence(
            count_density(real_ids, num_cells),
            count_density(generated_ids, num_cells),
        ),
        "Route Transition JSD": js_divergence(
            count_transitions(real_ids, num_cells),
            count_transitions(generated_ids, num_cells),
        ),
    }
    if any(value <= 0 for value in metrics.values()):
        raise ValueError(f"Log-scale figure requires strictly positive errors: {metrics}")
    return metrics


def require_matplotlib_panel_alignment(fig, axes, json_out: Path, tolerance_pt: float = 1.5) -> None:
    fig.canvas.draw()
    width_pt, height_pt = fig.get_size_inches() * 72
    boxes = []
    for index, ax in enumerate(axes):
        box = ax.get_position(original=False)
        boxes.append(
            {
                "id": chr(97 + index),
                "left": float(box.x0 * width_pt),
                "bottom": float(box.y0 * height_pt),
                "right": float(box.x1 * width_pt),
                "top": float(box.y1 * height_pt),
            }
        )
    spread = max(
        abs(boxes[0]["left"] - boxes[1]["left"]),
        abs(boxes[0]["right"] - boxes[1]["right"]),
    )
    alignment = {
        "figure_size_pt": [float(width_pt), float(height_pt)],
        "tolerance_pt": tolerance_pt,
        "panels": boxes,
        "maximum_column_alignment_spread_pt": float(spread),
        "verdict": "PASS" if spread <= tolerance_pt else "FIX BEFORE EXPORT",
    }
    json_out.write_text(json.dumps(alignment, indent=2), encoding="utf-8")
    if spread > tolerance_pt:
        raise RuntimeError(f"Panel alignment failed: {spread:.3f} pt")


def plot(rows: list[dict[str, object]]) -> None:
    colors = {
        "HDM-Traj": "#B34873",
        "Traj-Transformer": "#278A73",
        "DiffTraj": "#3C78A8",
        "ControlTraj": "#5E6872",
    }
    markers = {
        "HDM-Traj": "o",
        "Traj-Transformer": "D",
        "DiffTraj": "s",
        "ControlTraj": "^",
    }
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.30), constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.965, bottom=0.11, top=0.80, hspace=0.56)
    metrics = (
        ("Density Error", "Density error"),
        ("Route Transition JSD", "Route-transition JSD"),
    )
    for panel, (ax, (metric, title)) in enumerate(zip(axes, metrics)):
        y_positions = np.arange(len(GRIDS))[::-1]
        for y, grid_num in zip(y_positions, GRIDS):
            selected = [row for row in rows if int(row["grid_num"]) == grid_num]
            values = [float(row[metric]) for row in selected]
            ax.hlines(y, min(values), max(values), color="#D9DEE3", linewidth=1.3, zorder=1)
            winner = min(selected, key=lambda row: float(row[metric]))
            ax.scatter(
                [float(winner[metric])],
                [y],
                s=66,
                facecolors="none",
                edgecolors="#20262D",
                linewidths=0.75,
                zorder=2,
            )
        for model in INPUTS:
            selected = {
                int(row["grid_num"]): float(row[metric])
                for row in rows
                if row["model"] == model
            }
            ax.scatter(
                [selected[grid_num] for grid_num in GRIDS],
                y_positions,
                color=colors[model],
                marker=markers[model],
                s=31,
                linewidths=0.45,
                edgecolors="white",
                label=model,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_yticks(y_positions, [rf"${grid_num}\times{grid_num}$" for grid_num in GRIDS])
        ax.set_xlabel("Metric value (log scale; lower is better)")
        ax.grid(axis="x", color="#E1E5E9", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#AAB1B8")
        ax.spines[["left", "bottom"]].set_linewidth(0.55)
        ax.tick_params(width=0.55, length=2.5)
        ax.text(-0.18, 1.08, chr(97 + panel), transform=ax.transAxes, weight="bold", fontsize=8)
        ax.text(0.0, 1.08, title, transform=ax.transAxes, weight="bold", fontsize=8)
        ax.text(
            0.98,
            1.08,
            "ring = best",
            transform=ax.transAxes,
            ha="right",
            color="#616A73",
            fontsize=6.5,
        )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.54, 0.948),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.10,
        labelspacing=0.55,
        fontsize=6.7,
    )
    require_matplotlib_panel_alignment(fig, axes, OUT / "alignment-audit.json")
    fig.savefig(OUT / "porto_spatial_scale_robustness.pdf", facecolor="white")
    fig.savefig(OUT / "porto_spatial_scale_robustness.svg", facecolor="white")
    fig.savefig(OUT / "porto_spatial_scale_robustness.png", dpi=600, facecolor="white")
    fig.savefig(OUT / "porto_spatial_scale_robustness.tiff", dpi=600, facecolor="white")
    fig.savefig(REPO_ROOT / "latex/figures/porto_spatial_scale_sensitivity.pdf", facecolor="white")
    plt.close(fig)
    (OUT / "caption.txt").write_text(
        "Porto spatial-scale sensitivity using unchanged test predictions and fixed "
        "preprocessing-level global bounds. Panels report density error and route-transition "
        "JSD at three grid resolutions on logarithmic horizontal axes. Horizontal guides span "
        "the four methods, and rings mark the lowest error at each resolution.\n",
        encoding="utf-8",
    )
    (OUT / "figure_contract.md").write_text(
        "# Spatial-scale sensitivity figure contract\n\n"
        "Claim: the ranking among HDM-Traj, Traj-Transformer, and DiffTraj depends on the "
        "metric and spatial resolution, whereas HDM-Traj remains ahead of ControlTraj for "
        "both displayed metrics.\n\n"
        "Evidence: full Porto test predictions are held fixed while only the shared grid "
        "resolution changes among 16 x 16, 32 x 32, and 64 x 64. City bounds and the overflow "
        "policy are fixed. Rings identify the lowest metric-wise minimum and are not uncertainty intervals.\n\n"
        "Boundary: the figure does not assess endpoint error or length error, does not imply "
        "unseen-city transfer, and does not establish uniform multi-scale superiority. Full "
        "estimates are in full_results.csv; ten_shard_results.csv records the disjoint-shard audit.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        with (OUT / "full_results.csv").open(encoding="utf-8", newline="") as handle:
            plot(list(csv.DictReader(handle)))
        return

    real, real_ids = load_npz(REAL_PATH, "real_trajectories")
    if real_ids is None:
        raise ValueError("Canonical real file has no sample_id")

    generated: dict[str, np.ndarray] = {}
    audit_inputs: dict[str, object] = {
        "real_path": str(REAL_PATH),
        "real_sha256": sha256(REAL_PATH),
        "bounds_path": str(BOUNDS_PATH),
        "bounds_sha256": sha256(BOUNDS_PATH),
    }
    for model, spec in INPUTS.items():
        values, sample_ids = load_npz(spec["path"], spec["key"])
        if model == "ControlTraj":
            sample_ids = control_test_ids()
        if values.shape != real.shape:
            raise ValueError(f"{model}: shape mismatch {values.shape} != {real.shape}")
        if not np.array_equal(sample_ids, real_ids):
            raise ValueError(f"{model}: sample IDs are not aligned with canonical Porto test order")
        generated[model] = values
        audit_inputs[model] = {
            "prediction_path": str(spec["path"]),
            "prediction_sha256": sha256(spec["path"]),
            "generated_key": spec["key"],
            "sample_id_aligned": True,
        }

    bounds = load_fixed_bounds(BOUNDS_PATH)
    full_rows: list[dict[str, object]] = []
    shard_rows: list[dict[str, object]] = []
    win_counts: dict[str, dict[str, dict[str, int]]] = {}
    sample_indices = np.arange(len(real))

    for grid_num in GRIDS:
        print(f"Grid {grid_num} x {grid_num}", flush=True)
        num_cells = grid_num * grid_num + 1
        real_grid = grid_ids(real, bounds, grid_num)
        generated_grids = {
            model: grid_ids(values, bounds, grid_num) for model, values in generated.items()
        }
        for model, model_grid in generated_grids.items():
            metrics = metric_pair(real_grid, model_grid, num_cells)
            row = {"grid_num": grid_num, "model": model, "num_compared": len(real), **metrics}
            full_rows.append(row)
            print(
                f"  {model}: Density={metrics['Density Error']:.9f}, "
                f"Route={metrics['Route Transition JSD']:.9f}",
                flush=True,
            )
            for shard in range(NUM_SHARDS):
                index = sample_indices[sample_indices % NUM_SHARDS == shard]
                shard_metrics = metric_pair(real_grid[index], model_grid[index], num_cells)
                shard_rows.append(
                    {
                        "grid_num": grid_num,
                        "shard": shard,
                        "model": model,
                        "num_compared": len(index),
                        **shard_metrics,
                    }
                )
        win_counts[str(grid_num)] = {}
        for metric in ("Density Error", "Route Transition JSD"):
            wins = {model: 0 for model in INPUTS}
            for shard in range(NUM_SHARDS):
                candidates = [
                    row
                    for row in shard_rows
                    if row["grid_num"] == grid_num and row["shard"] == shard
                ]
                winner = min(candidates, key=lambda row: float(row[metric]))["model"]
                wins[str(winner)] += 1
            win_counts[str(grid_num)][metric] = wins

    fields = ["grid_num", "model", "num_compared", "Density Error", "Route Transition JSD"]
    with (OUT / "full_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(full_rows)
    shard_fields = ["grid_num", "shard", "model", "num_compared", "Density Error", "Route Transition JSD"]
    with (OUT / "ten_shard_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=shard_fields)
        writer.writeheader()
        writer.writerows(shard_rows)

    result = {
        "protocol": {
            "city": "porto",
            "grid_resolutions": list(GRIDS),
            "fixed_bounds": bounds,
            "bounds_source": str(BOUNDS_PATH),
            "out_of_bounds_policy": "overflow",
            "transition_protocol": "successive_changed_cell_directed_js",
            "num_disjoint_shards": NUM_SHARDS,
            "sample_shard_rule": "canonical_test_row_index_mod_10",
        },
        "inputs": audit_inputs,
        "full_results": full_rows,
        "ten_shard_win_counts": win_counts,
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot(full_rows)
    print(f"Saved audit: {OUT / 'audit.json'}")


if __name__ == "__main__":
    main()
