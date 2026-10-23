"""Build the paper Figure 2 reference-adherence diagnostic.

The figure is data-driven from exported Table 4 visual cases. It shows a fixed-OD
loose-to-tight reference-adherence edit and the associated continuous metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "outputs" / "figure2_reference_adherence_visual_cases"
OUT_DIR = ROOT / "latex" / "figures"
OUT_BASE = OUT_DIR / "figure2_reference_adherence_diagnostic"
CASE_INDICES = [0, 9]


COLORS = {
    "base": "#2D5F9A",
    "edited": "#E68128",
    "reference": "#303030",
    "target": "#B8B8B8",
    "neutral": "#666666",
    "light": "#E8E8E8",
    "green": "#2F8F46",
}


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["legend.frameon"] = False


def load_cases() -> list[tuple[dict, dict[str, np.ndarray]]]:
    manifest_path = CASE_DIR / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    cases = []
    for index in CASE_INDICES:
        record = manifest["records"][index]
        arrays = dict(np.load(CASE_DIR / record["npz"]))
        cases.append((record, arrays))
    return cases


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.15, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def path_extent(paths: list[np.ndarray], pad_frac: float = 0.07) -> tuple[float, float, float, float]:
    pts = np.concatenate(paths, axis=0)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    return (
        xmin - dx * pad_frac,
        xmax + dx * pad_frac,
        ymin - dy * pad_frac,
        ymax + dy * pad_frac,
    )


def plot_trajectory_panel(ax: plt.Axes, arrays: dict[str, np.ndarray], record: dict, label: str, title: str) -> None:
    reference = arrays["reference"]
    target = arrays["target"]
    base = arrays["base_selected"]
    edited = arrays["edited_selected"]

    ax.plot(target[:, 0], target[:, 1], color=COLORS["target"], lw=0.95, ls=(0, (1.0, 2.1)), zorder=1)
    ax.plot(reference[:, 0], reference[:, 1], color=COLORS["reference"], lw=1.35, ls="--", zorder=2)
    ax.plot(base[:, 0], base[:, 1], color=COLORS["base"], lw=1.8, zorder=4)
    ax.plot(edited[:, 0], edited[:, 1], color=COLORS["edited"], lw=1.8, zorder=5)

    origin = reference[0]
    dest = reference[-1]
    ax.scatter([origin[0]], [origin[1]], s=30, color="#D33F3F", edgecolor="white", linewidth=0.6, zorder=8)
    ax.scatter([dest[0]], [dest[1]], s=30, color="#2E9E44", edgecolor="white", linewidth=0.6, zorder=8)
    ax.text(origin[0], origin[1], " O", fontsize=6.4, ha="left", va="center")
    ax.text(dest[0], dest[1], " D", fontsize=6.4, ha="left", va="center")

    xmin, xmax, ymin, ymax = path_extent([reference, target, base, edited])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_box_aspect(1.0)
    ax.set_xlabel("Longitude", labelpad=0.8, fontsize=6.1)
    ax.set_ylabel("Latitude", labelpad=0.8, fontsize=6.1)
    ax.set_title(title, loc="left", pad=3, fontsize=7.2, fontweight="bold")
    ax.grid(color="#EFEFEF", lw=0.45)
    ax.tick_params(labelsize=5.5, pad=1.0)
    add_panel_label(ax, label, x=-0.18, y=1.04)

    chord_point = reference[int(len(reference) * 0.58)]
    ax.text(chord_point[0] + 0.0016, chord_point[1], "OD chord", color=COLORS["reference"], fontsize=5.7, ha="left", va="center")

    legend_x, legend_y = 0.04, 0.13
    for i, (name, color) in enumerate([("loose", COLORS["base"]), ("tight", COLORS["edited"])]):
        y = legend_y - i * 0.065
        ax.plot([legend_x, legend_x + 0.075], [y, y], transform=ax.transAxes, color=color, lw=1.8, clip_on=False)
        ax.text(legend_x + 0.085, y, name, transform=ax.transAxes, fontsize=5.7, color=color, va="center", ha="left")


def write_source_data(records: list[dict]) -> None:
    source_path = OUT_DIR / "figure2_reference_adherence_source_data.csv"
    fields = [
        "case",
        "transition",
        "success",
        "base_mean_reference_distance_norm",
        "edited_mean_reference_distance_norm",
        "reference_distance_delta",
        "reference_distance_improvement_percent",
        "target_effect_z",
        "independent_collateral_z",
        "base_instruction",
        "edited_instruction",
    ]
    with source_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            base_ref = float(record["base_mean_reference_distance_norm"])
            cf_ref = float(record["cf_mean_reference_distance_norm"])
            row = {
                "case": record["npz"],
                "transition": record["transition"],
                "success": record["success"],
                "base_mean_reference_distance_norm": f"{base_ref:.9f}",
                "edited_mean_reference_distance_norm": f"{cf_ref:.9f}",
                "reference_distance_delta": f"{base_ref - cf_ref:.9f}",
                "reference_distance_improvement_percent": f"{(base_ref - cf_ref) / base_ref * 100.0:.6f}",
                "target_effect_z": f"{float(record['target_effect_z']):.9f}",
                "independent_collateral_z": f"{float(record['independent_collateral_z']):.9f}",
                "base_instruction": record["base_instruction"],
                "edited_instruction": record["cf_instruction"],
            }
            writer.writerow(row)


def write_qa_note(records: list[dict]) -> None:
    base_values = np.array([float(record["base_mean_reference_distance_norm"]) for record in records])
    cf_values = np.array([float(record["cf_mean_reference_distance_norm"]) for record in records])
    collateral_values = np.array([float(record["independent_collateral_z"]) for record in records])
    qa_path = OUT_DIR / "figure2_reference_adherence_qa.md"
    qa_path.write_text(
        "\n".join(
            [
                "# Figure 2 QA Note",
                "",
                "Core conclusion: loose-to-tight reference-adherence edits under fixed OD conditions produce measurable continuous movement toward the OD-chord reference with low independent collateral.",
                "Figure archetype: asymmetric mixed-modality figure.",
                "Backend: Python / matplotlib only.",
                "Final size: compact single-column figure.",
                "Representative cases: " + ", ".join(f"`{record['npz']}`" for record in records) + ".",
                "Transition: `reference_adherence:loose->tight`.",
                f"Mean reference distance: {base_values.mean():.6f} -> {cf_values.mean():.6f}.",
                f"Mean target effect z: {np.mean([float(record['target_effect_z']) for record in records]):.6f}.",
                f"Mean independent collateral z: {collateral_values.mean():.6f}.",
                "Image-integrity note: trajectories are plotted directly from exported lon/lat arrays; no smoothing, local warping, or manual editing is applied.",
                "Reviewer risk: these are representative successful cases and should be interpreted as qualitative evidence supporting the aggregate Table 4 diagnostics, not as a claim that reference-adherence classification is solved.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    records = [record for record, _ in cases]

    fig = plt.figure(figsize=(3.35, 1.82), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, wspace=0.30)
    ax_traj_1 = fig.add_subplot(gs[0, 0])
    ax_traj_2 = fig.add_subplot(gs[0, 1])
    plot_trajectory_panel(ax_traj_1, cases[0][1], cases[0][0], "a", "Example 1")
    plot_trajectory_panel(ax_traj_2, cases[1][1], cases[1][0], "b", "Example 2")

    for ext, dpi in [("svg", 600), ("pdf", 600), ("png", 600)]:
        fig.savefig(OUT_BASE.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    write_source_data(records)
    write_qa_note(records)
    print(f"Saved {OUT_BASE.with_suffix('.svg')}")
    print(f"Saved {OUT_BASE.with_suffix('.pdf')}")
    print(f"Saved {OUT_BASE.with_suffix('.png')}")


if __name__ == "__main__":
    main()
