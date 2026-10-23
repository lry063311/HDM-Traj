from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from langdirecttraj.train.control_losses import MODE_ANCHOR_FEATURES


REPO_ROOT = Path(__file__).resolve().parents[2]


ROLE_Z_OFFSETS = np.asarray(
    [
        [-0.8, 0.8, -0.7, -0.8, -0.8],  # compact, high-progress, tight/straight
        [0.0, 0.0, 0.0, 0.0, 0.0],  # central medium mode
        [0.7, 0.7, -0.2, -0.2, -0.3],  # longer and high-progress
        [0.4, -0.8, 0.9, 0.4, 0.8],  # low-progress/high-curvature turn mode
        [-0.4, 0.1, 0.2, -0.7, 0.4],  # reference-following turning mode
        [0.9, -0.2, 0.5, 0.9, 0.6],  # loose-reference/exploratory mode
    ],
    dtype=np.float64,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _load_mode_means(metrics: Dict) -> np.ndarray:
    per_mode = metrics["per_mode"]
    rows = []
    for mode_idx in range(len(per_mode)):
        row = per_mode[str(mode_idx)]
        rows.append([float(row[name]) for name in MODE_ANCHOR_FEATURES])
    return np.asarray(rows, dtype=np.float64)


def _robust_scales(mode_means: np.ndarray) -> np.ndarray:
    spread = np.percentile(mode_means, 75, axis=0) - np.percentile(mode_means, 25, axis=0)
    fallback = np.asarray([0.20, 0.16, 0.030, 0.0025, 0.35], dtype=np.float64)
    return np.maximum(spread, fallback)


def _build_decoder_role_tables(anchors: np.ndarray, center: np.ndarray, spread: np.ndarray) -> Dict:
    z = (anchors - center[None, :]) / np.maximum(spread[None, :], 1e-6)
    length_z = z[:, 0]
    progress_z = z[:, 1]
    curv_z = z[:, 2]
    ref_z = z[:, 3]
    heading_z = z[:, 4]

    close_ref = _sigmoid(-1.2 * ref_z - 0.4 * curv_z)
    smooth = _sigmoid(-0.8 * curv_z - 0.5 * heading_z)
    loose_ref = _sigmoid(1.2 * ref_z)
    winding = _sigmoid(0.7 * curv_z + 0.7 * heading_z)
    fast_progress = _sigmoid(0.9 * progress_z + 0.35 * length_z)
    slow_progress = _sigmoid(-0.9 * progress_z)
    role_table = np.stack(
        [close_ref, smooth, loose_ref, winding, fast_progress, slow_progress],
        axis=1,
    )
    role_table = role_table / np.maximum(role_table.max(axis=1, keepdims=True), 1e-6)
    role_table = np.clip(role_table, 0.0, 1.0)

    endpoint_budget = 0.32 + 0.08 * np.maximum(length_z, 0.0) + 0.06 * np.maximum(np.abs(progress_z) - 0.3, 0.0)
    endpoint_budget -= 0.05 * np.maximum(curv_z, 0.0)
    mid_budget = 0.46 + 0.10 * np.maximum(curv_z, 0.0) + 0.08 * np.maximum(heading_z, 0.0)
    mid_budget += 0.05 * np.maximum(ref_z, 0.0)
    score_budget = 0.82 + 0.18 * role_table.max(axis=1)
    budget_table = np.stack(
        [
            np.clip(endpoint_budget, 0.18, 0.62),
            np.clip(mid_budget, 0.28, 0.74),
            np.clip(score_budget, 0.70, 1.05),
        ],
        axis=1,
    )

    tangent_bias = 0.18 * length_z + 0.22 * progress_z
    normal_bias = 0.20 * ref_z + 0.25 * curv_z + 0.18 * heading_z
    gate_bias = 0.14 * np.maximum(role_table.max(axis=1) - role_table.mean(axis=1), 0.0)
    control_table = np.stack(
        [
            np.clip(tangent_bias, -0.45, 0.45),
            np.clip(normal_bias, -0.55, 0.65),
            np.clip(gate_bias, -0.20, 0.25),
        ],
        axis=1,
    )

    return {
        "mode_role_table": role_table.astype(float).tolist(),
        "mode_role_budget_table": budget_table.astype(float).tolist(),
        "mode_role_control_table": control_table.astype(float).tolist(),
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
    }


def build_role_anchors(metrics: Dict, strength: float = 0.75) -> Dict:
    mode_means = _load_mode_means(metrics)
    num_modes = mode_means.shape[0]
    center = np.median(mode_means, axis=0)
    spread = _robust_scales(mode_means)
    offsets = ROLE_Z_OFFSETS
    if num_modes > offsets.shape[0]:
        extra = np.linspace(-0.5, 0.5, num_modes - offsets.shape[0], dtype=np.float64)[:, None]
        extra_offsets = np.concatenate([extra, -extra, extra, extra, -extra], axis=1)
        offsets = np.concatenate([offsets, extra_offsets], axis=0)
    offsets = offsets[:num_modes]
    anchors = center[None, :] + float(strength) * spread[None, :] * offsets
    anchors[:, 0] = np.clip(anchors[:, 0], 0.60, 1.80)
    anchors[:, 1] = np.clip(anchors[:, 1], 0.20, 1.10)
    anchors[:, 2] = np.clip(anchors[:, 2], 0.05, 1.20)
    anchors[:, 3] = np.clip(anchors[:, 3], 0.0005, 0.080)
    anchors[:, 4] = np.clip(anchors[:, 4], 0.05, np.pi)
    loss_scales = np.maximum(spread * 1.5, np.asarray([0.18, 0.14, 0.035, 0.0025, 0.30], dtype=np.float64))
    payload = {
        "features": list(MODE_ANCHOR_FEATURES),
        "anchors": anchors.astype(float).tolist(),
        "scales": loss_scales.astype(float).tolist(),
        "source_metrics": str(metrics.get("checkpoint", "unknown")),
        "role_strength": float(strength),
        "empirical_center": center.astype(float).tolist(),
        "empirical_spread": spread.astype(float).tolist(),
        "note": (
            "Distribution-calibrated role anchors: empirical feature magnitudes come from "
            "eval_mode_specialization, while mode offsets impose stable MTR-style geometry roles."
        ),
    }
    payload.update(_build_decoder_role_tables(anchors, center, spread))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build distribution-calibrated mode role anchors from specialization metrics.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "mode_role_anchors" / "anchors.json")
    parser.add_argument("--strength", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.input.read_text(encoding="utf-8"))
    payload = build_role_anchors(metrics, strength=args.strength)
    payload["input"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
