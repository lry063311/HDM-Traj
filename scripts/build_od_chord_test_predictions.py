from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical straight OD-chord predictions for Table 2."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "paper_full"
        / "table2_fixed_grid_32x32"
        / "od_chord_inputs",
    )
    return parser.parse_args()


def canonical_real_path(repo_root: Path, city: str) -> Path:
    if city == "porto":
        return repo_root / "outputs" / "road_full" / "real" / "porto_test_real_trajectories.npz"
    return (
        repo_root
        / "outputs"
        / "road_full"
        / city
        / "real"
        / f"{city}_test_real_trajectories.npz"
    )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for city in ("porto", "beijing", "rome"):
        real_path = canonical_real_path(repo_root, city)
        with np.load(real_path, allow_pickle=False) as payload:
            real = np.asarray(payload["real_trajectories"], dtype=np.float32)
            sample_ids = np.asarray(payload["sample_id"], dtype=np.int64)

        alpha = np.linspace(0.0, 1.0, real.shape[1], dtype=np.float32)[None, :, None]
        generated = real[:, :1, :] + alpha * (real[:, -1:, :] - real[:, :1, :])
        output_path = output_dir / f"{city}_od_chord_predictions.npz"
        np.savez_compressed(
            output_path,
            generated_trajectories=generated.astype(np.float32, copy=False),
            sample_id=sample_ids,
        )
        print(f"BUILT {city}: {generated.shape} -> {output_path}")


if __name__ == "__main__":
    main()
