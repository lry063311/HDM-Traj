from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from langdirecttraj.eval.distribution_metrics import (  # noqa: E402
    evaluate_distribution,
    load_fixed_bounds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute Table 2 with preprocessing-fixed city grids."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper_full" / "table2_fixed_grid_16x16",
    )
    parser.add_argument("--grid-num", type=int, default=16)
    parser.add_argument("--length-bins", type=int, default=32)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def canonical_paths(repo_root: Path, city: str) -> tuple[Path, Path]:
    bounds_candidates = [
        repo_root / "DiffTraj" / "data" / f"{city}_od_only" / f"{city}_global_stats.json",
        repo_root.parent / "DiffTraj" / "data" / f"{city}_od_only" / f"{city}_global_stats.json",
    ]
    bounds_path = next((path for path in bounds_candidates if path.is_file()), bounds_candidates[0])
    if city == "porto":
        real_path = repo_root / "outputs" / "road_full" / "real" / "porto_test_real_trajectories.npz"
    else:
        real_path = (
            repo_root
            / "outputs"
            / "road_full"
            / city
            / "real"
            / f"{city}_test_real_trajectories.npz"
        )
    return real_path.resolve(), bounds_path.resolve()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("models", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a non-empty list or a {'models': [...]} object")
    return rows


def inspect_npz(path: Path, key: str) -> tuple[tuple[int, ...], np.dtype, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload.files:
            raise KeyError(f"{path} has no key {key!r}; available keys: {payload.files}")
        values = payload[key]
        sample_ids = (
            np.asarray(payload["sample_id"], dtype=np.int64)
            if "sample_id" in payload.files
            else None
        )
        return values.shape, values.dtype, sample_ids


def load_manifest_sample_ids(path: Path, split: str) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample_splits = payload.get("sample_splits")
    if not isinstance(sample_splits, dict):
        raise ValueError(f"Split manifest has no sample_splits map: {path}")
    return np.asarray(
        sorted(int(sample_id) for sample_id, name in sample_splits.items() if name == split),
        dtype=np.int64,
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_").lower()


def precheck(
    rows: list[dict[str, Any]], repo_root: Path
) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        city = str(row["city"]).lower()
        if city not in {"porto", "beijing", "rome"}:
            raise ValueError(f"Row {index}: unsupported city {city!r}")
        model = str(row["model"])
        generated_key = str(row.get("generated_key", "generated_trajectories"))
        prediction_path = resolve_path(repo_root, row["predictions"])
        default_real, default_bounds = canonical_paths(repo_root, city)
        real_path = resolve_path(repo_root, row.get("real", default_real))
        bounds_path = resolve_path(repo_root, row.get("bounds", default_bounds))
        for label, path in (
            ("prediction", prediction_path),
            ("canonical real", real_path),
            ("fixed bounds", bounds_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Row {index} ({model}/{city}): missing {label}: {path}")

        generated_shape, _, generated_ids = inspect_npz(prediction_path, generated_key)
        real_shape, _, real_ids = inspect_npz(real_path, "real_trajectories")
        if generated_shape != real_shape:
            raise ValueError(
                f"Row {index} ({model}/{city}): shape mismatch "
                f"generated={generated_shape}, real={real_shape}"
            )
        sample_id_source = "prediction_npz" if generated_ids is not None else None
        if generated_ids is None and row.get("sample_id_manifest"):
            sample_id_manifest = resolve_path(repo_root, row["sample_id_manifest"])
            if not sample_id_manifest.is_file():
                raise FileNotFoundError(
                    f"Row {index} ({model}/{city}): missing sample-ID manifest: "
                    f"{sample_id_manifest}"
                )
            generated_ids = load_manifest_sample_ids(
                sample_id_manifest,
                str(row.get("sample_id_split", "test")),
            )
            sample_id_source = f"official_split_manifest:{sample_id_manifest}"
        if generated_ids is not None and len(generated_ids) != generated_shape[0]:
            raise ValueError(
                f"Row {index} ({model}/{city}): sample-ID count mismatch "
                f"ids={len(generated_ids)}, generated={generated_shape[0]}"
            )
        if generated_ids is not None and real_ids is not None:
            sample_id_aligned: bool | None = bool(np.array_equal(generated_ids, real_ids))
            if not sample_id_aligned:
                raise ValueError(f"Row {index} ({model}/{city}): sample_id is not aligned")
        else:
            sample_id_aligned = None
        bounds = load_fixed_bounds(bounds_path)
        checked.append(
            {
                "city": city,
                "model": model,
                "prediction_path": prediction_path,
                "generated_key": generated_key,
                "real_path": real_path,
                "bounds_path": bounds_path,
                "bounds": bounds,
                "shape": generated_shape,
                "sample_id_aligned": sample_id_aligned,
                "sample_id_source": sample_id_source,
            }
        )
        alignment = "verified" if sample_id_aligned else "unavailable"
        print(f"CHECKED {model}/{city}: shape={generated_shape}, sample_id={alignment}")
    return checked


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    rows = load_manifest(args.manifest.resolve())
    checked = precheck(rows, repo_root)
    print(f"PRECHECK PASSED: {len(checked)} model/city rows")
    if args.check_only:
        return

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for row in checked:
        with np.load(row["real_path"], allow_pickle=False) as payload:
            real = np.asarray(payload["real_trajectories"], dtype=np.float32)
        with np.load(row["prediction_path"], allow_pickle=False) as payload:
            generated = np.asarray(payload[row["generated_key"]], dtype=np.float32)

        bounds_source = f"preprocessing_global_stats:{row['city']}:{row['bounds_path']}"
        metrics = evaluate_distribution(
            real,
            generated,
            grid_num=args.grid_num,
            length_bins=args.length_bins,
            bounds=row["bounds"],
            bounds_source=bounds_source,
            out_of_bounds_policy="overflow",
        )
        metrics.update(
            {
                "model": row["model"],
                "city": row["city"],
                "real": str(row["real_path"]),
                "generated": str(row["prediction_path"]),
                "generated_key": row["generated_key"],
                "sample_id_aligned": row["sample_id_aligned"],
                "sample_id_source": row["sample_id_source"],
            }
        )
        city_dir = output_dir / row["city"]
        city_dir.mkdir(parents=True, exist_ok=True)
        output_path = city_dir / f"{safe_name(row['model'])}_metrics.json"
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append(
            {
                "city": row["city"],
                "model": row["model"],
                "num_compared": metrics["num_compared"],
                "Density Error": metrics["Density Error"],
                "Route Transition JSD": metrics["Route Transition JSD"],
                "Length Error": metrics["Length Error"],
                "Endpoint Error (m)": metrics["Endpoint Error (m)"],
                "OD Adherence <=1m": metrics["OD Adherence <=1m"],
                "generated_out_of_bounds_point_rate": metrics["generated_bounds_stats"][
                    "out_of_bounds_point_rate"
                ],
                "metrics_path": str(output_path),
            }
        )
        print(
            f"DONE {row['model']}/{row['city']}: "
            f"Density={metrics['Density Error']:.9f}, "
            f"Route={metrics['Route Transition JSD']:.9f}, "
            f"Length={metrics['Length Error']:.9f}, "
            f"OOB-point={metrics['generated_bounds_stats']['out_of_bounds_point_rate']:.6%}"
        )
        del real, generated

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "protocol": f"preprocessing_fixed_{int(args.grid_num)}x{int(args.grid_num)}_plus_overflow_v1",
                "grid_num": args.grid_num,
                "out_of_bounds_policy": "overflow",
                "results": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"SUMMARY: {output_dir / 'summary.tsv'}")


if __name__ == "__main__":
    main()
