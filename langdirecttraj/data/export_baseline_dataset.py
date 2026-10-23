from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.lib.format import open_memmap

from langdirecttraj.data.compact_store import (
    expand_compact_record,
    is_compact_meta_path,
    load_compact_manifest,
    open_compact_arrays,
)
from langdirecttraj.data.instruction_dataset import validate_official_record_protocol
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path, max_samples: int = 0) -> List[Dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_samples > 0 and len(records) >= max_samples:
                break
    if not records:
        raise ValueError(f"No records loaded from {path}")
    if is_compact_meta_path(path):
        arrays = _open_compact_arrays(path)
        records = [expand_compact_record(record, arrays) for record in records]
    return records


def _open_compact_arrays(meta_path: Path) -> Dict[str, np.ndarray]:
    manifest = load_compact_manifest(meta_path)
    manifest_base_dir = Path(str(manifest.get("base_dir", ".")).replace("\\", "/"))
    candidate_base_dirs = []
    if manifest_base_dir.is_absolute():
        candidate_base_dirs.append(manifest_base_dir)
    else:
        candidate_base_dirs.append((REPO_ROOT / manifest_base_dir).resolve())
        candidate_base_dirs.append(Path(meta_path).parent.resolve())

    for base_dir in candidate_base_dirs:
        try:
            resolved_manifest = dict(manifest)
            resolved_manifest["base_dir"] = str(base_dir)
            return open_compact_arrays(resolved_manifest)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Could not resolve compact arrays for {meta_path}")


def resample_polyline(points: np.ndarray, target_len: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f"Expected [T,2] polyline, got {points.shape}")
    if len(points) == target_len:
        return points
    if len(points) <= 1:
        return np.repeat(points[:1], target_len, axis=0)
    step = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(np.clip(step, 1e-8, None))], axis=0)
    if cumulative[-1] <= 1e-8:
        return np.repeat(points[:1], target_len, axis=0)
    targets = np.linspace(0.0, cumulative[-1], target_len, endpoint=True)
    lon = np.interp(targets, cumulative, points[:, 0])
    lat = np.interp(targets, cumulative, points[:, 1])
    return np.stack([lon, lat], axis=1).astype(np.float32)


def infer_bounds(records: Iterable[Dict], padding_ratio: float = 0.02) -> Dict[str, List[float]]:
    pts = []
    for record in records:
        pts.append(np.asarray(record["trajectory"], dtype=np.float32))
    all_pts = np.concatenate(pts, axis=0)
    lon_min, lat_min = all_pts.min(axis=0)
    lon_max, lat_max = all_pts.max(axis=0)
    lon_pad = max((lon_max - lon_min) * padding_ratio, 1e-6)
    lat_pad = max((lat_max - lat_min) * padding_ratio, 1e-6)
    return {
        "lon_bound": [float(lon_min - lon_pad), float(lon_max + lon_pad)],
        "lat_bound": [float(lat_min - lat_pad), float(lat_max + lat_pad)],
    }


def point_to_grid_id(point: np.ndarray, bounds: Dict[str, List[float]], grid_size: int) -> int:
    lon_min, lon_max = bounds["lon_bound"]
    lat_min, lat_max = bounds["lat_bound"]
    col = int((float(point[0]) - lon_min) / max(lon_max - lon_min, 1e-9) * grid_size)
    row = int((float(point[1]) - lat_min) / max(lat_max - lat_min, 1e-9) * grid_size)
    col = int(np.clip(col, 0, grid_size - 1))
    row = int(np.clip(row, 0, grid_size - 1))
    return row * grid_size + col


def tokenize_trajectory(points: np.ndarray, bounds: Dict[str, List[float]], grid_size: int) -> List[int]:
    raw = [point_to_grid_id(point, bounds, grid_size) for point in np.asarray(points, dtype=np.float32)]
    tokens = []
    for token in raw:
        if not tokens or tokens[-1] != token:
            tokens.append(int(token))
    return tokens


def summarize_tokens(records: List[Dict]) -> Dict:
    counts = {
        "num_records": len(records),
        "length": {},
        "unique_tokens": 0,
        "od_pairs": 0,
        "label_counts": defaultdict(Counter),
    }
    lengths = [len(record["grid_tokens"]) for record in records]
    all_tokens = []
    od_pairs = set()
    for record in records:
        all_tokens.extend(record["grid_tokens"])
        od_pairs.add((record["origin"], record["destination"]))
        for field, value in record["labels"].items():
            counts["label_counts"][field][value] += 1
    arr = np.asarray(lengths, dtype=np.float32)
    counts["length"] = {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }
    counts["unique_tokens"] = int(len(set(all_tokens)))
    counts["od_pairs"] = int(len(od_pairs))
    counts["label_counts"] = {
        field: {str(k): int(v) for k, v in sorted(counter.items())}
        for field, counter in counts["label_counts"].items()
    }
    return counts


def export_baseline_dataset(args: argparse.Namespace) -> Dict:
    dataset = args.dataset.lower()
    input_path = args.input or (
        REPO_ROOT
        / "outputs"
        / "local_porto_od_chord_256_verify"
        / "instruction_data"
        / f"{dataset}_instruction_samples.jsonl"
    )
    output_dir = args.output_dir / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(input_path, max_samples=args.max_samples)
    bounds = infer_bounds(records)

    continuous_mode = str(args.continuous_mode)
    write_trajs = continuous_mode in {"both", "trajectories"}
    write_refs = continuous_mode in {"both", "references"}
    continuous_trajs = None
    continuous_refs = None
    tmp_dir = output_dir / "_tmp_continuous"
    if write_trajs or write_refs:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    if write_trajs:
        continuous_trajs = open_memmap(
            tmp_dir / f"{dataset}_trajectories.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(records), int(args.target_len), 2),
        )
    if write_refs:
        continuous_refs = open_memmap(
            tmp_dir / f"{dataset}_references.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(records), int(args.target_len), 2),
        )
    token_records = []
    metadata_records = []
    for row_index, record in enumerate(records):
        validate_official_record_protocol(record, row_index=row_index)
        if "reference" not in record:
            raise ValueError(
                "Official OD-chord baseline export requires an explicit reference "
                f"field; row={row_index}, sample_id={record.get('sample_id', row_index)!r}."
            )
        traj = np.asarray(record["trajectory"], dtype=np.float32)
        ref = np.asarray(record["reference"], dtype=np.float32)
        traj_resampled = resample_polyline(traj, args.target_len)
        ref_resampled = resample_polyline(ref, args.target_len)
        if continuous_trajs is not None:
            continuous_trajs[row_index] = traj_resampled
        if continuous_refs is not None:
            continuous_refs[row_index] = ref_resampled

        grid_tokens = tokenize_trajectory(traj, bounds, args.grid_size)
        ref_grid_tokens = tokenize_trajectory(ref, bounds, args.grid_size)
        labels = dict(record.get("labels", {}))
        token_record = {
            "sample_id": int(record["sample_id"]),
            "dataset": dataset,
            "instruction": str(record.get("instruction", "")),
            "labels": labels,
            "mobility_tokens": DEFAULT_MOBILITY_VOCAB.encode(labels),
            "grid_tokens": grid_tokens,
            "reference_grid_tokens": ref_grid_tokens,
            "origin": int(grid_tokens[0]) if grid_tokens else -1,
            "destination": int(grid_tokens[-1]) if grid_tokens else -1,
            "length": int(len(grid_tokens)),
        }
        token_records.append(token_record)
        metadata_records.append(
            {
                "sample_id": int(record["sample_id"]),
                "instruction": str(record.get("instruction", "")),
                "labels": labels,
                "metrics": dict(record.get("metrics", {})),
            }
        )

    continuous_path = output_dir / f"{dataset}_continuous_eval.npz"
    token_path = output_dir / f"{dataset}_trajgpt_tokens.jsonl"
    metadata_path = output_dir / f"{dataset}_metadata.jsonl"
    summary_path = output_dir / f"{dataset}_baseline_summary.json"

    continuous_payload = {
        "sample_ids": np.asarray([record["sample_id"] for record in metadata_records], dtype=np.int64),
    }
    if continuous_trajs is not None:
        continuous_trajs.flush()
        continuous_payload["trajectories"] = np.load(tmp_dir / f"{dataset}_trajectories.npy", mmap_mode="r")
    if continuous_refs is not None:
        continuous_refs.flush()
        continuous_payload["references"] = np.load(tmp_dir / f"{dataset}_references.npy", mmap_mode="r")
    save_fn = np.savez if args.no_compress else np.savez_compressed
    save_fn(continuous_path, **continuous_payload)
    for payload in continuous_payload.values():
        if isinstance(payload, np.memmap):
            payload._mmap.close()
    with token_path.open("w", encoding="utf-8") as f:
        for record in token_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with metadata_path.open("w", encoding="utf-8") as f:
        for record in metadata_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "dataset": dataset,
        "input": str(input_path),
        "num_samples": len(records),
        "target_len": int(args.target_len),
        "grid_size": int(args.grid_size),
        "bounds": bounds,
        "continuous_npz": str(continuous_path),
        "trajgpt_token_jsonl": str(token_path),
        "metadata_jsonl": str(metadata_path),
        "token_summary": summarize_tokens(token_records),
        "protocols": {
            "continuous_diffusion_baselines": [
                "Density Error",
                "Endpoint Marginal JSD",
                "Route Transition JSD",
                "Length Error",
            ],
            "trajgpt_r_sequence_baseline": [
                "finished_rate",
                "exact_match_rate",
                "jaccard",
                "cosine",
                "bleu_like",
                "link_jsd",
                "connection_jsd",
                "unigram_entropy",
                "bigram_entropy",
                "ue",
                "be",
            ],
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export baseline-compatible LangDirectTraj data views.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "baseline_data")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--target-len", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument(
        "--continuous-mode",
        choices=["both", "references", "trajectories"],
        default="both",
        help="Which continuous arrays to export. Downstream OD-chord training only needs references.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed to reduce memory and CPU pressure on very large splits.",
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(export_baseline_dataset(parse_args()), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
