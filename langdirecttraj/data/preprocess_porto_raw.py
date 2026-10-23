from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.lib.format import open_memmap
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None

from langdirecttraj.instructions import mine_instruction_record
from langdirecttraj.instructions.geometry_contract import contract_from_lonlat
from langdirecttraj.instructions.templates import build_instruction
from langdirecttraj.data.split_manifest import (
    SPLIT_ORDER,
    assign_sample_splits,
    parse_split_ratios,
    write_split_manifest,
)
from langdirecttraj.data.compact_store import compact_instruction_split_files


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PATH = (
    REPO_ROOT.parent.parent
    / "DirectTraj"
    / "DirectTraj数据集"
    / "pkdd-15-predict-taxi-service-trajectory-i"
    / "train.csv"
)

PORTO_LON_BOUND = (-8.75, -8.45)
PORTO_LAT_BOUND = (41.05, 41.35)
GPS_SAMPLING_INTERVAL = 15
BOUNDARY_MARGIN = 0.02
TIMEZONE = timezone.utc
REFERENCE_PROTOCOL = "od_chord_only"
PREPROCESSING_PROTOCOL_VERSION = "porto_od_chord_v1"
CONDITION_SOURCE = "origin_destination_chord"
TARGET_SOURCE = "raw_gt_trajectory"
LEAKAGE_GUARD = {
    "reference_uses_gt_middle_geometry": False,
    "reference_uses_only_origin_destination": True,
    "trajectory_is_training_target": True,
}


def split_jsonl_suffix(path: Path, split: str) -> Path:
    return path.with_name(f"{path.stem}.{split}{path.suffix}")


def write_instruction_split_files(
    input_jsonl: Path,
    split_by_id: Dict[int, str],
    dataset: str,
    ratios: Dict[str, float],
    seed: int,
    manifest_path: Path,
) -> dict:
    split_paths = {split: split_jsonl_suffix(input_jsonl, split) for split in SPLIT_ORDER}
    handles = {split: split_paths[split].open("w", encoding="utf-8") for split in SPLIT_ORDER}
    counts = {split: 0 for split in SPLIT_ORDER}
    total = len(split_by_id)
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            total=total,
            desc="split instruction 8/1/1",
            unit="record",
            dynamic_ncols=True,
        )
    else:
        print(
            f"[preprocess] writing instruction split files for {total:,} records...",
            file=sys.stderr,
            flush=True,
        )
    started = time.time()
    try:
        with input_jsonl.open("r", encoding="utf-8") as src:
            for idx, line in enumerate(src, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = int(record["sample_id"])
                split = split_by_id[sample_id]
                record = dict(record)
                record["split"] = split
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                counts[split] += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                elif idx % 50000 == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    rate = idx / elapsed
                    remaining = max(total - idx, 0)
                    eta = remaining / max(rate, 1e-6)
                    print(
                        f"[preprocess] split instruction {idx:,}/{total:,} "
                        f"({idx / max(total, 1):.1%}), {rate:.1f} record/s, eta={eta / 60.0:.1f} min",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        if progress_bar is not None:
            progress_bar.close()
        for handle in handles.values():
            handle.close()
    print(
        "[preprocess] instruction split complete: "
        + ", ".join(f"{split}={counts[split]:,}" for split in SPLIT_ORDER)
        + f", elapsed={(time.time() - started) / 60.0:.1f} min",
        file=sys.stderr,
        flush=True,
    )
    manifest = write_split_manifest(
        manifest_path,
        split_by_id=split_by_id,
        ratios=ratios,
        seed=seed,
        dataset=dataset,
        source=str(input_jsonl),
    )
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "instruction_split_jsonl": {split: str(split_paths[split]) for split in SPLIT_ORDER},
        "instruction_split_counts": counts,
    }


def cleanup_expanded_instruction_jsonl(split_info: dict, jsonl_path: Path) -> dict:
    """Remove temporary expanded instruction JSONL after compact metadata is built."""

    removed: list[str] = []
    candidates = [Path(jsonl_path)]
    candidates.extend(Path(path) for path in split_info.get("instruction_split_jsonl", {}).values())
    for path in candidates:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    split_info["temporary_expanded_instruction_jsonl_removed"] = removed
    split_info["instruction_split_jsonl"] = {}
    split_info["instruction_jsonl"] = None
    return split_info


def count_csv_data_rows(path: Path) -> int:
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total += chunk.count(b"\n")
    return max(total - 1, 0)


def is_valid_point(lon: float, lat: float) -> bool:
    lon_min = PORTO_LON_BOUND[0] - (PORTO_LON_BOUND[1] - PORTO_LON_BOUND[0]) * BOUNDARY_MARGIN
    lon_max = PORTO_LON_BOUND[1] + (PORTO_LON_BOUND[1] - PORTO_LON_BOUND[0]) * BOUNDARY_MARGIN
    lat_min = PORTO_LAT_BOUND[0] - (PORTO_LAT_BOUND[1] - PORTO_LAT_BOUND[0]) * BOUNDARY_MARGIN
    lat_max = PORTO_LAT_BOUND[1] + (PORTO_LAT_BOUND[1] - PORTO_LAT_BOUND[0]) * BOUNDARY_MARGIN
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max and abs(lon) > 1e-4 and abs(lat) > 1e-4


def parse_polyline(polyline_raw: str) -> List[Tuple[float, float]]:
    if not polyline_raw or polyline_raw.strip() == "[]" or len(polyline_raw) < 5:
        return []
    try:
        parsed = json.loads(polyline_raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    points: List[Tuple[float, float]] = []
    for item in parsed:
        if not isinstance(item, list) or len(item) < 2:
            continue
        lon, lat = float(item[0]), float(item[1])
        if is_valid_point(lon, lat):
            points.append((lon, lat))
    return points


def iter_valid_porto_rows(
    raw_path: Path,
    min_traj_len: int,
    max_samples: int,
    scan_limit: int,
    progress: bool = True,
    progress_interval: int = 50000,
    estimate_total_rows: bool = True,
    total_rows_estimate: int | None = None,
    progress_desc: str = "scan raw Porto CSV",
) -> Iterable[Dict]:
    seen_hashes = set()
    estimated_total = total_rows_estimate
    if progress and scan_limit <= 0 and estimate_total_rows:
        print("[preprocess] estimating raw CSV row count for ETA...", file=sys.stderr, flush=True)
        estimate_start = time.time()
        estimated_total = count_csv_data_rows(raw_path)
        print(
            f"[preprocess] estimated raw CSV rows={estimated_total:,} "
            f"({time.time() - estimate_start:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
    with raw_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        scanned = 0
        kept = 0
        progress_total = scan_limit if scan_limit > 0 else estimated_total
        iterator = reader
        pbar = None
        if progress and tqdm is not None:
            pbar = tqdm(
                reader,
                total=progress_total,
                desc=progress_desc,
                unit="row",
                dynamic_ncols=True,
            )
            iterator = pbar
        start_time = time.time()
        last_report = 0
        for row in iterator:
            scanned += 1
            if scan_limit > 0 and scanned > scan_limit:
                break
            if progress and pbar is None and scanned - last_report >= progress_interval:
                elapsed = max(time.time() - start_time, 1e-6)
                rate = scanned / elapsed
                if scan_limit > 0:
                    pct = 100.0 * scanned / max(scan_limit, 1)
                    eta = (scan_limit - scanned) / max(rate, 1e-6)
                    msg = (
                        f"[preprocess] scanned={scanned:,}/{scan_limit:,} "
                        f"({pct:.1f}%) kept={kept:,} rate={rate:,.0f} rows/s eta={eta/60:.1f} min"
                    )
                else:
                    msg = (
                        f"[preprocess] scanned={scanned:,} kept={kept:,} "
                        f"rate={rate:,.0f} rows/s elapsed={elapsed/60:.1f} min"
                    )
                print(msg, file=sys.stderr, flush=True)
                last_report = scanned
            if str(row.get("MISSING_DATA", "")).strip().lower() != "false":
                continue
            try:
                timestamp = int(row.get("TIMESTAMP", "0"))
            except ValueError:
                continue
            if timestamp <= 0:
                continue
            points = parse_polyline(row.get("POLYLINE", ""))
            if len(points) < min_traj_len:
                continue
            key = hashlib.blake2b(
                np.asarray(points, dtype=np.float32).tobytes(),
                digest_size=16,
            ).digest()
            if key in seen_hashes:
                continue
            seen_hashes.add(key)
            yield {
                "trip_id": row.get("TRIP_ID", ""),
                "timestamp": timestamp,
                "trajectory": points,
            }
            kept += 1
            if pbar is not None:
                pbar.set_postfix(scanned=f"{scanned:,}", kept=f"{kept:,}", refresh=False)
            if max_samples > 0 and kept >= max_samples:
                break
        if pbar is not None:
            pbar.close()
        if progress:
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"[preprocess] raw scan complete: scanned={scanned:,}, kept={kept:,}, "
                f"elapsed={elapsed/60:.1f} min",
                file=sys.stderr,
                flush=True,
            )


def resample_polyline(points: np.ndarray, target_len: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((target_len, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points[:1], target_len, axis=0).astype(np.float32)
    if len(points) == target_len:
        return points.astype(np.float32)
    steps = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(np.clip(steps, 1e-8, None))])
    if cumulative[-1] <= 1e-8:
        return np.repeat(points[:1], target_len, axis=0).astype(np.float32)
    targets = np.linspace(0.0, float(cumulative[-1]), target_len, endpoint=True)
    lon = np.interp(targets, cumulative, points[:, 0])
    lat = np.interp(targets, cumulative, points[:, 1])
    return np.stack([lon, lat], axis=1).astype(np.float32)


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def compute_trip_attributes(points: List[Tuple[float, float]], timestamp: int) -> List[float]:
    timestamps = [timestamp + GPS_SAMPLING_INTERVAL * i for i in range(len(points))]
    start_dt = datetime.fromtimestamp(timestamps[0], tz=TIMEZONE)
    start_total_min = start_dt.hour * 60 + start_dt.minute
    start_cat = float(np.clip(start_total_min // 5, 0, 287))
    trip_duration = max((timestamps[-1] - timestamps[0]) / 60.0, 0.1)
    total_distance = 0.0
    for p1, p2 in zip(points[:-1], points[1:]):
        total_distance += haversine_km(p1, p2)
    total_distance = max(total_distance, 0.1)
    avg_speed = float(np.clip(total_distance / (trip_duration / 60.0), 0.0, 120.0))
    time_per_km = float(np.clip(trip_duration / total_distance, 0.0, 60.0))
    return [
        start_cat,
        float(start_total_min),
        float(trip_duration),
        float(total_distance),
        avg_speed,
        time_per_km,
    ]


def trip_attribute_dict(values: List[float]) -> Dict[str, float]:
    return {
        "start_time_bin_5min": float(values[0]),
        "start_total_min": float(values[1]),
        "trip_duration_min": float(values[2]),
        "total_distance_km": float(values[3]),
        "avg_speed_kmh": float(values[4]),
        "time_per_km_min": float(values[5]),
    }


def build_progress_pace_thresholds(heads: np.ndarray) -> Dict[str, float]:
    """Estimate pace-token thresholds from raw timestamp-derived attributes."""
    speeds = np.asarray(heads[:, 4], dtype=np.float32)
    speeds = speeds[np.isfinite(speeds)]
    if speeds.size == 0:
        return {"slow_max_avg_speed_kmh": 12.0, "fast_min_avg_speed_kmh": 24.0}
    slow, fast = np.percentile(speeds, [33.333, 66.667])
    if float(fast - slow) < 1e-3:
        center = float(np.median(speeds))
        slow = center * 0.85
        fast = center * 1.15
    return {
        "slow_max_avg_speed_kmh": float(slow),
        "fast_min_avg_speed_kmh": float(fast),
    }


def progress_label_from_trip_attrs(trip_attrs: Dict[str, float], thresholds: Dict[str, float]) -> str:
    speed = float(trip_attrs.get("avg_speed_kmh", 0.0))
    if speed <= float(thresholds["slow_max_avg_speed_kmh"]):
        return "slow"
    if speed >= float(thresholds["fast_min_avg_speed_kmh"]):
        return "fast"
    return "normal"


def normalize_traj(raw: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    out = np.empty_like(raw, dtype=np.float32)
    out[:, 0] = (raw[:, 0] - stats["lon_mean"]) / (stats["lon_std"] + 1e-8)
    out[:, 1] = (raw[:, 1] - stats["lat_mean"]) / (stats["lat_std"] + 1e-8)
    return np.clip(out, -5.0, 5.0).astype(np.float32)


def build_od_chord(points: np.ndarray, target_len: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((target_len, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points[:1], target_len, axis=0).astype(np.float32)
    start = points[0]
    end = points[-1]
    t = np.linspace(0.0, 1.0, target_len, endpoint=True, dtype=np.float32)[:, None]
    return (start[None, :] * (1.0 - t) + end[None, :] * t).astype(np.float32)


def lonlat_meter_scale(points: np.ndarray) -> Tuple[float, float]:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return 111_320.0, 110_540.0
    lat = float(np.nanmean(points[:, 1]))
    return 111_320.0 * math.cos(math.radians(lat)), 110_540.0


def pointwise_distance_m(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) != len(b):
        b = resample_polyline(b, len(a))
    sx, sy = lonlat_meter_scale(np.concatenate([a, b], axis=0))
    diff = np.empty_like(a, dtype=np.float32)
    diff[:, 0] = (a[:, 0] - b[:, 0]) * sx
    diff[:, 1] = (a[:, 1] - b[:, 1]) * sy
    return np.linalg.norm(diff, axis=1)


def audit_od_chord_reference(raw_resampled: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """Return leakage-audit stats for the official OD-chord reference.

    The reference is allowed to use only the first and last raw points. The
    middle GT trajectory can still be saved as the target, but it must not be
    reconstructed as a route/reference condition.
    """
    expected = build_od_chord(raw_resampled, len(reference))
    chord_error = pointwise_distance_m(reference, expected)
    target_distance = pointwise_distance_m(reference, raw_resampled)
    return {
        "reference_chord_error_max_m": float(chord_error.max(initial=0.0)),
        "reference_chord_error_mean_m": float(chord_error.mean() if len(chord_error) else 0.0),
        "reference_to_target_distance_mean_m": float(target_distance.mean() if len(target_distance) else 0.0),
        "reference_to_target_distance_p90_m": float(np.percentile(target_distance, 90) if len(target_distance) else 0.0),
        "reference_non_endpoint_source_points": 0.0,
    }


def make_road_segments(reference_points: np.ndarray, stats: Dict[str, float], target_len: int) -> np.ndarray:
    if len(reference_points) < 2:
        return np.zeros((target_len, 4), dtype=np.float32)
    segments = []
    for lon1, lat1, lon2, lat2 in np.column_stack(
        [
            reference_points[:-1, 0],
            reference_points[:-1, 1],
            reference_points[1:, 0],
            reference_points[1:, 1],
        ]
    ):
        segments.append(
            [
                (lon1 - stats["lon_mean"]) / (stats["lon_std"] + 1e-8),
                (lat1 - stats["lat_mean"]) / (stats["lat_std"] + 1e-8),
                (lon2 - stats["lon_mean"]) / (stats["lon_std"] + 1e-8),
                (lat2 - stats["lat_mean"]) / (stats["lat_std"] + 1e-8),
            ]
        )
    while len(segments) < target_len:
        segments.append(segments[-1] if segments else [0.0, 0.0, 0.0, 0.0])
    return np.clip(np.asarray(segments[:target_len], dtype=np.float32), -5.0, 5.0)


def summarize_label_counts(records: List[Dict]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for record in records:
        for field, value in record["labels"].items():
            counts.setdefault(field, {})
            counts[field][value] = counts[field].get(value, 0) + 1
    return {field: dict(sorted(values.items())) for field, values in sorted(counts.items())}


def build_progress_pace_thresholds_from_speeds(speeds: np.ndarray) -> Dict[str, float]:
    speeds = np.asarray(speeds, dtype=np.float32)
    speeds = speeds[np.isfinite(speeds)]
    if speeds.size == 0:
        return {"slow_max_avg_speed_kmh": 12.0, "fast_min_avg_speed_kmh": 24.0}
    slow, fast = np.percentile(speeds, [33.333, 66.667])
    if float(fast - slow) < 1e-3:
        center = float(np.median(speeds))
        slow = center * 0.85
        fast = center * 1.15
    return {
        "slow_max_avg_speed_kmh": float(slow),
        "fast_min_avg_speed_kmh": float(fast),
    }


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def update_counter(counter: Dict[str, Dict[str, int]], labels: Dict[str, str]) -> None:
    for field, value in labels.items():
        counter.setdefault(field, {})
        text = str(value)
        counter[field][text] = counter[field].get(text, 0) + 1


def counter_to_sorted_dict(counter: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    return {field: dict(sorted(values.items())) for field, values in sorted(counter.items())}


def stream_full_porto_preprocess(
    args: argparse.Namespace,
    raw_path: Path,
    data_dir: Path,
    instruction_dir: Path,
    total_rows_estimate: int | None,
) -> Dict:
    """Full-data path that avoids keeping every raw trajectory and JSON record in memory."""

    progress = not getattr(args, "no_progress", False)
    progress_interval = int(getattr(args, "progress_interval", 50000))
    estimate_total_rows = not getattr(args, "no_estimate_total_rows", False)

    stats_start = time.time()
    valid_count = 0
    point_count = 0
    lon_sum = 0.0
    lat_sum = 0.0
    lon_sq_sum = 0.0
    lat_sq_sum = 0.0
    speeds: List[float] = []
    lengths: List[float] = []
    for row in iter_valid_porto_rows(
        raw_path=raw_path,
        min_traj_len=args.min_traj_len,
        max_samples=0,
        scan_limit=args.scan_limit,
        progress=progress,
        progress_interval=progress_interval,
        estimate_total_rows=estimate_total_rows,
        total_rows_estimate=total_rows_estimate,
        progress_desc="pass 1/2 stats",
    ):
        pts = np.asarray(row["trajectory"], dtype=np.float32)
        valid_count += 1
        point_count += int(len(pts))
        lon_sum += float(pts[:, 0].sum())
        lat_sum += float(pts[:, 1].sum())
        lon_sq_sum += float(np.square(pts[:, 0], dtype=np.float64).sum())
        lat_sq_sum += float(np.square(pts[:, 1], dtype=np.float64).sum())
        trip_values = compute_trip_attributes(row["trajectory"], int(row["timestamp"]))
        speeds.append(float(trip_values[4]))
        lengths.append(float(trip_values[3] * 1000.0))

    if valid_count <= 0 or point_count <= 0:
        raise ValueError("No valid Porto trajectories found. Check --raw-path, --scan-limit, and --min-traj-len.")

    lon_mean = lon_sum / point_count
    lat_mean = lat_sum / point_count
    lon_var = max(lon_sq_sum / point_count - lon_mean * lon_mean, 1e-16)
    lat_var = max(lat_sq_sum / point_count - lat_mean * lat_mean, 1e-16)
    progress_pace_thresholds = build_progress_pace_thresholds_from_speeds(np.asarray(speeds, dtype=np.float32))
    sampling_summary = {
        "sampling_strategy": "all_streaming",
        "candidate_pool_size": int(valid_count),
        "selected_size": int(valid_count),
        "stats_pass_elapsed_min": float((time.time() - stats_start) / 60.0),
    }
    stats = {
        "dataset": args.dataset,
        "lon_mean": float(lon_mean),
        "lon_std": float(max(math.sqrt(lon_var), 1e-8)),
        "lat_mean": float(lat_mean),
        "lat_std": float(max(math.sqrt(lat_var), 1e-8)),
        "porto_lon_bound": list(PORTO_LON_BOUND),
        "porto_lat_bound": list(PORTO_LAT_BOUND),
        "boundary_margin": float(BOUNDARY_MARGIN),
        "source_raw_path": str(raw_path),
        "target_len": int(args.target_len),
        "preprocessing_protocol_version": PREPROCESSING_PROTOCOL_VERSION,
        "reference_protocol": REFERENCE_PROTOCOL,
        "condition_source": CONDITION_SOURCE,
        "target_source": TARGET_SOURCE,
        "progress_semantics": "trip_pace_from_raw_timestamp_attributes",
        "progress_pace_thresholds": progress_pace_thresholds,
    }

    traj_path = data_dir / f"{args.dataset}_trajs.npy"
    road_path = data_dir / f"{args.dataset}_roads.npy"
    head_path = data_dir / f"{args.dataset}_heads.npy"
    stats_path = data_dir / f"{args.dataset}_global_stats.json"
    jsonl_path = instruction_dir / f"{args.dataset}_instruction_samples.jsonl"
    summary_path = instruction_dir / f"{args.dataset}_instruction_summary.json"

    trajs_arr = open_memmap(
        traj_path,
        mode="w+",
        dtype=np.float32,
        shape=(valid_count, 2, int(args.target_len)),
    )
    roads_arr = open_memmap(
        road_path,
        mode="w+",
        dtype=np.float32,
        shape=(valid_count, int(args.target_len), 4),
    )
    heads_arr = open_memmap(head_path, mode="w+", dtype=np.float32, shape=(valid_count, 6))

    label_counts: Dict[str, Dict[str, int]] = {}
    chord_error_max = 0.0
    chord_errors: List[float] = []
    ref_target_distances_mean: List[float] = []
    ref_target_distances_p90: List[float] = []
    write_start = time.time()
    with jsonl_path.open("w", encoding="utf-8") as f:
        iterator = iter_valid_porto_rows(
            raw_path=raw_path,
            min_traj_len=args.min_traj_len,
            max_samples=0,
            scan_limit=args.scan_limit,
            progress=progress,
            progress_interval=progress_interval,
            estimate_total_rows=False,
            total_rows_estimate=total_rows_estimate,
            progress_desc="pass 2/2 write",
        )
        for idx, row in enumerate(iterator):
            raw_points = np.asarray(row["trajectory"], dtype=np.float32)
            raw_resampled = resample_polyline(raw_points, args.target_len)
            od_chord = build_od_chord(raw_points, args.target_len)
            audit = audit_od_chord_reference(raw_resampled, od_chord)
            if audit["reference_chord_error_max_m"] > 1e-3:
                raise AssertionError(
                    "Official Porto preprocessing requires an exact OD-chord reference; "
                    f"sample_id={idx} has chord error {audit['reference_chord_error_max_m']:.6f}m."
                )
            norm_resampled = normalize_traj(raw_resampled, stats)
            road_segments = make_road_segments(od_chord, stats, args.target_len)
            trip_values = compute_trip_attributes(row["trajectory"], int(row["timestamp"]))
            trip_attrs = trip_attribute_dict(trip_values)

            heads_arr[idx] = np.asarray(trip_values, dtype=np.float32)
            trajs_arr[idx] = norm_resampled.T.astype(np.float32)
            roads_arr[idx] = road_segments.astype(np.float32)

            record = mine_instruction_record(
                sample_id=idx,
                dataset=args.dataset,
                traj=norm_resampled.T,
                road=road_segments,
                stats=stats,
                include_coordinates=True,
            ).to_dict()
            record["labels"]["progress"] = progress_label_from_trip_attrs(trip_attrs, progress_pace_thresholds)
            record["instruction"] = build_instruction(record["labels"])
            record["metrics"].update(
                {
                    **trip_attrs,
                    "progress_semantics": "trip_pace_from_raw_timestamp_attributes",
                    "progress_speed_slow_threshold_kmh": float(progress_pace_thresholds["slow_max_avg_speed_kmh"]),
                    "progress_speed_fast_threshold_kmh": float(progress_pace_thresholds["fast_min_avg_speed_kmh"]),
                }
            )
            record["trip_attributes"] = {
                **trip_attrs,
                "progress_pace_thresholds": dict(progress_pace_thresholds),
            }
            record["progress_semantics"] = "trip_pace_from_raw_timestamp_attributes"
            record["progress_pace_thresholds"] = dict(progress_pace_thresholds)
            record["trajectory"] = raw_resampled.astype(np.float32).tolist()
            record["reference"] = od_chord.astype(np.float32).tolist()
            record["raw_trip_id"] = row["trip_id"]
            record["preprocessing_protocol_version"] = PREPROCESSING_PROTOCOL_VERSION
            record["reference_protocol"] = REFERENCE_PROTOCOL
            record["condition_source"] = CONDITION_SOURCE
            record["target_source"] = TARGET_SOURCE
            record["leakage_guard"] = dict(LEAKAGE_GUARD)
            record["reference_audit"] = audit
            update_counter(label_counts, record["labels"])
            chord_error = float(audit["reference_chord_error_max_m"])
            chord_error_max = max(chord_error_max, chord_error)
            chord_errors.append(chord_error)
            ref_target_distances_mean.append(float(audit["reference_to_target_distance_mean_m"]))
            ref_target_distances_p90.append(float(audit["reference_to_target_distance_p90_m"]))
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("[preprocess] pass 2/2 write complete; flushing numpy memmap arrays...", file=sys.stderr, flush=True)
    flush_start = time.time()
    trajs_arr.flush()
    roads_arr.flush()
    heads_arr.flush()
    del trajs_arr, roads_arr, heads_arr
    print(
        f"[preprocess] memmap flush complete: elapsed={(time.time() - flush_start) / 60.0:.1f} min",
        file=sys.stderr,
        flush=True,
    )

    lengths_arr = np.asarray(lengths, dtype=np.float32)
    chord_errors_arr = np.asarray(chord_errors, dtype=np.float32)
    ref_means = np.asarray(ref_target_distances_mean, dtype=np.float32)
    ref_p90 = np.asarray(ref_target_distances_p90, dtype=np.float32)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "dataset": args.dataset,
        "raw_path": str(raw_path),
        "num_samples": int(valid_count),
        "target_len": int(args.target_len),
        "min_traj_len": int(args.min_traj_len),
        "sampling": {**sampling_summary, "write_pass_elapsed_min": float((time.time() - write_start) / 60.0)},
        "trajs_shape": [int(valid_count), 2, int(args.target_len)],
        "roads_shape": [int(valid_count), int(args.target_len), 4],
        "heads_shape": [int(valid_count), 6],
        "data_dir": str(data_dir),
        "instruction_jsonl": str(jsonl_path),
        "preprocessing_protocol_version": PREPROCESSING_PROTOCOL_VERSION,
        "reference_protocol": REFERENCE_PROTOCOL,
        "condition_source": CONDITION_SOURCE,
        "target_source": TARGET_SOURCE,
        "progress_semantics": "trip_pace_from_raw_timestamp_attributes",
        "progress_pace_thresholds": progress_pace_thresholds,
        "leakage_guard": dict(LEAKAGE_GUARD),
        "reference_audit": {
            "reference_chord_error_max_m": float(chord_error_max),
            "reference_chord_error_p95_m": float(np.percentile(chord_errors_arr, 95) if chord_errors_arr.size else 0.0),
            "reference_to_target_distance_mean_m": float(ref_means.mean() if ref_means.size else 0.0),
            "reference_to_target_distance_p50_m": float(np.percentile(ref_means, 50) if ref_means.size else 0.0),
            "reference_to_target_distance_p90_m": float(np.percentile(ref_p90, 90) if ref_p90.size else 0.0),
            "reference_non_endpoint_source_points": 0,
        },
        "label_counts": counter_to_sorted_dict(label_counts),
        "length_m": {
            "mean": float(lengths_arr.mean()),
            "p50": float(np.percentile(lengths_arr, 50)),
            "p90": float(np.percentile(lengths_arr, 90)),
        },
        "stats": stats,
        "streaming_full_preprocess": True,
    }
    split_ratios = parse_split_ratios(getattr(args, "split_ratios", None))
    split_seed = int(getattr(args, "split_seed", 20260617))
    print(
        f"[preprocess] building split assignment: records={int(valid_count):,}, "
        f"ratios={split_ratios}, seed={split_seed}",
        file=sys.stderr,
        flush=True,
    )
    split_by_id = assign_sample_splits(range(int(valid_count)), ratios=split_ratios, seed=split_seed)
    print("[preprocess] writing train/val/test instruction split files...", file=sys.stderr, flush=True)
    split_info = write_instruction_split_files(
        input_jsonl=jsonl_path,
        split_by_id=split_by_id,
        dataset=args.dataset,
        ratios=split_ratios,
        seed=split_seed,
        manifest_path=instruction_dir / f"{args.dataset}_split_manifest.json",
    )
    if str(getattr(args, "storage_format", "expanded")) == "compact":
        compact_info = compact_instruction_split_files(
            input_paths={
                split: Path(split_info["instruction_split_jsonl"][split])
                for split in SPLIT_ORDER
            },
            output_dir=instruction_dir,
            dataset=args.dataset,
            target_len=int(args.target_len),
            overwrite=True,
        )
        split_info["compact"] = compact_info
        split_info = cleanup_expanded_instruction_jsonl(split_info, jsonl_path)
        summary["instruction_jsonl"] = None
        summary["storage_format"] = "compact"
    summary["split"] = {
        "split_unit": "base_sample_id",
        "split_ratios": split_info["manifest"]["split_ratios"],
        "split_counts": split_info["manifest"]["split_counts"],
        "split_seed": split_seed,
        "manifest_path": split_info["manifest_path"],
        "instruction_split_jsonl": split_info["instruction_split_jsonl"],
        "instruction_jsonl": split_info.get("instruction_jsonl", str(jsonl_path)),
        "temporary_expanded_instruction_jsonl_removed": split_info.get(
            "temporary_expanded_instruction_jsonl_removed",
            [],
        ),
        "compact": split_info.get("compact"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def label_row_for_sampling(
    row: Dict,
    target_len: int,
    dataset: str,
    progress_pace_thresholds: Dict[str, float] | None = None,
) -> Dict[str, str]:
    raw_points = np.asarray(row["trajectory"], dtype=np.float32)
    raw_resampled = resample_polyline(raw_points, target_len)
    od_chord = build_od_chord(raw_points, target_len)
    labels = dict(contract_from_lonlat(raw_resampled, od_chord, dataset).labels)
    if progress_pace_thresholds is not None:
        trip_attrs = trip_attribute_dict(compute_trip_attributes(row["trajectory"], int(row["timestamp"])))
        labels["progress"] = progress_label_from_trip_attrs(trip_attrs, progress_pace_thresholds)
    return labels


def select_coverage_rows(
    rows: List[Dict],
    max_samples: int,
    target_len: int,
    dataset: str,
    seed: int,
    balance_fields: List[str],
    progress_pace_thresholds: Dict[str, float] | None = None,
) -> Tuple[List[Dict], Dict]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows, {
            "sampling_strategy": "all",
            "candidate_pool_size": int(len(rows)),
            "selected_size": int(len(rows)),
        }

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(rows)).tolist()
    labels_by_idx = {
        idx: label_row_for_sampling(rows[idx], target_len, dataset, progress_pace_thresholds)
        for idx in order
    }

    total_counts: Dict[str, Dict[str, int]] = {}
    for labels in labels_by_idx.values():
        for field in balance_fields:
            value = str(labels.get(field, "unknown"))
            total_counts.setdefault(field, {})
            total_counts[field][value] = total_counts[field].get(value, 0) + 1

    selected: List[int] = []
    selected_set = set()
    selected_counts: Dict[str, Dict[str, int]] = {field: {} for field in balance_fields}

    def add_idx(idx: int) -> None:
        if idx in selected_set or len(selected) >= max_samples:
            return
        selected.append(idx)
        selected_set.add(idx)
        labels = labels_by_idx[idx]
        for field in balance_fields:
            value = str(labels.get(field, "unknown"))
            selected_counts.setdefault(field, {})
            selected_counts[field][value] = selected_counts[field].get(value, 0) + 1

    # First guarantee at least one available sample for every rare label. This
    # keeps local Table 3/4 smoke tests from silently missing a token class.
    for field in balance_fields:
        for value, _count in sorted(total_counts.get(field, {}).items(), key=lambda item: (item[1], item[0])):
            best_idx = None
            best_secondary = -1.0
            for idx in order:
                if idx in selected_set:
                    continue
                labels = labels_by_idx[idx]
                if str(labels.get(field, "unknown")) != value:
                    continue
                secondary = 0.0
                for other_field in balance_fields:
                    other_value = str(labels.get(other_field, "unknown"))
                    secondary += 1.0 / max(float(total_counts.get(other_field, {}).get(other_value, 1)), 1.0)
                if secondary > best_secondary:
                    best_idx = idx
                    best_secondary = secondary
            if best_idx is not None:
                add_idx(best_idx)

    while len(selected) < max_samples:
        best_idx = None
        best_score = -1.0
        for idx in order:
            if idx in selected_set:
                continue
            labels = labels_by_idx[idx]
            score = 0.0
            for field in balance_fields:
                value = str(labels.get(field, "unknown"))
                total = max(float(total_counts.get(field, {}).get(value, 1)), 1.0)
                current = float(selected_counts.get(field, {}).get(value, 0))
                rarity = 1.0 / np.sqrt(total)
                need = 1.0 / np.sqrt(current + 1.0)
                score += rarity * need
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            break
        add_idx(best_idx)

    selected_rows = [rows[idx] for idx in selected]
    return selected_rows, {
        "sampling_strategy": "coverage",
        "seed": int(seed),
        "candidate_pool_size": int(len(rows)),
        "selected_size": int(len(selected_rows)),
        "balance_fields": list(balance_fields),
        "candidate_label_counts": {
            field: dict(sorted(values.items())) for field, values in sorted(total_counts.items())
        },
    }


def preprocess_porto_raw(args: argparse.Namespace) -> Dict:
    raw_path = Path(args.raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw Porto CSV not found: {raw_path}")

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data" / args.dataset
    instruction_dir = output_dir / "instruction_data"
    if bool(getattr(args, "overwrite", False)):
        for path in (data_dir, instruction_dir):
            if path.exists():
                shutil.rmtree(path)
    data_dir.mkdir(parents=True, exist_ok=True)
    instruction_dir.mkdir(parents=True, exist_ok=True)

    sample_strategy = str(getattr(args, "sample_strategy", "coverage"))
    sample_seed = int(getattr(args, "sample_seed", 42))
    balance_fields = parse_csv_list(
        getattr(args, "balance_fields", "direction,curvature,progress,distance,reference_adherence")
    )
    if sample_strategy not in {"coverage", "sequential"}:
        raise ValueError(f"Unsupported --sample-strategy {sample_strategy!r}; expected coverage or sequential.")
    total_rows_estimate = None
    if (
        not getattr(args, "no_progress", False)
        and not getattr(args, "no_estimate_total_rows", False)
        and int(args.scan_limit) <= 0
    ):
        print("[preprocess] estimating raw CSV row count for ETA...", file=sys.stderr, flush=True)
        estimate_start = time.time()
        total_rows_estimate = count_csv_data_rows(raw_path)
        print(
            f"[preprocess] estimated raw CSV rows={total_rows_estimate:,} "
            f"({time.time() - estimate_start:.1f}s)",
            file=sys.stderr,
            flush=True,
        )

    if (
        int(args.max_samples) <= 0
        and sample_strategy == "coverage"
        and not getattr(args, "disable_streaming_full", False)
    ):
        return stream_full_porto_preprocess(
            args=args,
            raw_path=raw_path,
            data_dir=data_dir,
            instruction_dir=instruction_dir,
            total_rows_estimate=total_rows_estimate,
        )

    scan_max_samples = args.max_samples if sample_strategy == "sequential" else 0
    rows = list(
        iter_valid_porto_rows(
            raw_path=raw_path,
            min_traj_len=args.min_traj_len,
            max_samples=scan_max_samples,
            scan_limit=args.scan_limit,
            progress=not getattr(args, "no_progress", False),
            progress_interval=int(getattr(args, "progress_interval", 50000)),
            estimate_total_rows=not getattr(args, "no_estimate_total_rows", False),
            total_rows_estimate=total_rows_estimate,
        )
    )
    if not rows:
        raise ValueError("No valid Porto trajectories found. Check --raw-path, --scan-limit, and --min-traj-len.")
    candidate_heads = np.asarray(
        [compute_trip_attributes(row["trajectory"], int(row["timestamp"])) for row in rows],
        dtype=np.float32,
    )
    progress_pace_thresholds = build_progress_pace_thresholds(candidate_heads)
    if sample_strategy == "coverage":
        rows, sampling_summary = select_coverage_rows(
            rows=rows,
            max_samples=args.max_samples,
            target_len=args.target_len,
            dataset=args.dataset,
            seed=sample_seed,
            balance_fields=balance_fields,
            progress_pace_thresholds=progress_pace_thresholds,
        )
    else:
        sampling_summary = {
            "sampling_strategy": "sequential",
            "candidate_pool_size": int(len(rows)),
            "selected_size": int(len(rows)),
        }

    all_points = np.asarray([pt for row in rows for pt in row["trajectory"]], dtype=np.float32)
    stats = {
        "dataset": args.dataset,
        "lon_mean": float(all_points[:, 0].mean()),
        "lon_std": float(max(all_points[:, 0].std(), 1e-8)),
        "lat_mean": float(all_points[:, 1].mean()),
        "lat_std": float(max(all_points[:, 1].std(), 1e-8)),
        "porto_lon_bound": list(PORTO_LON_BOUND),
        "porto_lat_bound": list(PORTO_LAT_BOUND),
        "boundary_margin": float(BOUNDARY_MARGIN),
        "source_raw_path": str(raw_path),
        "target_len": int(args.target_len),
        "preprocessing_protocol_version": PREPROCESSING_PROTOCOL_VERSION,
        "reference_protocol": REFERENCE_PROTOCOL,
        "condition_source": CONDITION_SOURCE,
        "target_source": TARGET_SOURCE,
        "progress_semantics": "trip_pace_from_raw_timestamp_attributes",
        "progress_pace_thresholds": progress_pace_thresholds,
    }

    trajs = []
    roads = []
    heads = []
    instruction_records = []
    reference_audits = []
    for idx, row in enumerate(rows):
        raw_points = np.asarray(row["trajectory"], dtype=np.float32)
        raw_resampled = resample_polyline(raw_points, args.target_len)
        od_chord = build_od_chord(raw_points, args.target_len)
        audit = audit_od_chord_reference(raw_resampled, od_chord)
        if audit["reference_chord_error_max_m"] > 1e-3:
            raise AssertionError(
                "Official Porto preprocessing requires an exact OD-chord reference; "
                f"sample_id={idx} has chord error {audit['reference_chord_error_max_m']:.6f}m."
            )
        norm_resampled = normalize_traj(raw_resampled, stats)
        road_segments = make_road_segments(od_chord, stats, args.target_len)
        trip_values = compute_trip_attributes(row["trajectory"], int(row["timestamp"]))
        trip_attrs = trip_attribute_dict(trip_values)
        heads.append(trip_values)
        trajs.append(norm_resampled.T.astype(np.float32))
        roads.append(road_segments.astype(np.float32))
        record = mine_instruction_record(
            sample_id=idx,
            dataset=args.dataset,
            traj=norm_resampled.T,
            road=road_segments,
            stats=stats,
            include_coordinates=True,
        ).to_dict()
        record["labels"]["progress"] = progress_label_from_trip_attrs(trip_attrs, progress_pace_thresholds)
        record["instruction"] = build_instruction(record["labels"])
        record["metrics"].update(
            {
                **trip_attrs,
                "progress_semantics": "trip_pace_from_raw_timestamp_attributes",
                "progress_speed_slow_threshold_kmh": float(progress_pace_thresholds["slow_max_avg_speed_kmh"]),
                "progress_speed_fast_threshold_kmh": float(progress_pace_thresholds["fast_min_avg_speed_kmh"]),
            }
        )
        record["trip_attributes"] = {
            **trip_attrs,
            "progress_pace_thresholds": dict(progress_pace_thresholds),
        }
        record["progress_semantics"] = "trip_pace_from_raw_timestamp_attributes"
        record["progress_pace_thresholds"] = dict(progress_pace_thresholds)
        # Persist the audited lon/lat arrays directly. The mining helper
        # reconstructs coordinates through normalized road segments, which is
        # fine for labels but can introduce sub-meter serialization drift in
        # the leakage audit.
        record["trajectory"] = raw_resampled.astype(np.float32).tolist()
        record["reference"] = od_chord.astype(np.float32).tolist()
        record["raw_trip_id"] = row["trip_id"]
        record["preprocessing_protocol_version"] = PREPROCESSING_PROTOCOL_VERSION
        record["reference_protocol"] = REFERENCE_PROTOCOL
        record["condition_source"] = CONDITION_SOURCE
        record["target_source"] = TARGET_SOURCE
        record["leakage_guard"] = dict(LEAKAGE_GUARD)
        record["reference_audit"] = audit
        instruction_records.append(record)
        reference_audits.append(audit)

    trajs_arr = np.stack(trajs, axis=0).astype(np.float32)
    roads_arr = np.stack(roads, axis=0).astype(np.float32)
    heads_arr = np.asarray(heads, dtype=np.float32)

    traj_path = data_dir / f"{args.dataset}_trajs.npy"
    road_path = data_dir / f"{args.dataset}_roads.npy"
    head_path = data_dir / f"{args.dataset}_heads.npy"
    stats_path = data_dir / f"{args.dataset}_global_stats.json"
    jsonl_path = instruction_dir / f"{args.dataset}_instruction_samples.jsonl"
    summary_path = instruction_dir / f"{args.dataset}_instruction_summary.json"

    np.save(traj_path, trajs_arr, allow_pickle=False)
    np.save(road_path, roads_arr, allow_pickle=False)
    np.save(head_path, heads_arr, allow_pickle=False)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in instruction_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lengths = np.asarray([r["metrics"]["length_m"] for r in instruction_records], dtype=np.float32)
    chord_errors = np.asarray([a["reference_chord_error_max_m"] for a in reference_audits], dtype=np.float32)
    ref_target_distances = np.asarray(
        [a["reference_to_target_distance_mean_m"] for a in reference_audits],
        dtype=np.float32,
    )
    summary = {
        "dataset": args.dataset,
        "raw_path": str(raw_path),
        "num_samples": int(len(rows)),
        "target_len": int(args.target_len),
        "min_traj_len": int(args.min_traj_len),
        "sampling": sampling_summary,
        "trajs_shape": list(trajs_arr.shape),
        "roads_shape": list(roads_arr.shape),
        "heads_shape": list(heads_arr.shape),
        "data_dir": str(data_dir),
        "instruction_jsonl": str(jsonl_path),
        "preprocessing_protocol_version": PREPROCESSING_PROTOCOL_VERSION,
        "reference_protocol": REFERENCE_PROTOCOL,
        "condition_source": CONDITION_SOURCE,
        "target_source": TARGET_SOURCE,
        "progress_semantics": "trip_pace_from_raw_timestamp_attributes",
        "progress_pace_thresholds": progress_pace_thresholds,
        "leakage_guard": dict(LEAKAGE_GUARD),
        "reference_audit": {
            "reference_chord_error_max_m": float(chord_errors.max(initial=0.0)),
            "reference_chord_error_p95_m": float(np.percentile(chord_errors, 95) if len(chord_errors) else 0.0),
            "reference_to_target_distance_mean_m": float(
                ref_target_distances.mean() if len(ref_target_distances) else 0.0
            ),
            "reference_to_target_distance_p50_m": float(
                np.percentile(ref_target_distances, 50) if len(ref_target_distances) else 0.0
            ),
            "reference_to_target_distance_p90_m": float(
                np.percentile(ref_target_distances, 90) if len(ref_target_distances) else 0.0
            ),
            "reference_non_endpoint_source_points": 0,
        },
        "label_counts": summarize_label_counts(instruction_records),
        "length_m": {
            "mean": float(lengths.mean()),
            "p50": float(np.percentile(lengths, 50)),
            "p90": float(np.percentile(lengths, 90)),
        },
        "stats": stats,
    }
    split_ratios = parse_split_ratios(getattr(args, "split_ratios", None))
    split_seed = int(getattr(args, "split_seed", 20260617))
    print(
        f"[preprocess] building split assignment: records={len(instruction_records):,}, "
        f"ratios={split_ratios}, seed={split_seed}",
        file=sys.stderr,
        flush=True,
    )
    split_by_id = assign_sample_splits(range(len(instruction_records)), ratios=split_ratios, seed=split_seed)
    print("[preprocess] writing train/val/test instruction split files...", file=sys.stderr, flush=True)
    split_info = write_instruction_split_files(
        input_jsonl=jsonl_path,
        split_by_id=split_by_id,
        dataset=args.dataset,
        ratios=split_ratios,
        seed=split_seed,
        manifest_path=instruction_dir / f"{args.dataset}_split_manifest.json",
    )
    if str(getattr(args, "storage_format", "expanded")) == "compact":
        compact_info = compact_instruction_split_files(
            input_paths={
                split: Path(split_info["instruction_split_jsonl"][split])
                for split in SPLIT_ORDER
            },
            output_dir=instruction_dir,
            dataset=args.dataset,
            target_len=int(args.target_len),
            overwrite=True,
        )
        split_info["compact"] = compact_info
        split_info = cleanup_expanded_instruction_jsonl(split_info, jsonl_path)
        summary["instruction_jsonl"] = None
        summary["storage_format"] = "compact"
    summary["split"] = {
        "split_unit": "base_sample_id",
        "split_ratios": split_info["manifest"]["split_ratios"],
        "split_counts": split_info["manifest"]["split_counts"],
        "split_seed": split_seed,
        "manifest_path": split_info["manifest_path"],
        "instruction_split_jsonl": split_info["instruction_split_jsonl"],
        "instruction_jsonl": split_info.get("instruction_jsonl", str(jsonl_path)),
        "temporary_expanded_instruction_jsonl_removed": split_info.get(
            "temporary_expanded_instruction_jsonl_removed",
            [],
        ),
        "compact": split_info.get("compact"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess a local raw Porto CSV subset for LangDirectTraj using the single "
            "official OD-chord reference protocol. The GT trajectory middle geometry is "
            "saved only as the target, never reconstructed as a strong reference condition."
        )
    )
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "local_porto_od_chord")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--scan-limit", type=int, default=20000)
    parser.add_argument(
        "--sample-strategy",
        choices=["coverage", "sequential"],
        default="coverage",
        help=(
            "coverage scans --scan-limit valid trips and selects a deterministic token-covering subset "
            "for local Table 3/4 evaluation; sequential keeps the old prefix behavior."
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--balance-fields",
        default="direction,curvature,progress,distance,reference_adherence",
        help="Comma-separated instruction fields used by coverage sampling.",
    )
    parser.add_argument("--min-traj-len", type=int, default=30)
    parser.add_argument("--target-len", type=int, default=200)
    parser.add_argument("--no-progress", action="store_true", help="Disable raw CSV scan progress output.")
    parser.add_argument("--progress-interval", type=int, default=50000, help="Fallback progress print interval when tqdm is unavailable.")
    parser.add_argument(
        "--no-estimate-total-rows",
        action="store_true",
        help="Do not pre-count raw CSV rows for full-scan ETA.",
    )
    parser.add_argument(
        "--disable-streaming-full",
        action="store_true",
        help="Use the legacy in-memory full preprocessing path even when --max-samples 0.",
    )
    parser.add_argument(
        "--split-ratios",
        default="0.8,0.1,0.1",
        help="Base-sample split ratios as train,val,test. Default: 0.8,0.1,0.1.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=20260617,
        help="Seed for deterministic base-sample train/val/test split.",
    )
    parser.add_argument(
        "--storage-format",
        choices=["expanded", "compact"],
        default="expanded",
        help="expanded writes legacy self-contained JSONL; compact also writes .npy-backed metadata JSONL.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove generated data/<dataset> and instruction_data outputs before preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(preprocess_porto_raw(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
