from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from langdirecttraj.data.compact_store import compact_instruction_split_files
from langdirecttraj.data.preprocess_porto_raw import (
    CONDITION_SOURCE,
    LEAKAGE_GUARD,
    REFERENCE_PROTOCOL,
    TARGET_SOURCE,
    audit_od_chord_reference,
    build_od_chord,
    build_progress_pace_thresholds,
    cleanup_expanded_instruction_jsonl,
    make_road_segments,
    normalize_traj,
    progress_label_from_trip_attrs,
    resample_polyline,
    select_coverage_rows,
    summarize_label_counts,
    trip_attribute_dict,
    write_instruction_split_files,
)
from langdirecttraj.data.split_manifest import SPLIT_ORDER, assign_sample_splits, parse_split_ratios
from langdirecttraj.instructions import mine_instruction_record
from langdirecttraj.instructions.templates import build_instruction


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(r"C:\硕士\数据集")
CITY_BOUNDS = {
    "rome": {"lon": (12.20, 12.75), "lat": (41.65, 42.10)},
    "roma": {"lon": (12.20, 12.75), "lat": (41.65, 42.10)},
    "cabspotting": {"lon": (-122.60, -122.25), "lat": (37.55, 37.90)},
    "sf": {"lon": (-122.60, -122.25), "lat": (37.55, 37.90)},
    "sanfrancisco": {"lon": (-122.60, -122.25), "lat": (37.55, 37.90)},
    "beijing": {"lon": (115.70, 117.40), "lat": (39.40, 41.10)},
    "tdrive": {"lon": (115.70, 117.40), "lat": (39.40, 41.10)},
    "t-drive": {"lon": (115.70, 117.40), "lat": (39.40, 41.10)},
}
POINT_RE = re.compile(r"POINT\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\)")
UTC = timezone.utc


def _canonical_dataset(dataset: str) -> str:
    key = dataset.lower().strip()
    if key == "roma":
        return "rome"
    if key in {"sf", "sanfrancisco"}:
        return "cabspotting"
    if key in {"tdrive", "t-drive"}:
        return "beijing"
    return key


def _default_raw_path(dataset: str) -> Path:
    key = _canonical_dataset(dataset)
    if key == "rome":
        return DEFAULT_DATA_ROOT / "Rome" / "taxi_february.txt"
    if key == "cabspotting":
        return DEFAULT_DATA_ROOT / "cabspottingdata"
    if key == "beijing":
        return DEFAULT_DATA_ROOT / "T-Drive"
    raise ValueError(f"Unsupported dataset {dataset!r}. Use rome, cabspotting, or beijing.")


def _in_bounds(lon: float, lat: float, dataset: str) -> bool:
    bounds = CITY_BOUNDS[_canonical_dataset(dataset)]
    return (
        bounds["lon"][0] <= lon <= bounds["lon"][1]
        and bounds["lat"][0] <= lat <= bounds["lat"][1]
        and math.isfinite(lon)
        and math.isfinite(lat)
    )


def _distance_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lon1, lat1 = math.radians(float(a[0])), math.radians(float(a[1]))
    lon2, lat2 = math.radians(float(b[0])), math.radians(float(b[1]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def _path_length_km(points: List[Tuple[float, float]]) -> float:
    return float(sum(_distance_km(a, b) for a, b in zip(points[:-1], points[1:])))


def _trip_attributes(points: List[Tuple[float, float]], timestamps: List[int]) -> List[float]:
    start_ts = int(timestamps[0]) if timestamps else 0
    start_dt = datetime.fromtimestamp(max(start_ts, 0), tz=UTC)
    start_total_min = start_dt.hour * 60 + start_dt.minute
    start_cat = float(np.clip(start_total_min // 5, 0, 287))
    duration_min = max((int(timestamps[-1]) - int(timestamps[0])) / 60.0, 0.1) if len(timestamps) > 1 else 0.1
    dist_km = max(_path_length_km(points), 0.1)
    avg_speed = float(np.clip(dist_km / (duration_min / 60.0), 0.0, 160.0))
    time_per_km = float(np.clip(duration_min / dist_km, 0.0, 120.0))
    return [start_cat, float(start_total_min), float(duration_min), float(dist_km), avg_speed, time_per_km]


def _keep_trip(
    points: List[Tuple[float, float]],
    timestamps: List[int],
    *,
    min_traj_len: int,
    min_distance_km: float,
    max_duration_min: float,
    max_speed_kmh: float,
) -> bool:
    if len(points) < int(min_traj_len):
        return False
    if len(timestamps) != len(points):
        return False
    duration_min = max((int(timestamps[-1]) - int(timestamps[0])) / 60.0, 0.0)
    if duration_min <= 0.0 or duration_min > float(max_duration_min):
        return False
    dist_km = _path_length_km(points)
    if dist_km < float(min_distance_km):
        return False
    avg_speed = dist_km / max(duration_min / 60.0, 1e-6)
    if avg_speed <= 1.0 or avg_speed > float(max_speed_kmh):
        return False
    return True


def _yield_segment(
    trip_id: str,
    taxi_id: str,
    points: List[Tuple[float, float]],
    timestamps: List[int],
    args: argparse.Namespace,
) -> Dict | None:
    if not _keep_trip(
        points,
        timestamps,
        min_traj_len=args.min_traj_len,
        min_distance_km=args.min_distance_km,
        max_duration_min=args.max_duration_min,
        max_speed_kmh=args.max_speed_kmh,
    ):
        return None
    return {
        "trip_id": trip_id,
        "taxi_id": taxi_id,
        "timestamp": int(timestamps[0]),
        "timestamps": [int(t) for t in timestamps],
        "trajectory": [(float(lon), float(lat)) for lon, lat in points],
    }


def _parse_rome_line(line: str, dataset: str) -> Tuple[str, int, Tuple[float, float]] | None:
    parts = line.strip().split(";")
    if len(parts) < 3:
        return None
    taxi_id = parts[0].strip()
    try:
        dt = datetime.fromisoformat(parts[1].strip())
        ts = int(dt.timestamp())
    except ValueError:
        return None
    match = POINT_RE.search(parts[2])
    if not match:
        return None
    # The CRAWDAD Rome text stores POINT(latitude longitude).
    lat = float(match.group(1))
    lon = float(match.group(2))
    if not _in_bounds(lon, lat, dataset):
        return None
    return taxi_id, ts, (lon, lat)


def iter_rome_trips(args: argparse.Namespace) -> Iterator[Dict]:
    raw_path = Path(args.raw_path)
    active: Dict[str, Tuple[List[Tuple[float, float]], List[int]]] = {}
    scanned = 0
    kept = 0
    emitted = 0
    pbar = tqdm(desc="scan Rome points", unit="point", dynamic_ncols=True) if tqdm is not None and not args.no_progress else None
    started = time.time()
    with raw_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            scanned += 1
            if args.point_scan_limit > 0 and scanned > int(args.point_scan_limit):
                break
            parsed = _parse_rome_line(line, args.dataset)
            if pbar is not None:
                pbar.update(1)
            if parsed is None:
                continue
            taxi_id, ts, point = parsed
            points, timestamps = active.get(taxi_id, ([], []))
            if timestamps and (ts <= timestamps[-1] or ts - timestamps[-1] > int(args.max_gap_seconds)):
                rec = _yield_segment(f"rome_{taxi_id}_{timestamps[0]}", taxi_id, points, timestamps, args)
                if rec is not None:
                    kept += 1
                    yield rec
                    emitted += 1
                    if args.max_samples > 0 and emitted >= int(args.max_samples):
                        if pbar is not None:
                            pbar.close()
                        return
                points, timestamps = [], []
            points.append(point)
            timestamps.append(ts)
            active[taxi_id] = (points, timestamps)
        for taxi_id, (points, timestamps) in active.items():
            rec = _yield_segment(f"rome_{taxi_id}_{timestamps[0] if timestamps else 0}", taxi_id, points, timestamps, args)
            if rec is not None:
                kept += 1
                yield rec
                emitted += 1
                if args.max_samples > 0 and emitted >= int(args.max_samples):
                    break
    if pbar is not None:
        pbar.close()
    if not args.no_progress:
        elapsed = max(time.time() - started, 1e-6)
        print(f"[public-taxi] Rome scan complete: points={scanned:,}, kept_trips={kept:,}, elapsed={elapsed/60:.1f} min", file=sys.stderr)


def _read_sf_file(path: Path, dataset: str) -> List[Tuple[int, int, Tuple[float, float]]]:
    rows: List[Tuple[int, int, Tuple[float, float]]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                occ = int(float(parts[2]))
                ts = int(float(parts[3]))
            except ValueError:
                continue
            if not _in_bounds(lon, lat, dataset):
                continue
            rows.append((ts, occ, (lon, lat)))
    rows.sort(key=lambda item: item[0])
    return rows


def iter_cabspotting_trips(args: argparse.Namespace) -> Iterator[Dict]:
    raw_dir = Path(args.raw_path)
    files = sorted(raw_dir.glob("new_*.txt"))
    if args.file_limit > 0:
        files = files[: int(args.file_limit)]
    file_iter: Iterable[Path] = files
    if tqdm is not None and not args.no_progress:
        file_iter = tqdm(files, desc="scan SF taxis", unit="taxi", dynamic_ncols=True)
    emitted = 0
    kept = 0
    for path in file_iter:
        taxi_id = path.stem.replace("new_", "")
        points: List[Tuple[float, float]] = []
        timestamps: List[int] = []
        prev_occ: int | None = None
        for ts, occ, point in _read_sf_file(path, args.dataset):
            use_point = occ == int(args.sf_occupancy_value)
            gap_break = bool(timestamps and ts - timestamps[-1] > int(args.max_gap_seconds))
            occ_break = bool(prev_occ is not None and occ != prev_occ)
            if (gap_break or occ_break or not use_point) and points:
                rec = _yield_segment(f"sf_{taxi_id}_{timestamps[0]}", taxi_id, points, timestamps, args)
                if rec is not None:
                    kept += 1
                    yield rec
                    emitted += 1
                    if args.max_samples > 0 and emitted >= int(args.max_samples):
                        return
                points, timestamps = [], []
            if use_point:
                points.append(point)
                timestamps.append(ts)
            prev_occ = occ
        if points:
            rec = _yield_segment(f"sf_{taxi_id}_{timestamps[0]}", taxi_id, points, timestamps, args)
            if rec is not None:
                kept += 1
                yield rec
                emitted += 1
                if args.max_samples > 0 and emitted >= int(args.max_samples):
                    return
    if not args.no_progress:
        print(f"[public-taxi] SF scan complete: files={len(files):,}, kept_trips={kept:,}", file=sys.stderr)


def _tdrive_sort_key(path: Path) -> Tuple[int, str]:
    try:
        return int(path.stem), path.stem
    except ValueError:
        return 10**12, path.stem


def _parse_tdrive_line(line: str, dataset: str) -> Tuple[str, int, Tuple[float, float]] | None:
    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) < 4:
        return None
    taxi_id = parts[0]
    try:
        dt = datetime.strptime(parts[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        lon = float(parts[2])
        lat = float(parts[3])
    except ValueError:
        return None
    if not _in_bounds(lon, lat, dataset):
        return None
    return taxi_id, int(dt.timestamp()), (lon, lat)


def iter_tdrive_trips(args: argparse.Namespace) -> Iterator[Dict]:
    raw_path = Path(args.raw_path)
    raw_dir = raw_path / "taxi_log_2008_by_id" if (raw_path / "taxi_log_2008_by_id").exists() else raw_path
    files = sorted(raw_dir.glob("*.txt"), key=_tdrive_sort_key)
    if args.file_limit > 0:
        files = files[: int(args.file_limit)]
    file_iter: Iterable[Path] = files
    if tqdm is not None and not args.no_progress:
        file_iter = tqdm(files, desc="scan T-Drive taxis", unit="taxi", dynamic_ncols=True)
    emitted = 0
    kept = 0
    scanned = 0
    for path in file_iter:
        points: List[Tuple[float, float]] = []
        timestamps: List[int] = []
        taxi_id = path.stem
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                scanned += 1
                if args.point_scan_limit > 0 and scanned > int(args.point_scan_limit):
                    break
                parsed = _parse_tdrive_line(line, args.dataset)
                if parsed is None:
                    continue
                parsed_taxi_id, ts, point = parsed
                taxi_id = parsed_taxi_id or taxi_id
                if timestamps and ts <= timestamps[-1]:
                    if ts == timestamps[-1] and points and point == points[-1]:
                        continue
                    rec = _yield_segment(f"beijing_{taxi_id}_{timestamps[0]}", taxi_id, points, timestamps, args)
                    if rec is not None:
                        kept += 1
                        yield rec
                        emitted += 1
                        if args.max_samples > 0 and emitted >= int(args.max_samples):
                            return
                    points, timestamps = [], []
                if timestamps and ts - timestamps[-1] > int(args.max_gap_seconds):
                    rec = _yield_segment(f"beijing_{taxi_id}_{timestamps[0]}", taxi_id, points, timestamps, args)
                    if rec is not None:
                        kept += 1
                        yield rec
                        emitted += 1
                        if args.max_samples > 0 and emitted >= int(args.max_samples):
                            return
                    points, timestamps = [], []
                points.append(point)
                timestamps.append(ts)
        if points:
            rec = _yield_segment(f"beijing_{taxi_id}_{timestamps[0]}", taxi_id, points, timestamps, args)
            if rec is not None:
                kept += 1
                yield rec
                emitted += 1
                if args.max_samples > 0 and emitted >= int(args.max_samples):
                    return
        if args.point_scan_limit > 0 and scanned > int(args.point_scan_limit):
            break
    if not args.no_progress:
        print(f"[public-taxi] T-Drive scan complete: files={len(files):,}, points={scanned:,}, kept_trips={kept:,}", file=sys.stderr)


def iter_public_taxi_trips(args: argparse.Namespace) -> Iterator[Dict]:
    key = _canonical_dataset(args.dataset)
    if key == "rome":
        yield from iter_rome_trips(args)
        return
    if key == "cabspotting":
        yield from iter_cabspotting_trips(args)
        return
    if key == "beijing":
        yield from iter_tdrive_trips(args)
        return
    raise ValueError(f"Unsupported dataset {args.dataset!r}.")


def _stats_from_rows(rows: List[Dict], dataset: str) -> Dict[str, float]:
    all_points = np.asarray([pt for row in rows for pt in row["trajectory"]], dtype=np.float32)
    bounds = CITY_BOUNDS[_canonical_dataset(dataset)]
    return {
        "dataset": dataset,
        "lon_mean": float(all_points[:, 0].mean()),
        "lon_std": float(max(all_points[:, 0].std(), 1e-8)),
        "lat_mean": float(all_points[:, 1].mean()),
        "lat_std": float(max(all_points[:, 1].std(), 1e-8)),
        "lon_bound": list(bounds["lon"]),
        "lat_bound": list(bounds["lat"]),
        "source": "public_taxi_raw_gate",
    }


def _write_real_npz(dataset: str, instruction_dir: Path, output_dir: Path, split: str, max_samples: int = 0) -> Dict:
    meta_path = instruction_dir / f"{dataset}_instruction_meta.{split}.jsonl"
    base_path = instruction_dir / f"{dataset}_base_lonlat.npy"
    rows: List[int] = []
    sample_ids: List[int] = []
    lengths: List[float] = []
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_samples > 0 and len(rows) >= int(max_samples):
                break
            record = json.loads(line)
            spec = record.get("compact_refs", {}).get("trajectory", {})
            row = spec.get("row", record.get("base_index"))
            if row is None:
                raise ValueError(f"Missing compact trajectory row in {meta_path}")
            rows.append(int(row))
            sample_ids.append(int(record.get("sample_id", len(sample_ids))))
            lengths.append(float(record.get("metrics", {}).get("length_m", np.nan)))
    base = np.load(base_path, mmap_mode="r")
    row_arr = np.asarray(rows, dtype=np.int64)
    traj = np.asarray(base[row_arr], dtype=np.float32)
    output = output_dir / f"{dataset}_{split}_real_trajectories.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        real_trajectories=traj,
        sample_id=np.asarray(sample_ids, dtype=np.int64),
        base_index=row_arr,
        gt_length_m=np.asarray(lengths, dtype=np.float32),
    )
    return {
        "split": split,
        "num_samples": int(traj.shape[0]),
        "trajectory_shape": list(traj.shape),
        "output": str(output),
        "gt_avg_length_km": float(np.nanmean(lengths) / 1000.0) if lengths else float("nan"),
    }


def _label_health(label_counts: Dict[str, Dict[str, int]]) -> Dict:
    rows = {}
    pass_fields = 0
    for field, counts in sorted(label_counts.items()):
        total = int(sum(counts.values()))
        max_frac = max((int(v) / max(total, 1) for v in counts.values()), default=1.0)
        nonzero = sum(1 for v in counts.values() if int(v) > 0)
        ok = bool(total > 0 and nonzero >= 2 and max_frac <= 0.95)
        if field != "reference_adherence" and ok:
            pass_fields += 1
        rows[field] = {
            "total": total,
            "nonzero_labels": int(nonzero),
            "max_label_fraction": float(max_frac),
            "gate_ok": ok,
            "counts": counts,
        }
    return {
        "fields": rows,
        "non_reference_fields_passing": int(pass_fields),
        "table3_table4_support_likely": bool(pass_fields >= 3),
    }


def preprocess_public_taxi_raw(args: argparse.Namespace) -> Dict:
    args.dataset = _canonical_dataset(args.dataset)
    args.raw_path = Path(args.raw_path) if args.raw_path else _default_raw_path(args.dataset)
    base_output = Path(args.output_dir)
    output_root = base_output if base_output.name.lower() == args.dataset else base_output / args.dataset
    data_dir = output_root / "data" / args.dataset
    instruction_dir = output_root / "instruction_data"
    real_dir = output_root / "real"
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    instruction_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_public_taxi_trips(args))
    if not rows:
        raise ValueError(f"No valid {args.dataset} trips found. Check raw path and gate thresholds.")

    heads = np.asarray([_trip_attributes(row["trajectory"], row["timestamps"]) for row in rows], dtype=np.float32)
    progress_thresholds = build_progress_pace_thresholds(heads)
    balance_fields = [item.strip() for item in str(args.balance_fields).split(",") if item.strip()]
    if args.sample_strategy == "coverage":
        rows, sampling_summary = select_coverage_rows(
            rows=rows,
            max_samples=int(args.max_samples),
            target_len=int(args.target_len),
            dataset=args.dataset,
            seed=int(args.sample_seed),
            balance_fields=balance_fields,
            progress_pace_thresholds=progress_thresholds,
        )
    elif args.max_samples > 0 and len(rows) > int(args.max_samples):
        rows = rows[: int(args.max_samples)]
        sampling_summary = {"sampling_strategy": "sequential", "selected_size": int(len(rows))}
    else:
        sampling_summary = {"sampling_strategy": args.sample_strategy, "selected_size": int(len(rows))}

    stats = _stats_from_rows(rows, args.dataset)
    instruction_jsonl = instruction_dir / f"{args.dataset}_instruction_samples.jsonl"
    trajs = []
    roads = []
    heads = []
    records: List[Dict] = []
    audits = []
    for sample_id, row in enumerate(rows):
        raw_points = np.asarray(row["trajectory"], dtype=np.float32)
        raw_resampled = resample_polyline(raw_points, int(args.target_len))
        od_chord = build_od_chord(raw_points, int(args.target_len))
        audit = audit_od_chord_reference(raw_resampled, od_chord)
        norm_resampled = normalize_traj(raw_resampled, stats)
        road_segments = make_road_segments(od_chord, stats, int(args.target_len))
        trip_values = _trip_attributes(row["trajectory"], row["timestamps"])
        trip_attrs = trip_attribute_dict(trip_values)
        heads.append(trip_values)
        trajs.append(norm_resampled.T.astype(np.float32))
        roads.append(road_segments.astype(np.float32))
        record = mine_instruction_record(
            sample_id=sample_id,
            dataset=args.dataset,
            traj=norm_resampled.T,
            road=road_segments,
            stats=stats,
            include_coordinates=True,
        ).to_dict()
        record["labels"]["progress"] = progress_label_from_trip_attrs(trip_attrs, progress_thresholds)
        record["instruction"] = build_instruction(record["labels"])
        record["metrics"].update(
            {
                **trip_attrs,
                "progress_semantics": "trip_pace_from_raw_timestamps",
                "progress_speed_slow_threshold_kmh": float(progress_thresholds["slow_max_avg_speed_kmh"]),
                "progress_speed_fast_threshold_kmh": float(progress_thresholds["fast_min_avg_speed_kmh"]),
            }
        )
        record["trip_attributes"] = {**trip_attrs, "progress_pace_thresholds": dict(progress_thresholds)}
        record["progress_semantics"] = "trip_pace_from_raw_timestamps"
        record["progress_pace_thresholds"] = dict(progress_thresholds)
        record["trajectory"] = raw_resampled.astype(np.float32).tolist()
        record["reference"] = od_chord.astype(np.float32).tolist()
        record["raw_trip_id"] = row["trip_id"]
        record["taxi_id"] = row.get("taxi_id")
        record["raw_timestamps"] = [int(t) for t in row["timestamps"]]
        record["preprocessing_protocol_version"] = f"{args.dataset}_od_chord_v1"
        record["reference_protocol"] = REFERENCE_PROTOCOL
        record["condition_source"] = CONDITION_SOURCE
        record["target_source"] = TARGET_SOURCE
        record["leakage_guard"] = dict(LEAKAGE_GUARD)
        record["reference_audit"] = audit
        records.append(record)
        audits.append(audit)

    np.save(data_dir / f"{args.dataset}_trajs.npy", np.stack(trajs).astype(np.float32), allow_pickle=False)
    np.save(data_dir / f"{args.dataset}_roads.npy", np.stack(roads).astype(np.float32), allow_pickle=False)
    np.save(data_dir / f"{args.dataset}_heads.npy", np.asarray(heads, dtype=np.float32), allow_pickle=False)
    (data_dir / f"{args.dataset}_global_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    with instruction_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    split_ratios = parse_split_ratios(args.split_ratios)
    split_by_id = assign_sample_splits(range(len(records)), ratios=split_ratios, seed=int(args.split_seed))
    split_info = write_instruction_split_files(
        input_jsonl=instruction_jsonl,
        split_by_id=split_by_id,
        dataset=args.dataset,
        ratios=split_ratios,
        seed=int(args.split_seed),
        manifest_path=instruction_dir / f"{args.dataset}_split_manifest.json",
    )
    compact_info = None
    if args.storage_format == "compact":
        compact_info = compact_instruction_split_files(
            input_paths={split: Path(split_info["instruction_split_jsonl"][split]) for split in SPLIT_ORDER},
            output_dir=instruction_dir,
            dataset=args.dataset,
            target_len=int(args.target_len),
            overwrite=True,
        )
        split_info["compact"] = compact_info
        split_info = cleanup_expanded_instruction_jsonl(split_info, instruction_jsonl)

    real_npz = {
        split: _write_real_npz(args.dataset, instruction_dir, real_dir, split)
        for split in SPLIT_ORDER
    }

    lengths = np.asarray([r["metrics"]["length_m"] for r in records], dtype=np.float32)
    ref_mean = np.asarray([a["reference_to_target_distance_mean_m"] for a in audits], dtype=np.float32)
    label_counts = summarize_label_counts(records)
    summary = {
        "dataset": args.dataset,
        "raw_path": str(args.raw_path),
        "num_samples": int(len(records)),
        "target_len": int(args.target_len),
        "min_traj_len": int(args.min_traj_len),
        "sampling": sampling_summary,
        "split": {
            "split_ratios": split_info["manifest"]["split_ratios"],
            "split_counts": split_info["manifest"]["split_counts"],
            "split_seed": int(args.split_seed),
            "manifest_path": split_info["manifest_path"],
            "instruction_split_jsonl": split_info.get("instruction_split_jsonl", {}),
            "compact": compact_info,
        },
        "real_npz": real_npz,
        "stats": stats,
        "progress_pace_thresholds": progress_thresholds,
        "length_m": {
            "mean": float(lengths.mean()),
            "p50": float(np.percentile(lengths, 50)),
            "p90": float(np.percentile(lengths, 90)),
        },
        "reference_audit": {
            "reference_to_target_distance_mean_m": float(ref_mean.mean()),
            "reference_to_target_distance_p50_m": float(np.percentile(ref_mean, 50)),
            "reference_to_target_distance_p90_m": float(np.percentile(ref_mean, 90)),
            "reference_non_endpoint_source_points": 0,
        },
        "label_counts": label_counts,
        "gate0_label_health": _label_health(label_counts),
    }
    summary_path = instruction_dir / f"{args.dataset}_instruction_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess public taxi traces into OD-chord instruction splits for Gate 0.")
    parser.add_argument("--dataset", required=True, choices=["rome", "roma", "cabspotting", "sf", "sanfrancisco", "beijing", "tdrive", "t-drive"])
    parser.add_argument("--raw-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "public_taxi_gate")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--point-scan-limit", type=int, default=0, help="Raw point scan cap for sequential text datasets; 0 scans all points.")
    parser.add_argument("--file-limit", type=int, default=0, help="Taxi file cap for directory-style datasets; 0 scans all files.")
    parser.add_argument("--sample-strategy", choices=["coverage", "sequential"], default="coverage")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--split-ratios", default="0.8,0.1,0.1")
    parser.add_argument("--split-seed", type=int, default=20260617)
    parser.add_argument("--target-len", type=int, default=200)
    parser.add_argument("--min-traj-len", type=int, default=20)
    parser.add_argument("--min-distance-km", type=float, default=0.5)
    parser.add_argument("--max-duration-min", type=float, default=180.0)
    parser.add_argument("--max-speed-kmh", type=float, default=140.0)
    parser.add_argument("--max-gap-seconds", type=int, default=600)
    parser.add_argument("--sf-occupancy-value", type=int, default=1)
    parser.add_argument("--storage-format", choices=["expanded", "compact"], default="compact")
    parser.add_argument("--balance-fields", default="direction,curvature,progress,distance,reference_adherence")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(preprocess_public_taxi_raw(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
