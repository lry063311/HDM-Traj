from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None

from langdirecttraj.instructions.templates import build_instruction
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB, LABEL_VOCAB
from langdirecttraj.data.compact_store import (
    expand_compact_record,
    is_compact_meta_path,
    load_compact_manifest,
    open_compact_arrays,
)
from langdirecttraj.data.split_manifest import (
    SPLIT_ORDER,
    load_split_manifest,
    record_split,
    summarize_split_records,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def iter_jsonl(
    path: Path,
    max_samples: int = 0,
    progress_desc: str | None = None,
    total: int | None = None,
) -> Iterable[Dict]:
    progress_total = total
    if max_samples > 0:
        progress_total = min(max_samples, progress_total) if progress_total is not None else max_samples
    pbar = None
    if progress_desc and tqdm is not None:
        pbar = tqdm(total=progress_total, desc=progress_desc, unit="record", dynamic_ncols=True)
    elif progress_desc:
        total_text = f"/{progress_total:,}" if progress_total else ""
        print(f"[counterfactual] {progress_desc} started: total={total_text or 'unknown'}", file=sys.stderr, flush=True)
    started = time.time()
    emitted = 0
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if max_samples > 0 and emitted >= max_samples:
                    break
                record = json.loads(line)
                emitted += 1
                if pbar is not None:
                    pbar.update(1)
                elif progress_desc and emitted % 50000 == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    rate = emitted / elapsed
                    if progress_total:
                        eta = max(progress_total - emitted, 0) / max(rate, 1e-6)
                        print(
                            f"[counterfactual] {progress_desc}: {emitted:,}/{progress_total:,} "
                            f"({emitted / max(progress_total, 1):.1%}), {rate:.1f} record/s, eta={eta / 60.0:.1f} min",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        print(
                            f"[counterfactual] {progress_desc}: {emitted:,}, {rate:.1f} record/s",
                            file=sys.stderr,
                            flush=True,
                        )
                yield record
    finally:
        if pbar is not None:
            pbar.close()
        if progress_desc:
            print(
                f"[counterfactual] {progress_desc} complete: records={emitted:,}, "
                f"elapsed={(time.time() - started) / 60.0:.1f} min",
                file=sys.stderr,
                flush=True,
            )


def iter_instruction_records(
    path: Path,
    max_samples: int = 0,
    progress_desc: str | None = None,
    total: int | None = None,
) -> Iterable[Dict]:
    path = Path(path)
    if not is_compact_meta_path(path):
        yield from iter_jsonl(path, max_samples=max_samples, progress_desc=progress_desc, total=total)
        return
    manifest = load_compact_manifest(path)
    arrays = open_compact_arrays(manifest)
    for record in iter_jsonl(path, max_samples=max_samples, progress_desc=progress_desc, total=total):
        yield expand_compact_record(record, arrays)


def read_jsonl(path: Path, max_samples: int = 0) -> List[Dict]:
    records = list(iter_instruction_records(path, max_samples=max_samples))
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def to_local(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    points = np.asarray(points, dtype=np.float32)
    origin = points[0].copy()
    lat = float(np.nanmean(points[:, 1]))
    scale = (111_320.0 * math.cos(math.radians(lat)), 110_540.0)
    local = np.empty_like(points, dtype=np.float32)
    local[:, 0] = (points[:, 0] - origin[0]) * scale[0]
    local[:, 1] = (points[:, 1] - origin[1]) * scale[1]
    return local, origin, scale


def from_local(local: np.ndarray, origin: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    points = np.empty_like(local, dtype=np.float32)
    points[:, 0] = local[:, 0] / scale[0] + origin[0]
    points[:, 1] = local[:, 1] / scale[1] + origin[1]
    return points


def to_shared_local(points: np.ndarray, origin: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    local = np.empty_like(points, dtype=np.float32)
    local[:, 0] = (points[:, 0] - origin[0]) * scale[0]
    local[:, 1] = (points[:, 1] - origin[1]) * scale[1]
    return local


def tangent_normal(local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(local) < 2:
        tangent = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(local), 1))
    else:
        diffs = np.zeros_like(local, dtype=np.float32)
        diffs[1:-1] = local[2:] - local[:-2]
        diffs[0] = local[1] - local[0]
        diffs[-1] = local[-1] - local[-2]
        norm = np.linalg.norm(diffs, axis=1, keepdims=True).clip(min=1e-6)
        tangent = diffs / norm
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return tangent.astype(np.float32), normal.astype(np.float32)


def line_between_endpoints(local: np.ndarray) -> np.ndarray:
    t = np.linspace(0.0, 1.0, len(local), dtype=np.float32)[:, None]
    return local[:1] * (1.0 - t) + local[-1:] * t


def polyline_length(local: np.ndarray) -> float:
    local = np.asarray(local, dtype=np.float32)
    if len(local) < 2:
        return 0.0
    return float(np.linalg.norm(local[1:] - local[:-1], axis=1).sum())


def local_turn_density_score(local: np.ndarray) -> float:
    """Curvature descriptor aligned with Table-3/4 calibration.

    Calibration uses total local turn per 100 meters.  Keeping the same scalar
    here lets pseudo targets be generated toward the metric that later trains
    and evaluates curvature control, instead of relying on visually plausible
    but weak sine offsets.
    """

    local = np.asarray(local, dtype=np.float32)
    if len(local) < 3:
        return 0.0
    diffs = local[1:] - local[:-1]
    valid = np.linalg.norm(diffs, axis=1) > 1e-6
    diffs = diffs[valid]
    if len(diffs) < 2:
        return 0.0
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    deltas = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    length_m = polyline_length(local)
    return float(np.abs(deltas).sum()) / max(length_m / 100.0, 1.0)


def progress_ratio_target(progress: str) -> float | None:
    return {
        "slow": 0.52,
        "normal": 0.67,
        "fast": 0.80,
    }.get(str(progress), None)


def distance_length_target_m(distance: str, source_len: float, chord_len: float) -> float | None:
    """Endpoint-feasible absolute-length target for distance counterfactuals.

    The pseudo target is intentionally conservative.  Table 4 only needs a
    clean target-field movement, not an implausible very_short->long detour
    under fixed OD endpoints.
    """

    centers = {
        "very_short": 900.0,
        "short": 1600.0,
        "medium": 3400.0,
        "long": 6200.0,
    }
    if distance not in centers:
        return None
    chord_len = float(max(chord_len, 1.0))
    source_len = float(max(source_len, chord_len))
    target = float(centers[distance])
    if target > source_len:
        target = min(target, 1.65 * source_len)
    return max(target, 1.02 * chord_len)


def progress_speed_target_kmh(progress: str, trip_attrs: Dict | None) -> float | None:
    thresholds = {}
    if isinstance(trip_attrs, dict):
        raw_thresholds = trip_attrs.get("progress_pace_thresholds", {})
        if isinstance(raw_thresholds, dict):
            thresholds = raw_thresholds
    slow = float(thresholds.get("slow_max_avg_speed_kmh", 12.0))
    fast = float(thresholds.get("fast_min_avg_speed_kmh", 24.0))
    if progress == "slow":
        return max(1.0, 0.85 * slow)
    if progress == "normal":
        return max(1.0, 0.5 * (slow + fast))
    if progress == "fast":
        return max(1.0, 1.10 * fast)
    return None


def reference_signed_offsets(local: np.ndarray, reference_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(reference_local) != len(local):
        reference_local = resample_polyline(reference_local, len(local))
    _, normal = tangent_normal(reference_local)
    offsets = ((local - reference_local) * normal).sum(axis=1)
    return offsets.astype(np.float32), normal.astype(np.float32)


def preserve_reference_distance(
    source_local: np.ndarray,
    candidate_local: np.ndarray,
    reference_local: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Keep non-reference counterfactuals from silently changing adherence.

    Under the OD-chord protocol, reference adherence is an independent
    instruction field. Direction, curvature, and progress pseudo targets should
    not collapse toward the chord just because their template is chord-based.
    This rescales the candidate's lateral distance from the OD chord to match
    the source trajectory's average reference distance while preserving the
    candidate's endpoint geometry.
    """

    if len(candidate_local) < 3:
        return candidate_local
    if len(reference_local) != len(candidate_local):
        reference_local = resample_polyline(reference_local, len(candidate_local))
    if len(source_local) != len(candidate_local):
        source_local = resample_polyline(source_local, len(candidate_local))

    source_offsets, normal = reference_signed_offsets(source_local, reference_local)
    candidate_offsets, _ = reference_signed_offsets(candidate_local, reference_local)
    source_mean_abs = float(np.mean(np.abs(source_offsets)))
    if not np.isfinite(source_mean_abs) or source_mean_abs < 1e-3:
        return candidate_local

    candidate_mean_abs = float(np.mean(np.abs(candidate_offsets)))
    if np.isfinite(candidate_mean_abs) and candidate_mean_abs >= 1e-3:
        target_offsets = candidate_offsets * (source_mean_abs / max(candidate_mean_abs, 1e-6))
    else:
        t = np.linspace(0.0, 1.0, len(candidate_local), dtype=np.float32)
        envelope = np.sin(np.pi * t).clip(min=0.0)
        envelope_mean = float(np.mean(np.abs(envelope)))
        side = float(np.sign(np.mean(source_offsets)))
        if abs(side) < 1e-6:
            idx = int(np.argmax(np.abs(source_offsets)))
            side = float(np.sign(source_offsets[idx]))
        if abs(side) < 1e-6:
            side = 1.0
        target_offsets = side * source_mean_abs * envelope / max(envelope_mean, 1e-6)

    strength = float(np.clip(strength, 0.0, 1.0))
    corrected_offsets = (1.0 - strength) * candidate_offsets + strength * target_offsets
    corrected = reference_local + corrected_offsets[:, None].astype(np.float32) * normal
    corrected[0] = source_local[0]
    corrected[-1] = source_local[-1]
    return corrected.astype(np.float32)


def scale_lateral_to_length(candidate_local: np.ndarray, target_len: float) -> np.ndarray:
    """Match path length by scaling lateral shape in the endpoint frame."""

    if len(candidate_local) < 3:
        return candidate_local
    start, _, ex, ey, _ = endpoint_frame(candidate_local)
    rel = candidate_local - start[None, :]
    x = rel @ ex
    y = rel @ ey
    max_abs_y = float(np.max(np.abs(y)))
    if max_abs_y <= 1e-5:
        return candidate_local

    current_len = polyline_length(candidate_local)
    if current_len <= 1e-6:
        return candidate_local
    if current_len >= target_len:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = 1.0, 1.0
        max_lateral = max(180.0, min(900.0, 0.95 * float(np.linalg.norm(candidate_local[-1] - candidate_local[0]))))
        max_scale = max(1.0, max_lateral / max(max_abs_y, 1e-6))
        for _ in range(16):
            trial = from_endpoint_xy(np.stack([x, y * hi], axis=1), start, ex, ey)
            if polyline_length(trial) >= target_len or hi >= max_scale:
                break
            hi = min(max_scale, hi * 1.35)

    best = candidate_local
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        trial = from_endpoint_xy(np.stack([x, y * mid], axis=1), start, ex, ey)
        trial[0] = candidate_local[0]
        trial[-1] = candidate_local[-1]
        trial_len = polyline_length(trial)
        best = trial
        if current_len < target_len:
            if trial_len < target_len:
                lo = mid
            else:
                hi = mid
        else:
            if trial_len > target_len:
                hi = mid
            else:
                lo = mid
    return best.astype(np.float32)


def preserve_progress_contract(
    candidate_local: np.ndarray,
    progress: str,
    direction: str | None = None,
) -> np.ndarray:
    """Keep non-progress edits from silently changing path efficiency.

    OD-chord counterfactuals are endpoint preserving.  A direction template can
    therefore satisfy "turn right" by replacing a multi-turn route with a short
    chord-like arc, which changes the progress token even though progress was
    not edited.  This function adjusts only the path length with a smooth
    one-lobe detour or line blend, keeping endpoints fixed and avoiding
    high-frequency wiggles that would turn the trajectory back into multi-turn.
    """

    if len(candidate_local) < 3:
        return candidate_local
    target_progress = progress_ratio_target(progress)
    if target_progress is None:
        return candidate_local
    line = line_between_endpoints(candidate_local)
    chord_len = float(np.linalg.norm(candidate_local[-1] - candidate_local[0]))
    if not np.isfinite(chord_len) or chord_len <= 1e-6:
        return candidate_local
    target_len = chord_len / max(float(target_progress), 1e-3)
    current_len = polyline_length(candidate_local)
    if current_len <= 1e-6:
        return candidate_local
    current_progress = chord_len / current_len
    if abs(current_progress - float(target_progress)) <= 0.05:
        return candidate_local

    if current_len < target_len:
        if direction in {"left_turn", "right_turn", "u_turn", "multi_turn"}:
            return scale_lateral_to_length(candidate_local, target_len)
        if direction == "straight":
            return candidate_local
        offsets, _ = reference_signed_offsets(candidate_local, line)
        side = float(np.sign(np.mean(offsets)))
        if abs(side) < 1e-6:
            idx = int(np.argmax(np.abs(offsets)))
            side = float(np.sign(offsets[idx]))
        if abs(side) < 1e-6:
            side = 1.0
        max_amp = float(np.clip(0.40 * chord_len, 80.0, 360.0))
        best = candidate_local
        lo, hi = 0.0, max_amp
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            trial = sine_offset(candidate_local, amplitude_m=side * mid, cycles=1.0)
            trial[0] = candidate_local[0]
            trial[-1] = candidate_local[-1]
            trial_len = polyline_length(trial)
            best = trial
            if trial_len < target_len:
                lo = mid
            else:
                hi = mid
        return best.astype(np.float32)

    lo, hi = 0.0, 1.0
    best = candidate_local
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        trial = (1.0 - mid) * candidate_local + mid * line
        trial[0] = candidate_local[0]
        trial[-1] = candidate_local[-1]
        trial_len = polyline_length(trial)
        best = trial
        if trial_len > target_len:
            lo = mid
        else:
            hi = mid
    return best.astype(np.float32)


def endpoint_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    points = np.asarray(points, dtype=np.float32)
    start = points[0].copy()
    end = points[-1].copy()
    chord = end - start
    chord_len = float(np.linalg.norm(chord))
    if chord_len <= 1e-6:
        if len(points) >= 2:
            seed = points[1] - points[0]
            seed_norm = float(np.linalg.norm(seed))
            if seed_norm > 1e-6:
                ex = seed / seed_norm
            else:
                ex = np.asarray([1.0, 0.0], dtype=np.float32)
        else:
            ex = np.asarray([1.0, 0.0], dtype=np.float32)
        chord_len = max(float(np.linalg.norm(points[-1] - points[0])), 1.0)
    else:
        ex = chord / chord_len
    ey = np.asarray([-ex[1], ex[0]], dtype=np.float32)
    return start, end, ex.astype(np.float32), ey.astype(np.float32), float(max(chord_len, 1.0))


def from_endpoint_xy(
    xy: np.ndarray,
    start: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    return start[None, :] + xy[:, [0]] * ex[None, :] + xy[:, [1]] * ey[None, :]


def cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    num_points: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float32)[:, None]
    omt = 1.0 - t
    return (
        omt**3 * p0[None, :]
        + 3.0 * omt**2 * t * p1[None, :]
        + 3.0 * omt * t**2 * p2[None, :]
        + t**3 * p3[None, :]
    ).astype(np.float32)


def endpoint_wave(
    chord_len: float,
    amplitude: float,
    cycles: float,
    num_points: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    x = t * float(chord_len)
    envelope = np.sin(np.pi * t) ** 2
    y = float(amplitude) * envelope * np.sin(float(cycles) * 2.0 * np.pi * t)
    return np.stack([x, y], axis=1).astype(np.float32)


def turn_bezier_xy(chord_len: float, side: float, scale: float, num_points: int) -> np.ndarray:
    p0 = np.asarray([0.0, 0.0], dtype=np.float32)
    p3 = np.asarray([chord_len, 0.0], dtype=np.float32)
    scale = float(np.clip(scale, 0.10, 2.25))
    return cubic_bezier(
        p0,
        np.asarray([0.12 * chord_len, -0.65 * side * scale * chord_len], dtype=np.float32),
        np.asarray([0.78 * chord_len, 1.00 * side * scale * chord_len], dtype=np.float32),
        p3,
        num_points,
    )


def single_turn_arc_xy(chord_len: float, side: float, num_points: int) -> np.ndarray:
    """Endpoint-preserving one-sided arc for OD-chord left/right edits.

    The old direction template used an S-shaped Bezier and then scaled it to a
    progress ratio. Under the OD-chord protocol that made many left/right
    pseudo targets look like multi-turn trajectories. A single broad side arc
    keeps high-frequency turn count low. In this endpoint-preserving local
    frame, a negative middle offset produces a positive final signed turn, so
    caller-side direction semantics must not be inferred from lateral side.
    """

    chord_len = float(max(chord_len, 1.0))
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    x = t * chord_len
    amp = float(np.clip(0.11 * chord_len, 45.0, 145.0))
    envelope = np.sin(np.pi * t).clip(min=0.0) ** 1.15
    y = float(side) * amp * envelope
    return np.stack([x, y], axis=1).astype(np.float32)


def fit_turn_bezier_to_progress(chord_len: float, side: float, progress: str | None, num_points: int) -> np.ndarray:
    target_progress = progress_ratio_target(progress or "")
    if target_progress is None:
        target_progress = 0.67
    target_len = float(chord_len) / max(float(target_progress), 1e-3)
    max_lateral = max(180.0, min(900.0, 0.95 * float(chord_len)))
    max_scale = max(0.10, max_lateral / max(float(chord_len), 1e-6))
    lo, hi = 0.10, min(2.25, max_scale)
    best = turn_bezier_xy(chord_len, side, 1.0, num_points)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        trial = turn_bezier_xy(chord_len, side, mid, num_points)
        trial_len = polyline_length(trial)
        best = trial
        if trial_len < target_len:
            lo = mid
        else:
            hi = mid
    return best.astype(np.float32)


def uturn_polyline_xy(
    chord_len: float,
    amplitude_frac: float,
    backtrack_frac: float,
    num_points: int,
) -> np.ndarray:
    keypoints = np.asarray(
        [
            [0.0, 0.0],
            [0.35 * chord_len, 0.0],
            [0.50 * chord_len, amplitude_frac * chord_len],
            [backtrack_frac * chord_len, amplitude_frac * chord_len],
            [1.10 * chord_len, 0.0],
            [1.00 * chord_len, 0.0],
        ],
        dtype=np.float32,
    )
    return resample_polyline(keypoints, num_points)


def fit_uturn_template(chord_len: float, progress: str | None, num_points: int) -> np.ndarray:
    target_progress = progress_ratio_target(progress or "")
    target_progress = 0.67 if target_progress is None else target_progress
    preset = [
        (0.12, 0.40),
        (0.24, 0.40),
        (0.30, 0.20),
    ]
    if target_progress >= 0.74:
        amp_frac, backtrack_frac = preset[0]
    elif target_progress >= 0.60:
        amp_frac, backtrack_frac = preset[1]
    else:
        amp_frac, backtrack_frac = preset[2]
    return uturn_polyline_xy(
        chord_len,
        amplitude_frac=amp_frac,
        backtrack_frac=backtrack_frac,
        num_points=num_points,
    )


def transition_aware_direction_template(
    local: np.ndarray,
    direction: str,
    progress: str | None = None,
) -> np.ndarray:
    start, _, ex, ey, chord_len = endpoint_frame(local)
    amp_uturn = float(np.clip(0.45 * chord_len, 120.0, 320.0))
    if direction == "straight":
        return from_endpoint_xy(
            np.stack(
                [
                    np.linspace(0.0, chord_len, len(local), dtype=np.float32),
                    np.zeros(len(local), dtype=np.float32),
                ],
                axis=1,
            ),
            start,
            ex,
            ey,
        )
    if direction == "left_turn":
        del progress
        curve = single_turn_arc_xy(chord_len, side=-1.0, num_points=len(local))
        return from_endpoint_xy(curve, start, ex, ey)
    if direction == "right_turn":
        del progress
        curve = single_turn_arc_xy(chord_len, side=1.0, num_points=len(local))
        return from_endpoint_xy(curve, start, ex, ey)
    if direction == "u_turn":
        curve = fit_uturn_template(chord_len, progress=progress, num_points=len(local))
        return from_endpoint_xy(curve, start, ex, ey)
    if direction == "multi_turn":
        target_progress = 0.68
        target_len = float(chord_len) / max(float(target_progress), 1e-3)
        lo, hi = 20.0, max(120.0, 1.20 * chord_len)
        best = endpoint_wave(chord_len, hi, cycles=2.0, num_points=len(local))
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            trial = endpoint_wave(chord_len, mid, cycles=2.0, num_points=len(local))
            best = trial
            if polyline_length(trial) < target_len:
                lo = mid
            else:
                hi = mid
        return from_endpoint_xy(best, start, ex, ey)
    return local


def sine_offset(local: np.ndarray, amplitude_m: float, cycles: float) -> np.ndarray:
    _, normal = tangent_normal(local)
    t = np.linspace(0.0, 1.0, len(local), dtype=np.float32)[:, None]
    envelope = np.sin(np.pi * t).clip(min=0.0)
    wave = np.sin(cycles * np.pi * t)
    return local + float(amplitude_m) * envelope * wave * normal


def apply_direction(local: np.ndarray, direction: str) -> np.ndarray:
    if direction == "straight":
        return 0.85 * line_between_endpoints(local) + 0.15 * local
    if direction == "left_turn":
        return sine_offset(local, amplitude_m=120.0, cycles=1.0)
    if direction == "right_turn":
        return sine_offset(local, amplitude_m=-120.0, cycles=1.0)
    if direction == "u_turn":
        return sine_offset(local, amplitude_m=260.0, cycles=2.0)
    if direction == "multi_turn":
        return sine_offset(local, amplitude_m=100.0, cycles=4.0)
    return local


def apply_transition_aware_direction(
    local: np.ndarray,
    reference_local: np.ndarray,
    direction: str,
    progress: str | None = None,
) -> np.ndarray:
    """Build a cleaner direction counterfactual target from endpoints/reference.

    Direction counterfactuals should not inherit all turns from the source
    trajectory.  For hard transitions such as multi_turn -> straight, using the
    source path as the basis keeps too much local-turn geometry and creates bad
    pseudo targets.  This policy starts from an endpoint-preserving chord with a
    light reference blend, then applies the desired direction shape.
    """
    if len(reference_local) != len(local):
        reference_local = resample_polyline(reference_local, len(local))
    template = transition_aware_direction_template(local, direction, progress=progress)
    line = line_between_endpoints(local)
    if direction == "straight":
        return 0.96 * template + 0.04 * reference_local
    if direction in {"left_turn", "right_turn"}:
        return 0.90 * template + 0.10 * line
    if direction == "u_turn":
        return 0.94 * template + 0.06 * line
    if direction == "multi_turn":
        return 0.90 * template + 0.10 * line
    return local


def fit_curvature_wave(
    chord_len: float,
    cycles: float,
    target_density: float,
    num_points: int,
    max_amplitude: float,
) -> np.ndarray:
    chord_len = float(max(chord_len, 1.0))
    target_density = float(max(target_density, 0.0))
    max_amplitude = float(max(max_amplitude, 1.0))
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    x = t * chord_len
    envelope = np.sin(np.pi * t).clip(min=0.0) ** 1.15

    def template(amplitude: float) -> np.ndarray:
        y = float(amplitude) * envelope * np.sin(float(cycles) * np.pi * t)
        return np.stack([x, y], axis=1).astype(np.float32)

    best = template(max_amplitude)
    lo, hi = 0.0, max_amplitude
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        trial = template(mid)
        best = trial
        if local_turn_density_score(trial) < target_density:
            lo = mid
        else:
            hi = mid
    return best.astype(np.float32)


def apply_curvature(local: np.ndarray, curvature: str) -> np.ndarray:
    if curvature == "smooth":
        line = line_between_endpoints(local)
        smoothed = 0.90 * line + 0.10 * local
        smoothed[0] = local[0]
        smoothed[-1] = local[-1]
        return smoothed.astype(np.float32)
    start, _, ex, ey, chord_len = endpoint_frame(local)
    if curvature == "moderate":
        curve = fit_curvature_wave(
            chord_len,
            cycles=1.0,
            target_density=0.82,
            num_points=len(local),
            max_amplitude=float(np.clip(0.22 * chord_len, 120.0, 260.0)),
        )
        return from_endpoint_xy(curve, start, ex, ey)
    if curvature == "winding":
        curve = fit_curvature_wave(
            chord_len,
            cycles=3.0,
            target_density=1.35,
            num_points=len(local),
            max_amplitude=float(np.clip(0.28 * chord_len, 160.0, 360.0)),
        )
        return from_endpoint_xy(curve, start, ex, ey)
    return local


def apply_reference(local: np.ndarray, reference_local: np.ndarray, adherence: str) -> np.ndarray:
    if len(reference_local) != len(local):
        reference_local = resample_polyline(reference_local, len(local))
    if adherence == "tight":
        return 0.80 * reference_local + 0.20 * local
    if adherence == "moderate":
        return sine_offset(0.65 * reference_local + 0.35 * local, amplitude_m=45.0, cycles=1.0)
    if adherence == "loose":
        return sine_offset(0.40 * reference_local + 0.60 * local, amplitude_m=120.0, cycles=1.5)
    if adherence == "unknown":
        return 0.50 * reference_local + 0.50 * local
    return local


def apply_progress(local: np.ndarray, progress: str, trip_attrs: Dict | None = None) -> np.ndarray:
    if len(local) < 3:
        return local
    duration_min = 0.0
    if isinstance(trip_attrs, dict):
        duration_min = float(trip_attrs.get("trip_duration_min", 0.0) or 0.0)
    target_speed = progress_speed_target_kmh(progress, trip_attrs)
    if target_speed is None or duration_min <= 1e-6:
        return local
    target_len = float(target_speed) * (duration_min / 60.0) * 1000.0
    chord_len = float(np.linalg.norm(local[-1] - local[0]))
    target_len = max(target_len, 1.02 * chord_len)
    current_len = polyline_length(local)
    if current_len <= 1e-6 or abs(current_len - target_len) <= 80.0:
        return local
    if current_len > target_len:
        line = line_between_endpoints(local)
        lo, hi = 0.0, 1.0
        best = local
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            trial = (1.0 - mid) * local + mid * line
            trial[0] = local[0]
            trial[-1] = local[-1]
            best = trial
            if polyline_length(trial) > target_len:
                lo = mid
            else:
                hi = mid
        return best.astype(np.float32)
    return scale_lateral_to_length(local, target_len)


def apply_distance(local: np.ndarray, distance: str) -> np.ndarray:
    if len(local) < 3:
        return local
    line = line_between_endpoints(local)
    chord_len = float(np.linalg.norm(local[-1] - local[0]))
    current_len = polyline_length(local)
    target_len = distance_length_target_m(distance, source_len=current_len, chord_len=chord_len)
    if target_len is None or current_len <= 1e-6:
        return local
    if abs(current_len - target_len) <= 80.0:
        return local
    if current_len > target_len:
        lo, hi = 0.0, 1.0
        best = local
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            trial = (1.0 - mid) * local + mid * line
            trial[0] = local[0]
            trial[-1] = local[-1]
            best = trial
            if polyline_length(trial) > target_len:
                lo = mid
            else:
                hi = mid
        return best.astype(np.float32)

    offsets, _ = reference_signed_offsets(local, line)
    side = float(np.sign(np.mean(offsets)))
    if abs(side) < 1e-6:
        idx = int(np.argmax(np.abs(offsets)))
        side = float(np.sign(offsets[idx]))
    if abs(side) < 1e-6:
        side = 1.0
    max_amp = float(np.clip(0.75 * max(chord_len, target_len), 120.0, 1800.0))
    lo, hi = 0.0, max_amp
    best = local
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        trial = sine_offset(local, amplitude_m=side * mid, cycles=1.0)
        trial[0] = local[0]
        trial[-1] = local[-1]
        best = trial
        if polyline_length(trial) < target_len:
            lo = mid
        else:
            hi = mid
    return best.astype(np.float32)


def resample_polyline(points: np.ndarray, target_len: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == target_len:
        return points
    if len(points) <= 1:
        return np.repeat(points[:1], target_len, axis=0)
    step = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(np.clip(step, 1e-8, None))], axis=0)
    if cumulative[-1] <= 1e-8:
        return np.repeat(points[:1], target_len, axis=0)
    targets = np.linspace(0.0, cumulative[-1], target_len, endpoint=True)
    x = np.interp(targets, cumulative, points[:, 0])
    y = np.interp(targets, cumulative, points[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def build_pseudo_trajectory(
    record: Dict,
    labels: Dict[str, str],
    changed_field: str | None = None,
    direction_target_policy: str = "legacy",
    counterfactual_geometry_policy: str = "field_isolated",
) -> List[List[float]]:
    traj = np.asarray(record["trajectory"], dtype=np.float32)
    protocol = record.get("reference_protocol")
    guard = record.get("leakage_guard", {})
    if protocol != "od_chord_only":
        raise ValueError(
            "Counterfactual pseudo targets require records generated by the "
            f"official OD-chord protocol; got reference_protocol={protocol!r} "
            f"for sample_id={record.get('sample_id')!r}."
        )
    if not isinstance(guard, dict) or guard.get("reference_uses_gt_middle_geometry") is not False:
        raise ValueError(
            "Counterfactual pseudo targets require an explicit leakage guard "
            f"for sample_id={record.get('sample_id')!r}."
        )
    if "reference" not in record or not record["reference"]:
        raise ValueError(
            "Official OD-chord records must contain an independent reference; "
            f"missing reference for sample_id={record.get('sample_id')!r}."
        )
    ref = np.asarray(record["reference"], dtype=np.float32)
    local, origin, scale = to_local(traj)
    ref_local = to_shared_local(ref if len(ref) else traj, origin, scale)
    ref_local = resample_polyline(ref_local, len(local))

    def apply_field(pseudo: np.ndarray, field: str) -> np.ndarray:
        if field == "distance":
            return apply_distance(pseudo, labels.get("distance", "medium"))
        if field == "reference_adherence":
            return apply_reference(pseudo, ref_local, labels.get("reference_adherence", "tight"))
        if field == "curvature":
            return apply_curvature(pseudo, labels.get("curvature", "moderate"))
        if field == "progress":
            return apply_progress(pseudo, labels.get("progress", "normal"), record.get("trip_attributes", {}))
        if field == "direction":
            direction = labels.get("direction", "multi_turn")
            if changed_field == "direction" and direction_target_policy == "transition_aware":
                return apply_transition_aware_direction(
                    local,
                    ref_local,
                    direction,
                    progress=labels.get("progress", "normal"),
                )
            return apply_direction(pseudo, direction)
        return pseudo

    transform_order = ["distance", "reference_adherence", "curvature", "progress", "direction"]
    if counterfactual_geometry_policy == "field_isolated" and changed_field in transform_order:
        transform_order = [changed_field]
    elif counterfactual_geometry_policy == "changed_field_last" and changed_field in transform_order:
        transform_order = [field for field in transform_order if field != changed_field] + [changed_field]
    elif counterfactual_geometry_policy != "composed":
        raise ValueError(f"Unsupported counterfactual_geometry_policy={counterfactual_geometry_policy!r}")

    pseudo = local.copy()
    for field in transform_order:
        pseudo = apply_field(pseudo, field)
    if changed_field == "direction":
        # Progress is timestamp pace in porto_od_chord_v1, not a spatial
        # efficiency target. Keep direction edits from leaking too much
        # reference-adherence change, but do not stretch them to match the
        # source path-length ratio.
        target_direction = labels.get("direction", "multi_turn")
        ref_strength = 0.0 if target_direction == "straight" else 0.15
        pseudo = preserve_reference_distance(local, pseudo, ref_local, strength=ref_strength)
    elif changed_field == "curvature":
        pseudo = preserve_progress_contract(pseudo, labels.get("progress", "normal"))
        pseudo = preserve_reference_distance(local, pseudo, ref_local, strength=0.35)
    elif changed_field == "progress":
        pseudo = preserve_reference_distance(local, pseudo, ref_local, strength=1.0)
    elif changed_field == "distance":
        pseudo = preserve_reference_distance(local, pseudo, ref_local, strength=0.20)
    pseudo[0] = local[0]
    pseudo[-1] = local[-1]
    return from_local(pseudo, origin, scale).astype(np.float32).tolist()


def choose_values(field: str, current: str, max_variants: int, rng: random.Random) -> List[str]:
    # ``unknown`` is a missing/weak-context fallback, not an actionable edit target.
    values = [value for value in LABEL_VOCAB[field] if value != current and value != "unknown"]
    if field == "direction":
        # OD-chord counterfactuals preserve endpoints, so a real U-turn target is
        # not a clean local edit: it usually requires endpoint/backtracking
        # semantics that are absent from the chord condition. Keep U-turn as a
        # source label, but do not synthesize it as an official Table-4 target.
        values = [value for value in values if value != "u_turn"]
    if field in {"distance", "progress"}:
        order = list(LABEL_VOCAB[field])
        if current in order:
            idx = order.index(current)
            adjacent = [
                order[j]
                for j in (idx - 1, idx + 1)
                if 0 <= j < len(order) and order[j] != current
            ]
            if adjacent:
                values = adjacent
    if max_variants <= 0:
        return values
    rng.shuffle(values)
    return values[:max_variants]


def source_length_and_chord_m(record: Dict) -> tuple[float, float]:
    traj = np.asarray(record["trajectory"], dtype=np.float32)
    local, _, _ = to_local(traj)
    source_len = polyline_length(local)
    chord_len = float(np.linalg.norm(local[-1] - local[0])) if len(local) >= 2 else 0.0
    return float(source_len), float(chord_len)


def feasible_scalar_counterfactual(record: Dict, field: str, value: str) -> bool:
    """Drop scalar pseudo targets that cannot satisfy OD-preserving quality gates."""

    if field not in {"distance", "progress"}:
        return True
    source_len, chord_len = source_length_and_chord_m(record)
    if source_len <= 1e-6:
        return False
    target_len: float | None
    if field == "distance":
        target_len = distance_length_target_m(value, source_len=source_len, chord_len=chord_len)
    else:
        trip_attrs = record.get("trip_attributes", {})
        duration_min = float(trip_attrs.get("trip_duration_min", 0.0) or 0.0) if isinstance(trip_attrs, dict) else 0.0
        speed = progress_speed_target_kmh(value, trip_attrs if isinstance(trip_attrs, dict) else {})
        target_len = None if speed is None or duration_min <= 1e-6 else float(speed) * (duration_min / 60.0) * 1000.0
        if target_len is not None:
            target_len = max(target_len, 1.02 * max(chord_len, 1.0))
    if target_len is None:
        return False
    ratio = float(target_len) / max(source_len, 1e-6)
    return 0.60 <= ratio <= 1.80


def make_counterfactual_record(
    record: Dict,
    source_labels: Dict[str, str],
    labels: Dict[str, str],
    field: str,
    value: str,
    balanced_fill: bool = False,
    direction_target_policy: str = "legacy",
    counterfactual_geometry_policy: str = "field_isolated",
) -> Dict:
    source_value = source_labels.get(field)
    if value == source_value:
        raise ValueError(f"Counterfactual no-op for field '{field}': {source_value!r} -> {value!r}")
    pseudo = build_pseudo_trajectory(
        record,
        labels,
        changed_field=field,
        direction_target_policy=direction_target_policy,
        counterfactual_geometry_policy=counterfactual_geometry_policy,
    )
    counterfactual = {
        "enabled": True,
        "source_sample_id": int(record["sample_id"]),
        "field": field,
        "from": source_value,
        "to": value,
    }
    if balanced_fill:
        counterfactual["balanced_fill"] = True
    if field == "direction":
        counterfactual["direction_target_policy"] = direction_target_policy
    counterfactual["geometry_policy"] = counterfactual_geometry_policy
    return {
        **record,
        "instruction": build_instruction(labels),
        "labels": labels,
        "mobility_tokens": DEFAULT_MOBILITY_VOCAB.encode(labels),
        "pseudo_trajectory": pseudo,
        "counterfactual": counterfactual,
    }


def make_counterfactual_records(
    record: Dict,
    fields: Iterable[str],
    variants_per_field: int,
    rng: random.Random,
    direction_target_policy: str = "legacy",
    counterfactual_geometry_policy: str = "field_isolated",
) -> List[Dict]:
    base_labels = dict(record["labels"])
    outputs = []
    for field in fields:
        for value in choose_values(field, base_labels.get(field, ""), variants_per_field, rng):
            if not feasible_scalar_counterfactual(record, field, value):
                continue
            labels = dict(base_labels)
            labels[field] = value
            outputs.append(
                make_counterfactual_record(
                    record,
                    base_labels,
                    labels,
                    field,
                    value,
                    direction_target_policy=direction_target_policy,
                    counterfactual_geometry_policy=counterfactual_geometry_policy,
                )
            )
    return outputs


def summarize(records: List[Dict]) -> Dict:
    counts = defaultdict(Counter)
    cf_counts = Counter()
    cf_transition_counts = Counter()
    no_op_counts = Counter()
    for record in records:
        for field, value in record["labels"].items():
            counts[field][value] += 1
        cf = record.get("counterfactual", {})
        if cf.get("enabled"):
            cf_counts[f"{cf.get('field')}:{cf.get('to')}"] += 1
            transition_key = f"{cf.get('field')}:{cf.get('from')}->{cf.get('to')}"
            cf_transition_counts[transition_key] += 1
            if cf.get("from") == cf.get("to"):
                no_op_counts[transition_key] += 1
    return {
        "num_records": len(records),
        "label_counts": {
            field: {str(k): int(v) for k, v in sorted(counter.items())}
            for field, counter in counts.items()
        },
        "counterfactual_counts": {str(k): int(v) for k, v in sorted(cf_counts.items())},
        "counterfactual_transition_counts": {str(k): int(v) for k, v in sorted(cf_transition_counts.items())},
        "counterfactual_no_op_counts": {str(k): int(v) for k, v in sorted(no_op_counts.items())},
    }


def update_summary_counters(
    record: Dict,
    counts: Dict[str, Counter],
    cf_counts: Counter,
    cf_transition_counts: Counter,
    no_op_counts: Counter,
) -> None:
    for field, value in record["labels"].items():
        counts[field][value] += 1
    cf = record.get("counterfactual", {})
    if cf.get("enabled"):
        cf_counts[f"{cf.get('field')}:{cf.get('to')}"] += 1
        transition_key = f"{cf.get('field')}:{cf.get('from')}->{cf.get('to')}"
        cf_transition_counts[transition_key] += 1
        if cf.get("from") == cf.get("to"):
            no_op_counts[transition_key] += 1


def summary_from_counters(
    num_records: int,
    counts: Dict[str, Counter],
    cf_counts: Counter,
    cf_transition_counts: Counter,
    no_op_counts: Counter,
) -> Dict:
    return {
        "num_records": int(num_records),
        "label_counts": {
            field: {str(k): int(v) for k, v in sorted(counter.items())}
            for field, counter in counts.items()
        },
        "counterfactual_counts": {str(k): int(v) for k, v in sorted(cf_counts.items())},
        "counterfactual_transition_counts": {str(k): int(v) for k, v in sorted(cf_transition_counts.items())},
        "counterfactual_no_op_counts": {str(k): int(v) for k, v in sorted(no_op_counts.items())},
    }


def split_output_path(path: Path, split: str) -> Path:
    return path.with_name(f"{path.stem}.{split}{path.suffix}")


def validate_counterfactual_records(records: List[Dict]) -> None:
    bad = []
    for idx, record in enumerate(records):
        cf = record.get("counterfactual", {})
        if not cf.get("enabled"):
            continue
        field = str(cf.get("field"))
        source_value = cf.get("from")
        target_value = cf.get("to")
        label_value = record.get("labels", {}).get(field)
        if source_value == target_value or target_value != label_value:
            bad.append(
                {
                    "row": idx,
                    "sample_id": record.get("sample_id"),
                    "field": field,
                    "from": source_value,
                    "to": target_value,
                    "label": label_value,
                }
            )
    if bad:
        preview = json.dumps(bad[:5], ensure_ascii=False)
        raise ValueError(f"Invalid counterfactual records found: {len(bad)}. Preview: {preview}")


def build_dataset(args: argparse.Namespace) -> Dict:
    rng = random.Random(args.seed)
    input_path = args.input or (REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_instruction_samples.jsonl")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_records = read_jsonl(input_path, max_samples=args.max_samples)
    fields = [field for field in args.fields if field in DEFAULT_MOBILITY_VOCAB.fields]
    split_payload = load_split_manifest(args.split_manifest) if args.split_manifest else None
    split_by_id = split_payload["sample_splits"] if split_payload else None

    records = []
    for record in base_records:
        if record.get("reference_protocol") != "od_chord_only":
            raise ValueError(
                "Input JSONL is not generated by the official OD-chord protocol. "
                f"First bad sample_id={record.get('sample_id')!r}; "
                f"reference_protocol={record.get('reference_protocol')!r}."
            )
        base = dict(record)
        base["mobility_tokens"] = DEFAULT_MOBILITY_VOCAB.encode(base["labels"])
        base["pseudo_trajectory"] = base["trajectory"]
        base["counterfactual"] = {
            "enabled": False,
            "source_sample_id": int(base["sample_id"]),
            "field": None,
            "from": None,
            "to": None,
        }
        if args.include_original:
            records.append(base)
        records.extend(
            make_counterfactual_records(
                base,
                fields,
                args.variants_per_field,
                rng,
                direction_target_policy=args.direction_target_policy,
                counterfactual_geometry_policy=args.counterfactual_geometry_policy,
            )
        )

    if args.balance_fields:
        for field in fields:
            current_counts = Counter(record["labels"].get(field) for record in records)
            target_count = int(max(current_counts.values()) if current_counts else 0)
            if args.balance_target > 0:
                target_count = int(args.balance_target)
            if target_count <= 0:
                continue
            candidates = []
            for base in base_records:
                base_labels = dict(base["labels"])
                for value in choose_values(field, base_labels.get(field, ""), max_variants=0, rng=rng):
                    if not feasible_scalar_counterfactual(base, field, value):
                        continue
                    labels = dict(base_labels)
                    labels[field] = value
                    candidates.append((base, base_labels, labels, value))
            rng.shuffle(candidates)
            for base, source_labels, labels, value in candidates:
                if current_counts[value] >= target_count:
                    continue
                records.append(
                    make_counterfactual_record(
                        base,
                        source_labels,
                        labels,
                        field,
                        value,
                        balanced_fill=True,
                        direction_target_policy=args.direction_target_policy,
                        counterfactual_geometry_policy=args.counterfactual_geometry_policy,
                    )
                )
                current_counts[value] += 1
                if all(current_counts[value_] >= target_count for value_ in LABEL_VOCAB[field]):
                    break

    if args.shuffle:
        rng.shuffle(records)
    validate_counterfactual_records(records)
    output_path = output_dir / f"{args.dataset}_counterfactual_instruction_samples.jsonl"
    summary_path = output_dir / f"{args.dataset}_counterfactual_instruction_summary.json"
    split_paths = {split: split_output_path(output_path, split) for split in SPLIT_ORDER}
    split_handles = (
        {split: split_paths[split].open("w", encoding="utf-8") for split in SPLIT_ORDER}
        if split_by_id is not None
        else {}
    )
    split_records_for_summary: list[Dict] = []
    with output_path.open("w", encoding="utf-8") as f:
        try:
            for i, record in enumerate(records):
                record = dict(record)
                record["counterfactual_id"] = int(i)
                if split_by_id is not None:
                    split = record_split(record, split_by_id)
                    record["split"] = split
                    split_handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    split_records_for_summary.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            for handle in split_handles.values():
                handle.close()
    summary = summarize(records)
    if split_by_id is not None:
        summary["split"] = {
            "manifest_path": str(args.split_manifest),
            "split_unit": "base_sample_id",
            "counterfactual_variants_follow_source_sample": True,
            "split_jsonl": {split: str(split_paths[split]) for split in SPLIT_ORDER},
            **summarize_split_records(split_records_for_summary),
        }
    summary.update(
        {
            "dataset": args.dataset,
            "input": str(input_path),
            "output_jsonl": str(output_path),
            "fields": fields,
            "variants_per_field": int(args.variants_per_field),
            "include_original": bool(args.include_original),
            "balance_fields": bool(args.balance_fields),
            "balance_target": int(args.balance_target),
            "target_for_control_training": "pseudo_trajectory",
            "direction_target_policy": str(args.direction_target_policy),
            "counterfactual_geometry_policy": str(args.counterfactual_geometry_policy),
            "pseudo_target_transform_order": (
                [args.counterfactual_geometry_policy]
                if args.counterfactual_geometry_policy == "field_isolated"
                else [
                    "distance",
                    "reference_adherence",
                    "curvature",
                    "progress",
                    "direction",
                ]
            ),
            "pseudo_target_primary_field_policy": (
                "only_changed_field_applied"
                if args.counterfactual_geometry_policy == "field_isolated"
                else "changed_field_applied_last"
            ),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_dataset_streaming(args: argparse.Namespace) -> Dict:
    if args.balance_fields:
        raise ValueError("--streaming-write does not support --balance-fields; use deterministic split files instead.")
    if args.shuffle:
        raise ValueError("--streaming-write does not support --shuffle because it writes records incrementally.")
    rng = random.Random(args.seed)
    input_path = args.input or (REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_instruction_samples.jsonl")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [field for field in args.fields if field in DEFAULT_MOBILITY_VOCAB.fields]
    split_payload = load_split_manifest(args.split_manifest) if args.split_manifest else None
    split_by_id = split_payload["sample_splits"] if split_payload else None
    if args.split_only_output and split_by_id is None:
        raise ValueError("--split-only-output requires --split-manifest.")
    output_path = output_dir / f"{args.dataset}_counterfactual_instruction_samples.jsonl"
    summary_path = output_dir / f"{args.dataset}_counterfactual_instruction_summary.json"
    split_paths = {split: split_output_path(output_path, split) for split in SPLIT_ORDER}
    split_handles = (
        {split: split_paths[split].open("w", encoding="utf-8") for split in SPLIT_ORDER}
        if split_by_id is not None
        else {}
    )
    counts: Dict[str, Counter] = defaultdict(Counter)
    cf_counts = Counter()
    cf_transition_counts = Counter()
    no_op_counts = Counter()
    split_counts = Counter()
    split_label_counts: dict[str, Counter] = defaultdict(Counter)
    split_cf_transition_counts = Counter()
    record_count = 0
    base_count = 0
    base_total = None
    if split_by_id is not None:
        base_total = len(split_by_id)
    if args.max_samples > 0:
        base_total = min(int(args.max_samples), int(base_total)) if base_total is not None else int(args.max_samples)
    out = None if args.split_only_output else output_path.open("w", encoding="utf-8")
    try:
        for base_record in iter_instruction_records(
            input_path,
            max_samples=args.max_samples,
            progress_desc="build counterfactual base records",
            total=base_total,
        ):
            if base_record.get("reference_protocol") != "od_chord_only":
                raise ValueError(
                    "Input JSONL is not generated by the official OD-chord protocol. "
                    f"First bad sample_id={base_record.get('sample_id')!r}; "
                    f"reference_protocol={base_record.get('reference_protocol')!r}."
                )
            base_count += 1
            base = dict(base_record)
            base["mobility_tokens"] = DEFAULT_MOBILITY_VOCAB.encode(base["labels"])
            base["pseudo_trajectory"] = base["trajectory"]
            base["counterfactual"] = {
                "enabled": False,
                "source_sample_id": int(base["sample_id"]),
                "field": None,
                "from": None,
                "to": None,
            }
            batch = []
            if args.include_original:
                batch.append(base)
            batch.extend(
                make_counterfactual_records(
                    base,
                    fields,
                    args.variants_per_field,
                    rng,
                    direction_target_policy=args.direction_target_policy,
                    counterfactual_geometry_policy=args.counterfactual_geometry_policy,
                )
            )
            validate_counterfactual_records(batch)
            for record in batch:
                record = dict(record)
                record["counterfactual_id"] = int(record_count)
                if split_by_id is not None:
                    split = record_split(record, split_by_id)
                    record["split"] = split
                    split_handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    split_counts[split] += 1
                    for field, value in record.get("labels", {}).items():
                        split_label_counts[f"{split}:{field}"][str(value)] += 1
                    cf = record.get("counterfactual", {})
                    if cf.get("enabled"):
                        split_cf_transition_counts[
                            f"{split}:{cf.get('field')}:{cf.get('from')}->{cf.get('to')}"
                        ] += 1
                if out is not None:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                update_summary_counters(record, counts, cf_counts, cf_transition_counts, no_op_counts)
                record_count += 1
    finally:
        if out is not None:
            out.close()
        for handle in split_handles.values():
            handle.close()
    summary = summary_from_counters(record_count, counts, cf_counts, cf_transition_counts, no_op_counts)
    if split_by_id is not None:
        summary["split"] = {
            "manifest_path": str(args.split_manifest),
            "split_unit": "base_sample_id",
            "counterfactual_variants_follow_source_sample": True,
            "split_jsonl": {split: str(split_paths[split]) for split in SPLIT_ORDER},
            "record_counts": {split: int(split_counts.get(split, 0)) for split in SPLIT_ORDER},
            "label_counts": {
                key: dict(sorted(counter.items()))
                for key, counter in sorted(split_label_counts.items())
            },
            "counterfactual_transition_counts": dict(sorted(split_cf_transition_counts.items())),
        }
    summary.update(
        {
            "dataset": args.dataset,
            "input": str(input_path),
            "output_jsonl": None if args.split_only_output else str(output_path),
            "fields": fields,
            "base_records": int(base_count),
            "variants_per_field": int(args.variants_per_field),
            "include_original": bool(args.include_original),
            "balance_fields": bool(args.balance_fields),
            "balance_target": int(args.balance_target),
            "target_for_control_training": "pseudo_trajectory",
            "direction_target_policy": str(args.direction_target_policy),
            "counterfactual_geometry_policy": str(args.counterfactual_geometry_policy),
            "pseudo_target_transform_order": (
                [args.counterfactual_geometry_policy]
                if args.counterfactual_geometry_policy == "field_isolated"
                else [
                    "distance",
                    "reference_adherence",
                    "curvature",
                    "progress",
                    "direction",
                ]
            ),
            "pseudo_target_primary_field_policy": (
                "only_changed_field_applied"
                if args.counterfactual_geometry_policy == "field_isolated"
                else "changed_field_applied_last"
            ),
            "streaming_write": True,
            "split_only_output": bool(args.split_only_output),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced counterfactual instruction-control dataset.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "instruction_data")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["distance", "direction", "curvature", "progress", "reference_adherence"],
    )
    parser.add_argument("--variants-per-field", type=int, default=2)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument("--balance-fields", action="store_true")
    parser.add_argument("--balance-target", type=int, default=0)
    parser.add_argument(
        "--direction-target-policy",
        choices=["legacy", "transition_aware"],
        default="legacy",
        help="Use a cleaner endpoint/reference-based basis for direction counterfactual pseudo targets.",
    )
    parser.add_argument(
        "--counterfactual-geometry-policy",
        choices=["field_isolated", "changed_field_last", "composed"],
        default="field_isolated",
        help=(
            "Pseudo-target composition policy. 'field_isolated' is the official controllability "
            "setting: only the changed field modifies geometry, keeping non-target fields stable "
            "for Table 4 collateral evaluation."
        ),
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional base-sample split manifest. Counterfactual variants follow source_sample_id.",
    )
    parser.add_argument(
        "--streaming-write",
        action="store_true",
        help="Write counterfactual JSONL incrementally instead of materializing all records in memory.",
    )
    parser.add_argument(
        "--split-only-output",
        action="store_true",
        help="With --split-manifest, write only train/val/test JSONL files and skip the duplicate unsplit JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.streaming_write:
        print(json.dumps(build_dataset_streaming(args), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(build_dataset(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
