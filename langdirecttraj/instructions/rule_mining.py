from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .geometry_contract import contract_from_lonlat, drop_duplicate_points, valid_trajectory
from .schema import InstructionRecord
from .templates import build_instruction


def _to_lonlat_traj(traj: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected trajectory with 2 dims, got shape {arr.shape}")
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    if arr.shape[-1] < 2:
        raise ValueError(f"Trajectory needs xy/lonlat channels, got shape {arr.shape}")
    lon = arr[:, 0] * float(stats["lon_std"]) + float(stats["lon_mean"])
    lat = arr[:, 1] * float(stats["lat_std"]) + float(stats["lat_mean"])
    return np.stack([lon, lat], axis=1).astype(np.float32)


def _to_lonlat_reference(road: Optional[np.ndarray], stats: Dict[str, float]) -> np.ndarray:
    if road is None:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.asarray(road, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] < 4:
        return np.empty((0, 2), dtype=np.float32)
    starts = np.stack(
        [
            arr[:, 0] * float(stats["lon_std"]) + float(stats["lon_mean"]),
            arr[:, 1] * float(stats["lat_std"]) + float(stats["lat_mean"]),
        ],
        axis=1,
    )
    end = np.array(
        [
            arr[-1, 2] * float(stats["lon_std"]) + float(stats["lon_mean"]),
            arr[-1, 3] * float(stats["lat_std"]) + float(stats["lat_mean"]),
        ],
        dtype=np.float32,
    )
    poly = np.concatenate([starts.astype(np.float32), end[None, :]], axis=0)
    return drop_duplicate_points(poly)


def mine_instruction_record(
    sample_id: int,
    dataset: str,
    traj: np.ndarray,
    road: Optional[np.ndarray],
    stats: Dict[str, float],
    include_coordinates: bool = True,
) -> InstructionRecord:
    traj_lonlat = valid_trajectory(_to_lonlat_traj(traj, stats))
    ref_lonlat = _to_lonlat_reference(road, stats)
    contract = contract_from_lonlat(traj_lonlat, ref_lonlat, dataset)
    labels = contract.labels
    metrics = contract.metrics
    instruction = build_instruction(labels)

    trajectory = traj_lonlat.tolist() if include_coordinates else []
    reference = ref_lonlat.tolist() if include_coordinates else []
    return InstructionRecord(
        sample_id=int(sample_id),
        dataset=str(dataset),
        instruction=instruction,
        labels=labels,
        metrics=metrics,
        trajectory=trajectory,
        reference=reference,
    )
