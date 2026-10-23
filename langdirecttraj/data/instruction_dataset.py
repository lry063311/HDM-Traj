from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object

from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB, MobilityTokenVocab
from langdirecttraj.data.compact_store import (
    expand_compact_record,
    is_compact_meta_path,
    load_compact_manifest,
    open_compact_arrays,
)


OFFICIAL_REFERENCE_PROTOCOL = "od_chord_only"
OFFICIAL_CONDITION_SOURCE = "origin_destination_chord"
OFFICIAL_TARGET_SOURCE = "raw_gt_trajectory"
FAST_PATH_EMPTY_TRAJ = None


def _fast_path_empty_traj() -> torch.Tensor:
    global FAST_PATH_EMPTY_TRAJ
    if FAST_PATH_EMPTY_TRAJ is None:
        if torch is None:
            raise RuntimeError("Fast-path trajectory placeholder requires PyTorch.")
        FAST_PATH_EMPTY_TRAJ = torch.zeros(1, 2, dtype=torch.float32)
    return FAST_PATH_EMPTY_TRAJ


class LazyRecordSequence:
    def __init__(self, dataset: "InstructionTrajectoryDataset"):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[Dict]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[idx] for idx in range(*index.indices(len(self)))]
        return self.dataset.get_record(int(index))


def validate_official_record_protocol(record: Dict, row_index: int) -> None:
    sample_id = record.get("sample_id", row_index)
    protocol = record.get("reference_protocol")
    condition_source = record.get("condition_source")
    target_source = record.get("target_source")
    guard = record.get("leakage_guard")
    if protocol != OFFICIAL_REFERENCE_PROTOCOL:
        raise ValueError(
            "InstructionTrajectoryDataset only accepts the official OD-chord protocol. "
            f"row={row_index}, sample_id={sample_id!r}, reference_protocol={protocol!r}."
        )
    if condition_source != OFFICIAL_CONDITION_SOURCE:
        raise ValueError(
            "InstructionTrajectoryDataset requires OD chord as the only condition source. "
            f"row={row_index}, sample_id={sample_id!r}, condition_source={condition_source!r}."
        )
    if target_source != OFFICIAL_TARGET_SOURCE:
        raise ValueError(
            "InstructionTrajectoryDataset requires the raw GT trajectory to be target-only. "
            f"row={row_index}, sample_id={sample_id!r}, target_source={target_source!r}."
        )
    if not isinstance(guard, dict):
        raise ValueError(
            "InstructionTrajectoryDataset requires an explicit leakage_guard. "
            f"row={row_index}, sample_id={sample_id!r}."
        )
    if guard.get("reference_uses_gt_middle_geometry") is not False:
        raise ValueError(
            "Protocol violation: reference must not use GT middle geometry. "
            f"row={row_index}, sample_id={sample_id!r}."
        )
    if guard.get("reference_uses_only_origin_destination") is not True:
        raise ValueError(
            "Protocol violation: reference must use only origin/destination endpoints. "
            f"row={row_index}, sample_id={sample_id!r}."
        )


class InstructionTrajectoryDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        vocab: MobilityTokenVocab = DEFAULT_MOBILITY_VOCAB,
        max_samples: Optional[int] = None,
        include_reference: bool = True,
        target_key: str = "trajectory",
        require_official_protocol: bool = True,
        counterfactual_filter: str | None = None,
        counterfactual_source_filter: str | None = None,
        qwen_plan_cache_path: str | Path | None = None,
        geometry_cache_dir: str | Path | None = None,
        fast_meta_cache_dir: str | Path | None = None,
        static_metric_cache_dir: str | Path | None = None,
        road_corridor_cache_path: str | Path | None = None,
    ):
        if torch is None:
            raise RuntimeError("InstructionTrajectoryDataset requires PyTorch.")
        self.jsonl_path = Path(jsonl_path)
        self.vocab = vocab
        self.include_reference = bool(include_reference)
        self.target_key = str(target_key)
        self.require_official_protocol = bool(require_official_protocol)
        self.counterfactual_filter = str(counterfactual_filter or "").strip()
        self.counterfactual_source_filter = str(counterfactual_source_filter or "").strip()
        self.qwen_plan_cache_path = Path(qwen_plan_cache_path) if qwen_plan_cache_path else None
        self.qwen_plan_cache = (
            np.load(self.qwen_plan_cache_path, mmap_mode="r")
            if self.qwen_plan_cache_path is not None
            else None
        )
        self.geometry_cache_dir = Path(geometry_cache_dir) if geometry_cache_dir else None
        self.geometry_cache = {}
        if self.geometry_cache_dir is not None:
            self.geometry_cache = {
                "target_norm": np.load(self.geometry_cache_dir / "target_norm.npy", mmap_mode="r"),
                "reference_norm": np.load(self.geometry_cache_dir / "reference_norm.npy", mmap_mode="r"),
                "route_scale": np.load(self.geometry_cache_dir / "route_scale.npy", mmap_mode="r"),
            }
        self.fast_meta_cache_dir = Path(fast_meta_cache_dir) if fast_meta_cache_dir else None
        self.fast_meta_cache = {}
        self.fast_meta_manifest = {}
        if self.fast_meta_cache_dir is not None:
            if self.geometry_cache_dir is None:
                raise ValueError("--fast-meta-cache-dir requires --geometry-cache-dir.")
            manifest_path = self.fast_meta_cache_dir / "manifest.json"
            self.fast_meta_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.fast_meta_cache = {
                "sample_id": np.load(self.fast_meta_cache_dir / "sample_id.npy", mmap_mode="r"),
                "mobility_tokens": np.load(self.fast_meta_cache_dir / "mobility_tokens.npy", mmap_mode="r"),
                "trip_duration_min": np.load(self.fast_meta_cache_dir / "trip_duration_min.npy", mmap_mode="r"),
                "counterfactual_enabled": np.load(
                    self.fast_meta_cache_dir / "counterfactual_enabled.npy",
                    mmap_mode="r",
                ),
                "counterfactual_field_id": np.load(
                    self.fast_meta_cache_dir / "counterfactual_field_id.npy",
                    mmap_mode="r",
                ),
            }
        self.static_metric_cache_dir = Path(static_metric_cache_dir) if static_metric_cache_dir else None
        self.static_metric_cache = {}
        self.static_metric_manifest = {}
        if self.static_metric_cache_dir is not None:
            manifest_path = self.static_metric_cache_dir / "manifest.json"
            self.static_metric_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.static_metric_cache = {
                "trip_duration_min": np.load(self.static_metric_cache_dir / "trip_duration_min.npy", mmap_mode="r"),
                "base_token_scores": np.load(self.static_metric_cache_dir / "base_token_scores.npy", mmap_mode="r"),
                "reference_distance_base": np.load(
                    self.static_metric_cache_dir / "reference_distance_base.npy",
                    mmap_mode="r",
                ),
            }
        self.road_corridor_cache_path = Path(road_corridor_cache_path) if road_corridor_cache_path else None
        self.road_corridor_cache = {}
        if self.road_corridor_cache_path is not None:
            cache = np.load(self.road_corridor_cache_path)
            self.road_corridor_cache = {
                "road_corridor_tokens": cache["road_corridor_tokens"],
                "road_corridor_mask": cache["road_corridor_mask"],
                "road_path_length_ratio": cache["road_path_length_ratio"],
                "road_path_total_length_m": cache["road_path_total_length_m"],
                "road_nearest_origin_dist_m": cache["road_nearest_origin_dist_m"],
                "road_nearest_destination_dist_m": cache["road_nearest_destination_dist_m"],
                "road_path_success": cache["road_path_success"],
            }
        if self.counterfactual_filter and self.counterfactual_source_filter:
            raise ValueError("Use only one of counterfactual_filter or counterfactual_source_filter.")
        global FAST_PATH_EMPTY_TRAJ
        if FAST_PATH_EMPTY_TRAJ is None and torch is not None:
            FAST_PATH_EMPTY_TRAJ = torch.zeros(1, 2, dtype=torch.float32)
        self.compact_manifest = load_compact_manifest(self.jsonl_path) if is_compact_meta_path(self.jsonl_path) else None
        self.compact_arrays = open_compact_arrays(self.compact_manifest) if self.compact_manifest else {}
        self._offsets: List[int] = []
        self._line_numbers: List[int] = []
        self._label_counts: Dict[str, Counter] | None = None
        if self.fast_meta_cache and not self.counterfactual_filter and not self.counterfactual_source_filter:
            cache_len = int(self.fast_meta_cache["sample_id"].shape[0])
            max_count = cache_len if max_samples is None or int(max_samples) <= 0 else min(int(max_samples), cache_len)
            self._offsets = list(range(max_count))
            self._line_numbers = list(range(max_count))
            self._label_counts = defaultdict(Counter)
        else:
            self._build_index(max_samples=max_samples)
        if self.qwen_plan_cache is not None and int(self.qwen_plan_cache.shape[0]) < len(self._offsets):
            raise ValueError(
                f"Qwen plan cache is shorter than selected dataset: cache={self.qwen_plan_cache.shape[0]}, "
                f"dataset={len(self._offsets)}, path={self.qwen_plan_cache_path}"
            )
        if self.geometry_cache:
            cache_len = int(self.geometry_cache["target_norm"].shape[0])
            if cache_len < len(self._offsets):
                raise ValueError(
                    f"Geometry cache is shorter than selected dataset: cache={cache_len}, "
                    f"dataset={len(self._offsets)}, path={self.geometry_cache_dir}"
                )
        if self.road_corridor_cache:
            cache_len = int(self.road_corridor_cache["road_corridor_tokens"].shape[0])
            if cache_len < len(self._offsets):
                raise ValueError(
                    f"Road corridor cache is shorter than selected dataset: cache={cache_len}, "
                    f"dataset={len(self._offsets)}, path={self.road_corridor_cache_path}"
                )
        self.records = LazyRecordSequence(self)

    @staticmethod
    def _matches_counterfactual_filter(record: Dict, filter_spec: str) -> bool:
        if not filter_spec:
            return True
        cf = record.get("counterfactual", {})
        if not bool(cf.get("enabled", False)):
            return False
        field_spec, sep, transition_spec = filter_spec.partition(":")
        field_spec = field_spec.strip()
        if field_spec and field_spec != "*" and str(cf.get("field", "")) != field_spec:
            return False
        if not sep:
            return True
        from_spec, arrow, to_spec = transition_spec.partition("->")
        if not arrow:
            raise ValueError(
                "counterfactual_filter must use 'field' or 'field:from->to' syntax, "
                f"got {filter_spec!r}."
            )
        from_spec = from_spec.strip()
        to_spec = to_spec.strip()
        if from_spec and from_spec != "*" and str(cf.get("from", "")) != from_spec:
            return False
        if to_spec and to_spec != "*" and str(cf.get("to", "")) != to_spec:
            return False
        return True

    def _iter_raw_records(self) -> Iterator[tuple[int, int, Dict]]:
        with self.jsonl_path.open("rb") as f:
            row_index = 0
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = json.loads(line.decode("utf-8"))
                yield offset, row_index, record
                row_index += 1

    def _read_raw_record_at(self, offset: int) -> Dict:
        with self.jsonl_path.open("rb") as f:
            f.seek(int(offset))
            line = f.readline()
        if not line.strip():
            raise ValueError(f"Empty record line at offset {offset} in {self.jsonl_path}")
        return json.loads(line.decode("utf-8"))

    def _finalize_record(self, record: Dict, row_index: int) -> Dict:
        if self.compact_manifest is not None:
            record = expand_compact_record(record, self.compact_arrays)
        if self.require_official_protocol:
            validate_official_record_protocol(record, row_index=row_index)
        return record

    def _selected_offsets(self, max_samples: Optional[int]) -> Iterable[tuple[int, int, Dict]]:
        max_count = None if max_samples is None or int(max_samples) <= 0 else int(max_samples)
        source_ids: set[int] | None = None
        if self.counterfactual_source_filter:
            source_ids = set()
            for _offset, _row_index, record in self._iter_raw_records():
                if not self._matches_counterfactual_filter(record, self.counterfactual_source_filter):
                    continue
                cf = record.get("counterfactual", {})
                source_id = cf.get("source_sample_id", record.get("sample_id"))
                try:
                    source_ids.add(int(source_id))
                except (TypeError, ValueError):
                    continue

        emitted = 0
        for offset, row_index, record in self._iter_raw_records():
            if source_ids is not None:
                if bool(record.get("counterfactual", {}).get("enabled", False)):
                    continue
                try:
                    sample_id = int(record.get("sample_id"))
                except (TypeError, ValueError):
                    continue
                if sample_id not in source_ids:
                    continue
            elif not self._matches_counterfactual_filter(record, self.counterfactual_filter):
                continue
            yield offset, row_index, record
            emitted += 1
            if max_count is not None and emitted >= max_count:
                break

    def _build_index(self, max_samples: Optional[int]) -> None:
        label_counts: Dict[str, Counter] = defaultdict(Counter)
        for offset, row_index, record in self._selected_offsets(max_samples):
            if self.require_official_protocol:
                validate_official_record_protocol(record, row_index=row_index)
            self._offsets.append(int(offset))
            self._line_numbers.append(int(row_index))
            for field, value in record.get("labels", {}).items():
                label_counts[str(field)][str(value)] += 1
        self._label_counts = label_counts
        if not self._offsets:
            suffix = (
                f" with counterfactual_filter={self.counterfactual_filter!r}"
                if self.counterfactual_filter
                else ""
            )
            if self.counterfactual_source_filter:
                suffix = f" with counterfactual_source_filter={self.counterfactual_source_filter!r}"
            raise ValueError(f"No records loaded from {self.jsonl_path}{suffix}")

    def __len__(self) -> int:
        return len(self._offsets)

    def get_record(self, index: int) -> Dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record = self._read_raw_record_at(self._offsets[index])
        return self._finalize_record(record, row_index=self._line_numbers[index])

    def __getitem__(self, index: int) -> Dict:
        if self.fast_meta_cache:
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError(index)
            cf_field_id = int(self.fast_meta_cache["counterfactual_field_id"][index])
            item = {
                "sample_id": int(self.fast_meta_cache["sample_id"][index]),
                "dataset": str(self.fast_meta_manifest.get("dataset", "")),
                "mobility_tokens": self.fast_meta_cache["mobility_tokens"][index],
                "trip_duration_min": float(self.fast_meta_cache["trip_duration_min"][index]),
                "counterfactual_enabled": bool(self.fast_meta_cache["counterfactual_enabled"][index]),
                "counterfactual_field_id": cf_field_id,
                "_fast_path": True,
            }
            if self.qwen_plan_cache is not None:
                item["qwen_semantic_plan_tokens"] = self.qwen_plan_cache[index]
            if self.geometry_cache:
                item["target_norm"] = self.geometry_cache["target_norm"][index]
                item["reference_norm"] = self.geometry_cache["reference_norm"][index]
                item["route_scale"] = float(self.geometry_cache["route_scale"][index].reshape(-1)[0])
            if self.static_metric_cache:
                item["base_token_scores"] = self.static_metric_cache["base_token_scores"][index]
                item["reference_distance_base"] = float(
                    self.static_metric_cache["reference_distance_base"][index].reshape(-1)[0]
                )
            if self.road_corridor_cache:
                item.update(self._road_cache_item(index))
            return item

        record = self.get_record(index)
        traj = np.asarray(record.get(self.target_key, record.get("trajectory", [])), dtype=np.float32)
        if traj.ndim != 2 or traj.shape[-1] != 2:
            raise ValueError(f"Record {index} has invalid trajectory shape {traj.shape}")
        labels = record.get("labels", {})
        item = {
            "sample_id": int(record["sample_id"]),
            "dataset": record.get("dataset", ""),
            "traj": torch.from_numpy(traj).float(),
            "mobility_tokens": torch.tensor(self.vocab.encode(labels), dtype=torch.long),
            "labels": dict(labels),
            "instruction": str(record.get("instruction", "")),
            "metrics": dict(record.get("metrics", {})),
            "trip_attributes": dict(record.get("trip_attributes", {})),
            "counterfactual": dict(record.get("counterfactual", {})),
        }
        if self.include_reference:
            reference = np.asarray(record.get("reference", []), dtype=np.float32)
            if reference.ndim != 2 or reference.shape[-1] != 2 or reference.shape[0] == 0:
                raise ValueError(
                    f"Record {index} has invalid or missing official OD-chord reference shape {reference.shape}"
                )
            item["reference"] = torch.from_numpy(reference).float()
        if self.qwen_plan_cache is not None:
            plan = np.asarray(self.qwen_plan_cache[index], dtype=np.float32).copy()
            item["qwen_semantic_plan_tokens"] = torch.from_numpy(plan).float()
        if self.geometry_cache:
            item["target_norm"] = torch.from_numpy(
                np.asarray(self.geometry_cache["target_norm"][index], dtype=np.float32).copy()
            ).float()
            item["reference_norm"] = torch.from_numpy(
                np.asarray(self.geometry_cache["reference_norm"][index], dtype=np.float32).copy()
            ).float()
            item["route_scale"] = torch.tensor(
                float(np.asarray(self.geometry_cache["route_scale"][index]).reshape(-1)[0]),
                dtype=torch.float32,
            )
        if self.road_corridor_cache:
            item.update(self._road_cache_item(index))
        return item

    def _road_cache_item(self, index: int) -> Dict:
        return {
            "road_corridor_tokens": np.asarray(
                self.road_corridor_cache["road_corridor_tokens"][index],
                dtype=np.float32,
            ).copy(),
            "road_corridor_mask": np.asarray(
                self.road_corridor_cache["road_corridor_mask"][index],
                dtype=bool,
            ).copy(),
            "road_path_length_ratio": np.asarray(
                self.road_corridor_cache["road_path_length_ratio"][index],
                dtype=np.float32,
            ).copy(),
            "road_path_total_length_m": np.asarray(
                self.road_corridor_cache["road_path_total_length_m"][index],
                dtype=np.float32,
            ).copy(),
            "road_nearest_origin_dist_m": float(self.road_corridor_cache["road_nearest_origin_dist_m"][index]),
            "road_nearest_destination_dist_m": float(self.road_corridor_cache["road_nearest_destination_dist_m"][index]),
            "road_path_success": np.asarray(
                self.road_corridor_cache["road_path_success"][index],
                dtype=bool,
            ).copy(),
        }

    def label_distribution(self) -> Dict[str, Dict[str, int]]:
        counts = self._label_counts or {}
        return {field: dict(sorted(counter.items())) for field, counter in counts.items()}


def collate_instruction_batch(batch: List[Dict]) -> Dict:
    if torch is None:
        raise RuntimeError("collate_instruction_batch requires PyTorch.")
    if batch and batch[0].get("_fast_path", False):
        out = {
            "sample_id": torch.tensor([item["sample_id"] for item in batch], dtype=torch.long),
            "traj": _fast_path_empty_traj().unsqueeze(0).expand(len(batch), -1, -1).clone(),
            "traj_mask": torch.ones(len(batch), 1, dtype=torch.bool),
            "mobility_tokens": torch.as_tensor(
                np.stack([item["mobility_tokens"] for item in batch], axis=0),
                dtype=torch.long,
            ),
            "instruction": [""] * len(batch),
            "labels": [{} for _ in batch],
            "metrics": [{} for _ in batch],
            "trip_attributes": [],
            "counterfactual": [],
            "trip_duration_min": torch.tensor(
                [item["trip_duration_min"] for item in batch],
                dtype=torch.float32,
            ),
            "counterfactual_enabled": torch.tensor(
                [item["counterfactual_enabled"] for item in batch],
                dtype=torch.bool,
            ),
            "counterfactual_field_id": torch.tensor(
                [item["counterfactual_field_id"] for item in batch],
                dtype=torch.long,
            ),
        }
        if "target_norm" in batch[0]:
            out["reference"] = _fast_path_empty_traj().unsqueeze(0).expand(len(batch), -1, -1).clone()
            out["reference_mask"] = torch.ones(len(batch), 1, dtype=torch.bool)
        if "qwen_semantic_plan_tokens" in batch[0]:
            out["qwen_semantic_plan_tokens"] = torch.as_tensor(
                np.stack([item["qwen_semantic_plan_tokens"] for item in batch], axis=0),
                dtype=torch.float32,
            )
        if "target_norm" in batch[0]:
            out["target_norm"] = torch.as_tensor(
                np.stack([item["target_norm"] for item in batch], axis=0),
                dtype=torch.float32,
            )
            out["reference_norm"] = torch.as_tensor(
                np.stack([item["reference_norm"] for item in batch], axis=0),
                dtype=torch.float32,
            )
            out["route_scale"] = torch.tensor([item["route_scale"] for item in batch], dtype=torch.float32)
        if "base_token_scores" in batch[0]:
            out["base_token_scores"] = torch.as_tensor(
                np.stack([item["base_token_scores"] for item in batch], axis=0),
                dtype=torch.float32,
            )
            out["reference_distance_base"] = torch.tensor(
                [item["reference_distance_base"] for item in batch],
                dtype=torch.float32,
            )
        if "road_corridor_tokens" in batch[0]:
            out["road_corridor_tokens"] = torch.as_tensor(
                np.stack([item["road_corridor_tokens"] for item in batch], axis=0),
                dtype=torch.float32,
            )
            out["road_corridor_mask"] = torch.as_tensor(
                np.stack([item["road_corridor_mask"] for item in batch], axis=0),
                dtype=torch.bool,
            )
            out["road_path_length_ratio"] = torch.as_tensor(
                np.stack([item["road_path_length_ratio"] for item in batch], axis=0),
                dtype=torch.float32,
            )
            out["road_path_total_length_m"] = torch.as_tensor(
                np.stack([item["road_path_total_length_m"] for item in batch], axis=0),
                dtype=torch.float32,
            )
            out["road_nearest_origin_dist_m"] = torch.tensor(
                [item["road_nearest_origin_dist_m"] for item in batch],
                dtype=torch.float32,
            )
            out["road_nearest_destination_dist_m"] = torch.tensor(
                [item["road_nearest_destination_dist_m"] for item in batch],
                dtype=torch.float32,
            )
            out["road_path_success"] = torch.as_tensor(
                np.stack([item["road_path_success"] for item in batch], axis=0),
                dtype=torch.bool,
            )
        return out

    max_traj_len = max(item["traj"].shape[0] for item in batch)
    traj = torch.zeros(len(batch), max_traj_len, 2, dtype=torch.float32)
    traj_mask = torch.zeros(len(batch), max_traj_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        length = item["traj"].shape[0]
        traj[i, :length] = item["traj"]
        traj_mask[i, :length] = True

    out = {
        "sample_id": torch.tensor([item["sample_id"] for item in batch], dtype=torch.long),
        "traj": traj,
        "traj_mask": traj_mask,
        "mobility_tokens": torch.stack([item["mobility_tokens"] for item in batch], dim=0),
        "instruction": [item["instruction"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "metrics": [item["metrics"] for item in batch],
        "trip_attributes": [item.get("trip_attributes", {}) for item in batch],
        "counterfactual": [item.get("counterfactual", {}) for item in batch],
    }
    if "reference" in batch[0]:
        max_ref_len = max(item["reference"].shape[0] for item in batch)
        reference = torch.zeros(len(batch), max_ref_len, 2, dtype=torch.float32)
        reference_mask = torch.zeros(len(batch), max_ref_len, dtype=torch.bool)
        for i, item in enumerate(batch):
            length = item["reference"].shape[0]
            reference[i, :length] = item["reference"]
            reference_mask[i, :length] = True
        out["reference"] = reference
        out["reference_mask"] = reference_mask
    if "qwen_semantic_plan_tokens" in batch[0]:
        out["qwen_semantic_plan_tokens"] = torch.stack(
            [item["qwen_semantic_plan_tokens"] for item in batch],
            dim=0,
        )
    if "target_norm" in batch[0]:
        out["target_norm"] = torch.stack([item["target_norm"] for item in batch], dim=0)
        out["reference_norm"] = torch.stack([item["reference_norm"] for item in batch], dim=0)
        out["route_scale"] = torch.stack([item["route_scale"] for item in batch], dim=0)
    if "base_token_scores" in batch[0]:
        out["base_token_scores"] = torch.stack([item["base_token_scores"] for item in batch], dim=0)
        out["reference_distance_base"] = torch.stack([item["reference_distance_base"] for item in batch], dim=0)
    if "road_corridor_tokens" in batch[0]:
        out["road_corridor_tokens"] = torch.as_tensor(
            np.stack([item["road_corridor_tokens"] for item in batch], axis=0),
            dtype=torch.float32,
        )
        out["road_corridor_mask"] = torch.as_tensor(
            np.stack([item["road_corridor_mask"] for item in batch], axis=0),
            dtype=torch.bool,
        )
        out["road_path_length_ratio"] = torch.as_tensor(
            np.stack([item["road_path_length_ratio"] for item in batch], axis=0),
            dtype=torch.float32,
        )
        out["road_path_total_length_m"] = torch.as_tensor(
            np.stack([item["road_path_total_length_m"] for item in batch], axis=0),
            dtype=torch.float32,
        )
        out["road_nearest_origin_dist_m"] = torch.tensor(
            [item["road_nearest_origin_dist_m"] for item in batch],
            dtype=torch.float32,
        )
        out["road_nearest_destination_dist_m"] = torch.tensor(
            [item["road_nearest_destination_dist_m"] for item in batch],
            dtype=torch.float32,
        )
        out["road_path_success"] = torch.as_tensor(
            np.stack([item["road_path_success"] for item in batch], axis=0),
            dtype=torch.bool,
        )
    if "trip_attributes" in batch[0]:
        out["trip_duration_min"] = torch.tensor(
            [
                float((item.get("trip_attributes") or {}).get("trip_duration_min", 0.0) or 0.0)
                for item in batch
            ],
            dtype=torch.float32,
        )
    return out
