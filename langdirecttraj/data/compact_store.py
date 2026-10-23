from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator

import numpy as np
from numpy.lib.format import open_memmap

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None


COMPACT_SCHEMA_VERSION = "compact_v1"
SPLIT_ORDER = ("train", "val", "test")


def progress_iter(iterable: Iterable, desc: str, total: int | None = None, prefix: str = "compact") -> Iterator:
    if tqdm is not None:
        yield from tqdm(iterable, total=total, desc=desc, unit="record", dynamic_ncols=True)
        return
    started = time.time()
    for idx, item in enumerate(iterable, start=1):
        if idx % 50000 == 0:
            elapsed = max(time.time() - started, 1e-6)
            rate = idx / elapsed
            if total:
                eta = max(total - idx, 0) / max(rate, 1e-6)
                print(
                    f"[{prefix}] {desc}: {idx:,}/{total:,} ({idx / max(total, 1):.1%}), "
                    f"{rate:.1f} record/s, eta={eta / 60.0:.1f} min",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(f"[{prefix}] {desc}: {idx:,}, {rate:.1f} record/s", file=sys.stderr, flush=True)
        yield item


def read_jsonl(path: Path, max_samples: int = 0) -> Iterator[Dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        emitted = 0
        for line in handle:
            if not line.strip():
                continue
            if max_samples > 0 and emitted >= max_samples:
                break
            emitted += 1
            yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _as_lonlat(record: Dict, key: str) -> np.ndarray:
    arr = np.asarray(record.get(key, []), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[-1] != 2 or arr.shape[0] == 0:
        raise ValueError(f"Record {record.get('sample_id', '?')} has invalid {key!r} shape {arr.shape}.")
    return arr


def _drop_large_arrays(record: Dict) -> Dict:
    out = dict(record)
    out.pop("trajectory", None)
    out.pop("reference", None)
    out.pop("pseudo_trajectory", None)
    return out


def compact_manifest_path(meta_path: Path) -> Path:
    return Path(meta_path).with_suffix(".compact_manifest.json")


def is_compact_meta_path(path: Path) -> bool:
    path = Path(path)
    return compact_manifest_path(path).exists()


def load_compact_manifest(meta_path: Path) -> Dict:
    manifest_path = compact_manifest_path(Path(meta_path))
    if not manifest_path.exists():
        raise FileNotFoundError(f"Compact manifest not found for {meta_path}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = Path(str(manifest.get("base_dir", ".")).replace("\\", "/"))
    if not base_dir.exists():
        # Compact packages are often generated locally and then moved to a
        # server data disk. Keep manifests portable by resolving stale relative
        # base_dir values against the colocated manifest.
        manifest["base_dir"] = str(manifest_path.parent)
    return manifest


def expand_compact_record(record: Dict, arrays: Dict[str, np.ndarray]) -> Dict:
    out = dict(record)
    refs = out.get("compact_refs", {})
    if "trajectory" in refs:
        spec = refs["trajectory"]
        out["trajectory"] = np.asarray(arrays[spec["array"]][int(spec["row"])], dtype=np.float32).tolist()
    if "reference" in refs:
        spec = refs["reference"]
        out["reference"] = np.asarray(arrays[spec["array"]][int(spec["row"])], dtype=np.float32).tolist()
    if "pseudo_trajectory" in refs:
        spec = refs["pseudo_trajectory"]
        out["pseudo_trajectory"] = np.asarray(arrays[spec["array"]][int(spec["row"])], dtype=np.float32).tolist()
    if "trajectory" not in out and "pseudo_trajectory" in out:
        out["trajectory"] = out["pseudo_trajectory"]
    return out


def open_compact_arrays(manifest: Dict) -> Dict[str, np.ndarray]:
    base_dir = Path(str(manifest.get("base_dir", ".")).replace("\\", "/"))
    arrays = {}
    for name, rel_path in manifest.get("arrays", {}).items():
        rel = Path(str(rel_path).replace("\\", "/"))
        arrays[name] = np.load(base_dir / rel, mmap_mode="r")
    return arrays


def compact_instruction_split_files(
    input_paths: Dict[str, Path],
    output_dir: Path,
    dataset: str,
    target_len: int,
    overwrite: bool = False,
) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_paths = {
        split: output_dir / f"{dataset}_instruction_meta.{split}.jsonl"
        for split in SPLIT_ORDER
        if split in input_paths
    }
    base_path = output_dir / f"{dataset}_base_lonlat.npy"
    reference_path = output_dir / f"{dataset}_reference_lonlat.npy"
    if not overwrite and all(path.exists() for path in meta_paths.values()) and base_path.exists() and reference_path.exists():
        return {
            "skipped_existing": True,
            "meta_paths": {split: str(path) for split, path in meta_paths.items()},
            "arrays": {"base_lonlat": str(base_path), "reference_lonlat": str(reference_path)},
        }

    counts = {split: sum(1 for _ in read_jsonl(path)) for split, path in input_paths.items()}
    total = sum(counts.values())
    base_arr = open_memmap(base_path, mode="w+", dtype=np.float32, shape=(total, int(target_len), 2))
    ref_arr = open_memmap(reference_path, mode="w+", dtype=np.float32, shape=(total, int(target_len), 2))

    handles = {split: meta_paths[split].open("w", encoding="utf-8") for split in meta_paths}
    split_counts = {split: 0 for split in meta_paths}
    row = 0
    try:
        for split in SPLIT_ORDER:
            if split not in input_paths:
                continue
            for record in progress_iter(read_jsonl(input_paths[split]), f"compact instruction {split}", counts[split]):
                base_arr[row] = _as_lonlat(record, "trajectory")
                ref_arr[row] = _as_lonlat(record, "reference")
                meta = _drop_large_arrays(record)
                meta["compact_schema"] = COMPACT_SCHEMA_VERSION
                meta["base_index"] = int(row)
                meta["reference_index"] = int(row)
                meta["compact_refs"] = {
                    "trajectory": {"array": "base_lonlat", "row": int(row)},
                    "reference": {"array": "reference_lonlat", "row": int(row)},
                }
                handles[split].write(json.dumps(meta, ensure_ascii=False) + "\n")
                split_counts[split] += 1
                row += 1
    finally:
        for handle in handles.values():
            handle.close()
        base_arr.flush()
        ref_arr.flush()
        del base_arr, ref_arr

    manifest = {
        "schema": COMPACT_SCHEMA_VERSION,
        "dataset": dataset,
        "kind": "instruction",
        "base_dir": str(output_dir),
        "target_len": int(target_len),
        "num_records": int(total),
        "split_counts": split_counts,
        "arrays": {
            "base_lonlat": base_path.name,
            "reference_lonlat": reference_path.name,
        },
        "meta_paths": {split: meta_paths[split].name for split in meta_paths},
    }
    for split, meta_path in meta_paths.items():
        compact_manifest_path(meta_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "skipped_existing": False,
        "meta_paths": {split: str(path) for split, path in meta_paths.items()},
        "arrays": {"base_lonlat": str(base_path), "reference_lonlat": str(reference_path)},
        "manifest": manifest,
    }


class CompactCounterfactualWriter:
    def __init__(
        self,
        output_dir: Path,
        dataset: str,
        split: str,
        target_len: int,
        prefix: str = "filtered",
        shard_size: int = 100000,
    ):
        self.output_dir = Path(output_dir)
        self.dataset = str(dataset)
        self.split = str(split)
        self.target_len = int(target_len)
        self.prefix = str(prefix)
        self.shard_size = int(max(shard_size, 1))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.output_dir / f"{self.dataset}_counterfactual_instruction_samples.{self.prefix}.{self.split}.compact.jsonl"
        self._meta_handle = self.meta_path.open("w", encoding="utf-8")
        self._pseudo_buffer: list[np.ndarray] = []
        self._shard_paths: list[Path] = []
        self._num_records = 0

    def _array_name(self, shard_index: int) -> str:
        return f"pseudo_{self.prefix}_{self.split}_{shard_index:04d}"

    def _flush_shard(self) -> None:
        if not self._pseudo_buffer:
            return
        shard_index = len(self._shard_paths)
        path = self.output_dir / f"{self.dataset}_{self._array_name(shard_index)}.npy"
        arr = np.stack(self._pseudo_buffer, axis=0).astype(np.float32)
        np.save(path, arr, allow_pickle=False)
        self._shard_paths.append(path)
        self._pseudo_buffer.clear()

    def add(self, record: Dict) -> None:
        pseudo = _as_lonlat(record, "pseudo_trajectory")
        shard_index = len(self._shard_paths)
        row = len(self._pseudo_buffer)
        self._pseudo_buffer.append(pseudo.astype(np.float32))
        base_ref = record.get("compact_refs", {})
        has_external_base = "trajectory" in base_ref and "reference" in base_ref
        meta = _drop_large_arrays(record) if has_external_base else dict(record)
        base_spec = base_ref.get("trajectory", {})
        reference_spec = base_ref.get("reference", {})
        base_row = int(record.get("base_index", base_spec.get("row", record.get("source_sample_id", record.get("sample_id", 0)))))
        reference_row = int(record.get("reference_index", reference_spec.get("row", base_row)))
        meta["compact_schema"] = COMPACT_SCHEMA_VERSION
        meta["base_index"] = base_row
        meta["reference_index"] = reference_row
        compact_refs = {"pseudo_trajectory": {"array": self._array_name(shard_index), "row": int(row)}}
        if has_external_base:
            compact_refs["trajectory"] = {"array": "base_lonlat", "row": base_row}
            compact_refs["reference"] = {"array": "reference_lonlat", "row": reference_row}
        meta["compact_refs"] = compact_refs
        self._meta_handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self._num_records += 1
        if len(self._pseudo_buffer) >= self.shard_size:
            self._flush_shard()

    def close(self, base_arrays: Dict[str, str] | None = None) -> Dict:
        self._flush_shard()
        self._meta_handle.close()
        arrays = dict(base_arrays or {})
        for idx, path in enumerate(self._shard_paths):
            arrays[self._array_name(idx)] = path.name
        manifest = {
            "schema": COMPACT_SCHEMA_VERSION,
            "dataset": self.dataset,
            "kind": "counterfactual",
            "base_dir": str(self.output_dir),
            "split": self.split,
            "target_len": self.target_len,
            "num_records": self._num_records,
            "shard_size": self.shard_size,
            "arrays": arrays,
            "meta_path": self.meta_path.name,
        }
        compact_manifest_path(self.meta_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "meta_path": str(self.meta_path),
            "pseudo_paths": [str(path) for path in self._shard_paths],
            "num_records": self._num_records,
            "manifest": manifest,
        }
