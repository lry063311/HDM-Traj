from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping


DEFAULT_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_ORDER = ("train", "val", "test")


def parse_split_ratios(raw: str | Mapping[str, float] | None) -> dict[str, float]:
    if raw is None:
        return dict(DEFAULT_SPLIT_RATIOS)
    if isinstance(raw, Mapping):
        ratios = {str(key): float(value) for key, value in raw.items()}
    else:
        text = str(raw).strip()
        if not text:
            return dict(DEFAULT_SPLIT_RATIOS)
        if "=" in text:
            ratios = {}
            for item in text.replace(";", ",").split(","):
                if not item.strip():
                    continue
                key, value = item.split("=", 1)
                ratios[key.strip()] = float(value)
        else:
            values = [float(item.strip()) for item in text.split(",") if item.strip()]
            if len(values) != 3:
                raise ValueError("--split-ratios must be '0.8,0.1,0.1' or 'train=0.8,val=0.1,test=0.1'.")
            ratios = dict(zip(SPLIT_ORDER, values))
    missing = [name for name in SPLIT_ORDER if name not in ratios]
    if missing:
        raise ValueError(f"Missing split ratios for: {missing}")
    total = sum(float(ratios[name]) for name in SPLIT_ORDER)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return {name: float(ratios[name]) / total for name in SPLIT_ORDER}


def split_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    total = int(total)
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    train = int(round(total * float(ratios["train"])))
    val = int(round(total * float(ratios["val"])))
    train = min(max(train, 0), total)
    val = min(max(val, 0), total - train)
    test = total - train - val
    return {"train": train, "val": val, "test": test}


def assign_sample_splits(
    sample_ids: Iterable[int],
    ratios: Mapping[str, float] | None = None,
    seed: int = 20260617,
) -> dict[int, str]:
    ratios = dict(ratios or DEFAULT_SPLIT_RATIOS)
    ids = [int(sample_id) for sample_id in sample_ids]
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    counts = split_counts(len(ids), ratios)
    split_by_id: dict[int, str] = {}
    cursor = 0
    for split in SPLIT_ORDER:
        end = cursor + counts[split]
        for sample_id in ids[cursor:end]:
            split_by_id[int(sample_id)] = split
        cursor = end
    return split_by_id


def source_sample_id(record: Mapping) -> int:
    cf = record.get("counterfactual", {})
    if isinstance(cf, Mapping) and cf.get("source_sample_id") is not None:
        return int(cf["source_sample_id"])
    return int(record["sample_id"])


def record_split(record: Mapping, split_by_id: Mapping[int, str], default: str | None = None) -> str:
    sample_id = source_sample_id(record)
    if sample_id in split_by_id:
        return str(split_by_id[sample_id])
    if default is not None:
        return str(default)
    raise KeyError(f"sample_id={sample_id} is not present in the split manifest.")


def summarize_split_records(records: Iterable[Mapping]) -> dict:
    counts = Counter()
    label_counts: dict[str, Counter] = defaultdict(Counter)
    cf_counts = Counter()
    for record in records:
        split = str(record.get("split", "unknown"))
        counts[split] += 1
        for field, value in dict(record.get("labels", {})).items():
            label_counts[f"{split}:{field}"][str(value)] += 1
        cf = record.get("counterfactual", {})
        if isinstance(cf, Mapping) and cf.get("enabled"):
            cf_counts[f"{split}:{cf.get('field')}:{cf.get('from')}->{cf.get('to')}"] += 1
    return {
        "record_counts": {key: int(counts.get(key, 0)) for key in (*SPLIT_ORDER, "unknown")},
        "label_counts": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(label_counts.items())
        },
        "counterfactual_transition_counts": dict(sorted(cf_counts.items())),
    }


def write_split_manifest(
    path: Path,
    split_by_id: Mapping[int, str],
    ratios: Mapping[str, float],
    seed: int,
    dataset: str,
    source: str,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(split_by_id.values())
    payload = {
        "dataset": str(dataset),
        "source": str(source),
        "split_seed": int(seed),
        "split_ratios": {name: float(ratios[name]) for name in SPLIT_ORDER},
        "split_counts": {name: int(counts.get(name, 0)) for name in SPLIT_ORDER},
        "sample_splits": {str(int(sample_id)): str(split) for sample_id, split in sorted(split_by_id.items())},
        "leakage_guard": {
            "split_unit": "base_sample_id",
            "counterfactual_variants_follow_source_sample": True,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_split_manifest(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sample_splits = payload.get("sample_splits", {})
    if not isinstance(sample_splits, dict) or not sample_splits:
        raise ValueError(f"Split manifest has no sample_splits: {path}")
    payload["sample_splits"] = {int(sample_id): str(split) for sample_id, split in sample_splits.items()}
    return payload


def iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
