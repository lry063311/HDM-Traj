from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable, List

from langdirecttraj.eval.sequence_metrics import read_token_records


def _as_int_sequence(seq: Iterable[Any]) -> List[int]:
    return [int(x) for x in list(seq)]


def _load_pickle_sequences(path: Path) -> List[List[int]]:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        items = [obj[key] for key in sorted(obj)]
    else:
        items = list(obj)
    sequences: List[List[int]] = []
    for item in items:
        if isinstance(item, dict):
            if "grid_tokens" in item:
                item = item["grid_tokens"]
            elif "tokens" in item:
                item = item["tokens"]
            elif "trajectory" in item:
                item = item["trajectory"]
            else:
                raise KeyError(f"Cannot infer token sequence from pickle record keys: {sorted(item.keys())}")
        sequences.append(_as_int_sequence(item))
    if not sequences:
        raise ValueError(f"No token sequences loaded from {path}")
    return sequences


def import_trajgpt_pickle(args: argparse.Namespace) -> dict[str, Any]:
    real_records = read_token_records(args.real_jsonl, args.real_key)
    generated = _load_pickle_sequences(args.generated_pickle)
    n = min(len(real_records), len(generated), args.max_samples or 10**18)
    if n <= 0:
        raise ValueError("No overlapping records to export.")
    output_records = []
    od_mismatches = []
    for idx in range(n):
        source = real_records[idx]
        tokens = generated[idx]
        record = {
            "sample_id": int(source.get("sample_id", idx)),
            "dataset": str(source.get("dataset", args.dataset)),
            "instruction": str(source.get("instruction", "")),
            "labels": dict(source.get("labels", {})),
            "grid_tokens": tokens,
            "origin": int(tokens[0]) if tokens else -1,
            "destination": int(tokens[-1]) if tokens else -1,
            "length": int(len(tokens)),
            "source": args.source_name,
        }
        real_od = (source.get("origin"), source.get("destination"))
        gen_od = (record["origin"], record["destination"])
        if real_od != gen_od:
            od_mismatches.append(
                {
                    "index": idx,
                    "sample_id": record["sample_id"],
                    "real_od": [int(real_od[0]) if real_od[0] is not None else -1, int(real_od[1]) if real_od[1] is not None else -1],
                    "generated_od": [record["origin"], record["destination"]],
                }
            )
        output_records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "protocol": "trajgpt_pickle_to_langdirecttraj_table2_jsonl",
        "real_jsonl": str(args.real_jsonl),
        "generated_pickle": str(args.generated_pickle),
        "output": str(args.output),
        "num_real_records": len(real_records),
        "num_generated_records": len(generated),
        "num_exported": n,
        "od_aligned": not od_mismatches,
        "od_mismatch_examples": od_mismatches[:10],
        "source_name": args.source_name,
        "note": (
            "Use compare_trajgpt_table2.py after this export. A fair paper comparison "
            "requires od_aligned=true and matching sample order."
        ),
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert TrajGPT-R pickle token sequences into LangDirectTraj JSONL so "
            "Table-2 aligned comparison can use the same real OD/sample records."
        )
    )
    parser.add_argument("--generated-pickle", type=Path, required=True)
    parser.add_argument("--real-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--real-key", default="grid_tokens")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--source-name", default="TrajGPT-R")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = import_trajgpt_pickle(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
