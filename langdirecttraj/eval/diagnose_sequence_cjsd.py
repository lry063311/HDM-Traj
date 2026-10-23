from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from langdirecttraj.eval.sequence_metrics import (
    entropy_from_values,
    evaluate_sequences,
    read_token_records,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _bigrams(seq: List[int]) -> List[Tuple[int, int]]:
    return [tuple(seq[i : i + 2]) for i in range(max(len(seq) - 1, 0))]


def _sequence(record: Dict[str, Any]) -> List[int]:
    return [int(x) for x in record["_sequence"]]


def _json_bigram_rows(rows: List[Tuple[Tuple[int, int], int, int, int]]) -> List[Dict[str, Any]]:
    return [
        {
            "bigram": [int(bigram[0]), int(bigram[1])],
            "real_count": int(real_count),
            "generated_count": int(gen_count),
            "count_delta": int(delta),
        }
        for bigram, real_count, gen_count, delta in rows
    ]


def diagnose_sequences(
    real_records: List[Dict[str, Any]],
    gen_records: List[Dict[str, Any]],
    top_k: int = 20,
) -> Dict[str, Any]:
    n = min(len(real_records), len(gen_records))
    real_sequences = [_sequence(record) for record in real_records[:n]]
    gen_sequences = [_sequence(record) for record in gen_records[:n]]
    table2 = evaluate_sequences(real_records[:n], gen_records[:n])

    real_unigrams = Counter(token for seq in real_sequences for token in seq)
    gen_unigrams = Counter(token for seq in gen_sequences for token in seq)
    real_bigrams = Counter(bigram for seq in real_sequences for bigram in _bigrams(seq))
    gen_bigrams = Counter(bigram for seq in gen_sequences for bigram in _bigrams(seq))

    all_bigrams = set(real_bigrams) | set(gen_bigrams)
    real_bigram_total = sum(real_bigrams.values())
    gen_bigram_total = sum(gen_bigrams.values())
    shared_unique = set(real_bigrams) & set(gen_bigrams)
    overlap_count = sum(min(real_bigrams[bigram], gen_bigrams[bigram]) for bigram in all_bigrams)

    undercovered = sorted(
        [
            (bigram, real_bigrams[bigram], gen_bigrams[bigram], real_bigrams[bigram] - gen_bigrams[bigram])
            for bigram in all_bigrams
            if real_bigrams[bigram] > gen_bigrams[bigram]
        ],
        key=lambda row: row[3],
        reverse=True,
    )
    overgenerated = sorted(
        [
            (bigram, real_bigrams[bigram], gen_bigrams[bigram], gen_bigrams[bigram] - real_bigrams[bigram])
            for bigram in all_bigrams
            if gen_bigrams[bigram] > real_bigrams[bigram]
        ],
        key=lambda row: row[3],
        reverse=True,
    )

    per_sequence = []
    for idx, (real, gen) in enumerate(zip(real_sequences, gen_sequences)):
        real_bi = Counter(_bigrams(real))
        gen_bi = Counter(_bigrams(gen))
        keys = set(real_bi) | set(gen_bi)
        overlap = sum(min(real_bi[key], gen_bi[key]) for key in keys)
        real_count = sum(real_bi.values())
        gen_count = sum(gen_bi.values())
        denom = max(real_count, gen_count, 1)
        overlap_ratio = 1.0 if real_count == 0 and gen_count == 0 else float(overlap / denom)
        per_sequence.append(
            {
                "index": int(idx),
                "real_length": int(len(real)),
                "generated_length": int(len(gen)),
                "abs_length_error": int(abs(len(real) - len(gen))),
                "bigram_overlap_ratio": overlap_ratio,
                "real_bigram_count": int(real_count),
                "generated_bigram_count": int(gen_count),
            }
        )
    per_sequence.sort(key=lambda row: (row["bigram_overlap_ratio"], -row["abs_length_error"]))

    real_lengths = np.asarray([len(seq) for seq in real_sequences], dtype=np.float64)
    gen_lengths = np.asarray([len(seq) for seq in gen_sequences], dtype=np.float64)
    length_errors = np.abs(real_lengths - gen_lengths)

    diagnosis = []
    if table2["connection_jsd"] > 0.25:
        diagnosis.append("connection_jsd_high")
    if float(gen_bigram_total) < 0.9 * max(float(real_bigram_total), 1.0):
        diagnosis.append("generated_bigram_undercoverage")
    if float(np.mean(length_errors)) > 2.0:
        diagnosis.append("token_length_mismatch")
    if entropy_from_values(list(gen_unigrams.elements())) < 0.8 * max(entropy_from_values(list(real_unigrams.elements())), 1e-8):
        diagnosis.append("unigram_entropy_collapse")
    if entropy_from_values(list(gen_bigrams.elements())) < 0.8 * max(entropy_from_values(list(real_bigrams.elements())), 1e-8):
        diagnosis.append("bigram_entropy_collapse")

    return {
        "num_sequences": int(n),
        "table2": table2,
        "length": {
            "mean_real_token_length": float(np.mean(real_lengths)) if n else float("nan"),
            "mean_generated_token_length": float(np.mean(gen_lengths)) if n else float("nan"),
            "mean_abs_token_length_error": float(np.mean(length_errors)) if n else float("nan"),
            "p90_abs_token_length_error": float(np.percentile(length_errors, 90)) if n else float("nan"),
        },
        "unigrams": {
            "real_unique": int(len(real_unigrams)),
            "generated_unique": int(len(gen_unigrams)),
            "real_entropy": entropy_from_values(list(real_unigrams.elements())),
            "generated_entropy": entropy_from_values(list(gen_unigrams.elements())),
        },
        "bigrams": {
            "real_total": int(real_bigram_total),
            "generated_total": int(gen_bigram_total),
            "real_unique": int(len(real_bigrams)),
            "generated_unique": int(len(gen_bigrams)),
            "shared_unique": int(len(shared_unique)),
            "count_overlap_ratio": float(overlap_count / max(real_bigram_total, gen_bigram_total, 1)),
            "real_entropy": entropy_from_values(list(real_bigrams.elements())),
            "generated_entropy": entropy_from_values(list(gen_bigrams.elements())),
            "top_undercovered": _json_bigram_rows(undercovered[:top_k]),
            "top_overgenerated": _json_bigram_rows(overgenerated[:top_k]),
        },
        "worst_sequences": per_sequence[:top_k],
        "diagnosis_flags": diagnosis,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Table-2 C-JSD by comparing real/generated token bigram coverage."
    )
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--real-key", default="grid_tokens")
    parser.add_argument("--generated-key", default="grid_tokens")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_records = read_token_records(args.real, args.real_key)
    gen_records = read_token_records(args.generated, args.generated_key)
    payload = diagnose_sequences(real_records, gen_records, top_k=max(1, int(args.top_k)))
    payload.update({"real": str(args.real), "generated": str(args.generated)})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
