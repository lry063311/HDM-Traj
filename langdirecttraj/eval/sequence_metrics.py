from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_token_records(path: Path, key: str) -> List[Dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            seq = record.get(key)
            if seq is None and key == "grid_tokens":
                seq = record.get("tokens")
            if seq is None:
                raise KeyError(f"Record missing token key '{key}': {record.keys()}")
            record = dict(record)
            record["_sequence"] = [int(x) for x in seq]
            records.append(record)
    if not records:
        raise ValueError(f"No token records loaded from {path}")
    return records


def jaccard_similarity(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def cosine_similarity(a: List[int], b: List[int]) -> float:
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    dot = sum(ca[k] * cb[k] for k in keys)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return float(dot / max(na * nb, 1e-12))


def ngram_counts(seq: List[int], n: int) -> Counter:
    if len(seq) < n:
        return Counter()
    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))


def bleu_like(candidate: List[int], references: List[List[int]], max_n: int = 4) -> float:
    if not candidate or not references:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        cand_counts = ngram_counts(candidate, n)
        if not cand_counts:
            precisions.append(1e-9)
            continue
        max_ref_counts = Counter()
        for ref in references:
            ref_counts = ngram_counts(ref, n)
            for gram, count in ref_counts.items():
                max_ref_counts[gram] = max(max_ref_counts[gram], count)
        clipped = sum(min(count, max_ref_counts[gram]) for gram, count in cand_counts.items())
        precisions.append((clipped + 1.0) / (sum(cand_counts.values()) + 1.0))
    ref_lens = np.asarray([len(ref) for ref in references], dtype=np.float32)
    ref_len = float(ref_lens[np.argmin(np.abs(ref_lens - len(candidate)))])
    bp = 1.0 if len(candidate) > ref_len else math.exp(1.0 - ref_len / max(len(candidate), 1))
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / max_n))


def js_divergence_from_values(real_values: List[int], gen_values: List[int], bins: int = 400) -> float:
    values = real_values + gen_values
    if not values:
        return 0.0
    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        return 0.0
    real_hist, _ = np.histogram(real_values, bins=bins, range=(min_v, max_v))
    gen_hist, _ = np.histogram(gen_values, bins=bins, range=(min_v, max_v))
    p = real_hist.astype(np.float64) / max(real_hist.sum(), 1)
    q = gen_hist.astype(np.float64) / max(gen_hist.sum(), 1)
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    kl_pm = np.sum(p[mask_p] * np.log(p[mask_p] / np.clip(m[mask_p], 1e-12, None)))
    kl_qm = np.sum(q[mask_q] * np.log(q[mask_q] / np.clip(m[mask_q], 1e-12, None)))
    return float(0.5 * (kl_pm + kl_qm))


def entropy_from_values(values: List[int]) -> float:
    if not values:
        return 0.0
    counts = np.asarray(list(Counter(values).values()), dtype=np.float64)
    probs = counts / max(counts.sum(), 1.0)
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))


def connection_ids(sequences: Iterable[List[int]], n: int = 2) -> Tuple[List[int], Dict[Tuple[int, ...], int]]:
    mapping: Dict[Tuple[int, ...], int] = {}
    values = []
    for seq in sequences:
        for i in range(max(len(seq) - n + 1, 0)):
            gram = tuple(seq[i : i + n])
            if gram not in mapping:
                mapping[gram] = len(mapping)
            values.append(mapping[gram])
    return values, mapping


def evaluate_sequences(real_records: List[Dict], gen_records: List[Dict]) -> Dict:
    n = min(len(real_records), len(gen_records))
    real_records = real_records[:n]
    gen_records = gen_records[:n]
    real_sequences = [record["_sequence"] for record in real_records]
    gen_sequences = [record["_sequence"] for record in gen_records]

    od_references = defaultdict(list)
    for seq in real_sequences:
        if seq:
            od_references[(seq[0], seq[-1])].append(seq)

    jaccard = []
    cosine = []
    bleu = []
    finished = []
    exact = []
    length_abs_error = []
    for real, gen in zip(real_sequences, gen_sequences):
        jaccard.append(jaccard_similarity(gen, real))
        cosine.append(cosine_similarity(gen, real))
        refs = od_references.get((gen[0], gen[-1]), real_sequences) if gen else real_sequences
        bleu.append(bleu_like(gen, refs))
        finished.append(1.0 if real and gen and gen[-1] == real[-1] else 0.0)
        exact.append(1.0 if gen == real else 0.0)
        length_abs_error.append(abs(len(gen) - len(real)))

    real_links = [token for seq in real_sequences for token in seq]
    gen_links = [token for seq in gen_sequences for token in seq]
    real_conn, conn_map = connection_ids(real_sequences, n=2)
    gen_conn = []
    for seq in gen_sequences:
        for i in range(max(len(seq) - 1, 0)):
            gram = tuple(seq[i : i + 2])
            if gram not in conn_map:
                conn_map[gram] = len(conn_map)
            gen_conn.append(conn_map[gram])

    return {
        "num_compared": int(n),
        "finished_rate": float(np.mean(finished)),
        "exact_match_rate": float(np.mean(exact)),
        "jaccard": float(np.mean(jaccard)),
        "cosine": float(np.mean(cosine)),
        "bleu_like": float(np.mean(bleu)),
        "length_abs_error": float(np.mean(length_abs_error)),
        "link_jsd": js_divergence_from_values(real_links, gen_links),
        "connection_jsd": js_divergence_from_values(real_conn, gen_conn),
        "unigram_entropy": entropy_from_values(gen_links),
        "bigram_entropy": entropy_from_values(gen_conn),
        "ue": entropy_from_values(gen_links),
        "be": entropy_from_values(gen_conn),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrajGPT-R-style sequence metrics for grid/link token trajectories.")
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--real-key", default="grid_tokens")
    parser.add_argument("--generated-key", default="grid_tokens")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_records = read_token_records(args.real, args.real_key)
    gen_records = read_token_records(args.generated, args.generated_key)
    metrics = evaluate_sequences(real_records, gen_records)
    metrics.update({"real": str(args.real), "generated": str(args.generated)})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    except OSError:
        # Windows/PowerShell can fail on stdout for non-ASCII workspace paths.
        # The metrics file has already been written when --output is provided.
        if args.output is None:
            raise


if __name__ == "__main__":
    main()
