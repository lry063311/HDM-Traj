from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langdirecttraj.eval.sequence_metrics import evaluate_sequences, read_token_records


HIGHER_IS_BETTER = {
    "finished_rate",
    "exact_match_rate",
    "jaccard",
    "cosine",
    "bleu_like",
    "unigram_entropy",
    "bigram_entropy",
    "ue",
    "be",
}

LOWER_IS_BETTER = {
    "length_abs_error",
    "link_jsd",
    "connection_jsd",
}


def _identity(record: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    return (
        record.get("sample_id"),
        record.get("origin"),
        record.get("destination"),
        record.get("dataset"),
    )


def _alignment_report(real: List[Dict[str, Any]], ours: List[Dict[str, Any]], baseline: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = min(len(real), len(ours), len(baseline))
    sample_mismatches = []
    od_mismatches = []
    for idx in range(n):
        real_id = _identity(real[idx])
        ours_id = _identity(ours[idx])
        baseline_id = _identity(baseline[idx])
        if real_id[0] != ours_id[0] or real_id[0] != baseline_id[0]:
            sample_mismatches.append(
                {
                    "index": idx,
                    "real_sample_id": real_id[0],
                    "ours_sample_id": ours_id[0],
                    "baseline_sample_id": baseline_id[0],
                }
            )
        if real_id[1:3] != ours_id[1:3] or real_id[1:3] != baseline_id[1:3]:
            od_mismatches.append(
                {
                    "index": idx,
                    "real_od": list(real_id[1:3]),
                    "ours_od": list(ours_id[1:3]),
                    "baseline_od": list(baseline_id[1:3]),
                }
            )
    return {
        "num_real": len(real),
        "num_ours": len(ours),
        "num_baseline": len(baseline),
        "num_compared": n,
        "same_count": len(real) == len(ours) == len(baseline),
        "sample_id_aligned": not sample_mismatches,
        "od_aligned": not od_mismatches,
        "sample_mismatch_examples": sample_mismatches[:5],
        "od_mismatch_examples": od_mismatches[:5],
        "fair_comparison": len(real) == len(ours) == len(baseline)
        and not sample_mismatches
        and not od_mismatches,
    }


def _metric_delta(ours: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    deltas: Dict[str, Any] = {}
    metric_names = sorted((set(ours) & set(baseline)) & (HIGHER_IS_BETTER | LOWER_IS_BETTER))
    for metric in metric_names:
        ours_value = float(ours[metric])
        baseline_value = float(baseline[metric])
        delta = ours_value - baseline_value
        if metric in HIGHER_IS_BETTER:
            better = delta > 0.0
            direction = "higher_is_better"
        else:
            better = delta < 0.0
            direction = "lower_is_better"
        deltas[metric] = {
            "ours": ours_value,
            "baseline": baseline_value,
            "delta": delta,
            "relative_delta": delta / max(abs(baseline_value), 1e-12),
            "direction": direction,
            "ours_better": bool(better),
        }
    return deltas


def compare_table2(args: argparse.Namespace) -> Dict[str, Any]:
    real = read_token_records(args.real, args.real_key)
    ours = read_token_records(args.ours, args.ours_key)
    baseline = read_token_records(args.baseline, args.baseline_key)
    alignment = _alignment_report(real, ours, baseline)
    n = int(alignment["num_compared"])
    ours_metrics = evaluate_sequences(real[:n], ours[:n])
    baseline_metrics = evaluate_sequences(real[:n], baseline[:n])
    deltas = _metric_delta(ours_metrics, baseline_metrics)
    return {
        "protocol": "trajgpt_r_style_table2_aligned_compare",
        "real": str(args.real),
        "ours": str(args.ours),
        "baseline": str(args.baseline),
        "baseline_name": args.baseline_name,
        "ours_name": args.ours_name,
        "alignment": alignment,
        "ours_metrics": ours_metrics,
        "baseline_metrics": baseline_metrics,
        "metric_deltas": deltas,
        "summary": {
            "fair_comparison": alignment["fair_comparison"],
            "ours_better_count": sum(1 for item in deltas.values() if item["ours_better"]),
            "num_metrics": len(deltas),
            "primary_note": (
                "Use this report only when fair_comparison=true. "
                "It compares generated token sequences under the same real OD/sample order."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LangDirectTraj against a TrajGPT-R-style baseline using the "
            "same real token JSONL and Table-2 token mobility metrics."
        )
    )
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--real-key", default="grid_tokens")
    parser.add_argument("--ours-key", default="grid_tokens")
    parser.add_argument("--baseline-key", default="grid_tokens")
    parser.add_argument("--ours-name", default="LangDirectTraj")
    parser.add_argument("--baseline-name", default="TrajGPT-R-style baseline")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_table2(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
