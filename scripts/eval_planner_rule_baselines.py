#!/usr/bin/env python
"""Audit deterministic rule coverage and compare planner interfaces."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langdirecttraj.eval.instruction_geometry import macro_f1  # noqa: E402
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB  # noqa: E402
from langdirecttraj.text.text_instruction_encoder import rule_token_overrides  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", required=True, type=Path)
    parser.add_argument("--test-input", required=True, type=Path)
    parser.add_argument("--qwen-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--include-empty-labels", action="store_true")
    return parser.parse_args()


def iter_records(path: Path, max_samples: int = 0) -> Iterable[Dict]:
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "instruction" not in record:
                continue
            labels = record.get("labels")
            if not isinstance(labels, dict):
                token_ids = record.get("mobility_tokens")
                if token_ids is None:
                    raise KeyError(f"Missing labels and mobility_tokens at {path}:{line_number}")
                labels = DEFAULT_MOBILITY_VOCAB.decode([int(value) for value in token_ids])
            yield {
                "instruction": str(record["instruction"]),
                "labels": {field: str(labels[field]) for field in DEFAULT_MOBILITY_VOCAB.fields},
            }
            count += 1
            if max_samples > 0 and count >= max_samples:
                return


def training_majorities(path: Path, max_samples: int) -> tuple[Dict[str, str], Dict[str, Dict[str, int]], int]:
    counts = {field: Counter() for field in DEFAULT_MOBILITY_VOCAB.fields}
    num_samples = 0
    for record in iter_records(path, max_samples=max_samples):
        num_samples += 1
        for field in DEFAULT_MOBILITY_VOCAB.fields:
            counts[field][record["labels"][field]] += 1
    if num_samples == 0:
        raise ValueError(f"No training records loaded from {path}")
    majority = {field: counts[field].most_common(1)[0][0] for field in DEFAULT_MOBILITY_VOCAB.fields}
    serializable_counts = {field: dict(counts[field]) for field in DEFAULT_MOBILITY_VOCAB.fields}
    return majority, serializable_counts, num_samples


def planner_summary(field_metrics: Dict[str, Dict], exact_match: float) -> Dict[str, float]:
    fields = DEFAULT_MOBILITY_VOCAB.fields
    return {
        "mean_accuracy": float(np.mean([field_metrics[field]["accuracy"] for field in fields])),
        "mean_macro_f1": float(np.mean([field_metrics[field]["macro_f1"] for field in fields])),
        "exact_match": float(exact_match),
    }


def evaluate_rules_only(
    path: Path,
    majority: Dict[str, str],
    max_samples: int,
    include_empty_labels: bool,
) -> Dict:
    fields = DEFAULT_MOBILITY_VOCAB.fields
    y_true: Dict[str, List[str]] = {field: [] for field in fields}
    y_pred: Dict[str, List[str]] = {field: [] for field in fields}
    covered = Counter()
    covered_correct = Counter()
    exact_count = 0
    num_samples = 0

    for record in iter_records(path, max_samples=max_samples):
        num_samples += 1
        overrides = rule_token_overrides(record["instruction"])
        predicted = dict(majority)
        for field, token_id in overrides.items():
            predicted[field] = DEFAULT_MOBILITY_VOCAB.id_to_token[field][int(token_id)]
            covered[field] += 1
            covered_correct[field] += int(predicted[field] == record["labels"][field])
        exact_count += int(all(predicted[field] == record["labels"][field] for field in fields))
        for field in fields:
            y_true[field].append(record["labels"][field])
            y_pred[field].append(predicted[field])

    if num_samples == 0:
        raise ValueError(f"No test records loaded from {path}")

    field_metrics = {}
    rule_audit = {}
    for field in fields:
        labels = DEFAULT_MOBILITY_VOCAB.token_to_id[field].keys()
        field_metrics[field] = macro_f1(
            y_true[field],
            y_pred[field],
            labels,
            include_empty=bool(include_empty_labels),
        )
        rule_audit[field] = {
            "covered_samples": int(covered[field]),
            "coverage_rate": float(covered[field] / num_samples),
            "covered_correct_samples": int(covered_correct[field]),
            "covered_accuracy": float(covered_correct[field] / covered[field]) if covered[field] else None,
            "fallback_label": majority[field],
        }

    exact_match = exact_count / num_samples
    return {
        "num_samples": int(num_samples),
        "fallback_policy": "training_split_majority_per_field",
        "field_metrics": field_metrics,
        "summary": planner_summary(field_metrics, exact_match),
        "rule_audit": rule_audit,
    }


def load_qwen_comparison(path: Path) -> Dict[str, Dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload["raw_field_metrics"]
    corrected = payload["field_metrics"]
    table = payload["table2_semantic_planning"]
    return {
        "Qwen head only": planner_summary(raw, table["raw_exact_match"]),
        "Qwen + rules": planner_summary(corrected, table["exact_match"]),
    }


def write_csv(path: Path, comparison: Dict[str, Dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Planner", "Mean Accuracy", "Mean Macro-F1", "5-field Exact"])
        for name, metrics in comparison.items():
            writer.writerow(
                [
                    name,
                    f"{metrics['mean_accuracy']:.9f}",
                    f"{metrics['mean_macro_f1']:.9f}",
                    f"{metrics['exact_match']:.9f}",
                ]
            )


def write_markdown(path: Path, comparison: Dict[str, Dict[str, float]], rule_audit: Dict[str, Dict]) -> None:
    lines = [
        "# Planner Rule Baseline Audit",
        "",
        "| Planner | Mean Accuracy | Mean Macro-F1 | 5-field Exact |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in comparison.items():
        lines.append(
            f"| {name} | {metrics['mean_accuracy']:.3f} | "
            f"{metrics['mean_macro_f1']:.3f} | {metrics['exact_match']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Field | Rule Coverage | Covered Accuracy | Majority Fallback |",
            "|---|---:|---:|---|",
        ]
    )
    for field, audit in rule_audit.items():
        covered_accuracy = audit["covered_accuracy"]
        accuracy_text = "n/a" if covered_accuracy is None else f"{covered_accuracy:.3f}"
        lines.append(
            f"| {field} | {audit['coverage_rate']:.3f} | {accuracy_text} | {audit['fallback_label']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    majority, train_counts, train_samples = training_majorities(args.train_input, args.max_train_samples)
    rules = evaluate_rules_only(
        args.test_input,
        majority,
        args.max_test_samples,
        args.include_empty_labels,
    )
    comparison = load_qwen_comparison(args.qwen_metrics)
    comparison = {
        "Qwen head only": comparison["Qwen head only"],
        "Rules only": rules["summary"],
        "Qwen + rules": comparison["Qwen + rules"],
    }
    output = {
        "protocol": {
            "train_input": str(args.train_input),
            "test_input": str(args.test_input),
            "qwen_metrics": str(args.qwen_metrics),
            "rules_only_uncovered_field_policy": "training_split_majority_per_field",
            "include_empty_labels": bool(args.include_empty_labels),
        },
        "training": {
            "num_samples": int(train_samples),
            "label_counts": train_counts,
            "majority_labels": majority,
        },
        "rules_only": rules,
        "comparison": comparison,
    }
    (args.output_dir / "planner_rule_baseline_audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "planner_rule_baseline_comparison.csv", comparison)
    write_markdown(args.output_dir / "planner_rule_baseline_summary.md", comparison, rules["rule_audit"])
    print(json.dumps({"comparison": comparison, "rule_audit": rules["rule_audit"]}, indent=2))


if __name__ == "__main__":
    main()
