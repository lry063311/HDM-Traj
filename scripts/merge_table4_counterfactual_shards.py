#!/usr/bin/env python
"""Merge source-sharded Table IV counterfactual metrics.

This merger is intentionally conservative: it only aggregates count-weighted
mean metrics that remain valid across disjoint source_sample_id shards. Exact
percentiles require raw per-pair values and are therefore not reconstructed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MEAN_KEYS = (
    "counterfactual_target_effect",
    "counterfactual_success_rate",
    "collateral_drift",
    "raw_collateral_drift",
    "independent_collateral_drift",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge source-sharded Table IV metrics.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    weight = 0
    for row in rows:
        n = int(row.get("num_pairs", 0))
        value = row.get("table4_counterfactual_controllability", {}).get(key)
        if value is None:
            continue
        total += float(value) * n
        weight += n
    return total / weight if weight else float("nan")


def merge_by_field(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = sorted({field for row in rows for field in row.get("by_field", {})})
    merged: dict[str, Any] = {}
    for field in fields:
        field_rows = []
        for row in rows:
            payload = row.get("by_field", {}).get(field)
            if not payload:
                continue
            field_rows.append(payload)
        n_total = sum(int(item.get("num_pairs", 0)) for item in field_rows)
        if n_total <= 0:
            continue
        merged[field] = {
            "num_pairs": n_total,
            "counterfactual_success_rate": sum(
                float(item.get("counterfactual_success_rate", 0.0)) * int(item.get("num_pairs", 0))
                for item in field_rows
            )
            / n_total,
        }
        for source_key, target_key in (
            ("counterfactual_target_effect_z", "counterfactual_target_effect"),
            ("collateral_drift_z", "collateral_drift"),
            ("independent_collateral_drift_z", "independent_collateral_drift"),
            ("signed_target_metric_delta_z", "signed_target_metric_delta_z"),
        ):
            total = 0.0
            weight = 0
            for item in field_rows:
                stat = item.get(source_key, {})
                if "mean" not in stat:
                    continue
                n = int(item.get("num_pairs", 0))
                total += float(stat["mean"]) * n
                weight += n
            if weight:
                merged[field][target_key] = total / weight
    return merged


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise ValueError("No shard metrics provided.")

    num_pairs = sum(int(row.get("num_pairs", 0)) for row in rows)
    merged = {
        "num_pairs": num_pairs,
        "table4_counterfactual_controllability": {
            key: weighted_mean(rows, key) for key in MEAN_KEYS
        },
        "counterfactual_success_rate": weighted_mean(rows, "counterfactual_success_rate"),
        "by_field": merge_by_field(rows),
        "shards": [str(path) for path in args.inputs],
        "merge_note": (
            "Count-weighted merge over disjoint source_sample_id shards. "
            "Percentiles are intentionally not reconstructed."
        ),
    }
    first = rows[0]
    for key in (
        "dataset",
        "input",
        "checkpoint",
        "target_key",
        "fields",
        "success_margin_z",
        "selection_policy",
        "selection_hybrid",
        "qwen_semantic_conditioning",
    ):
        if key in first:
            merged[key] = first[key]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(merged, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
