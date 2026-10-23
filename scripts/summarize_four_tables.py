"""Summarize LangDirectTraj four-table evaluation metrics.

The local goal loop writes one metrics.json under each table directory. This
helper keeps branch comparisons reproducible and avoids ad-hoc JSON readers
with mismatched field names.

Naming contract: Table 1 reports full-route grid-transition divergence as
`Route Transition JSD`. Under OD-chord conditioning, endpoint matching and
middle-route transition realism are separate questions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 6) -> float | None:
    x = _as_float(value)
    return None if x is None else round(x, digits)


def _metric_issue(table: str, metric: str, severity: str, value: Any, message: str) -> dict[str, Any]:
    return {
        "table": table,
        "metric": metric,
        "severity": severity,
        "value": value,
        "message": message,
    }


def _has_value(value: Any) -> bool:
    return value is not None


def _is_finite_number(value: Any) -> bool:
    x = _as_float(value)
    return x is not None and math.isfinite(x)


def _normalize_fields(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [field.strip() for field in value.split(",") if field.strip()]
    if isinstance(value, (list, tuple, set)):
        fields: list[str] = []
        for item in value:
            field = str(item).strip()
            if field:
                fields.append(field)
        return fields
    return []


def _is_focused_direction_diagnostic(table4: dict[str, Any]) -> bool:
    fields = _normalize_fields(table4.get("fields"))
    return len(fields) == 1 and fields[0] == "direction"


def _route_transition_warning_threshold(num_compared: Any) -> tuple[float, str]:
    """Return a sample-aware warning threshold for local OD-chord diagnostics.

    `Route Transition JSD` is an aggregate sparse-grid transition metric. On
    the official 256-sample local Porto smoke package, even OD-chord and
    real-vs-real baselines are high at small sample sizes. The paper metric is
    unchanged; this threshold only controls local abnormality warnings.
    """

    n = _as_float(num_compared)
    if n is None:
        return 0.25, "unknown_n_default"
    if n <= 16:
        return 0.40, "local_smoke_n_le_16"
    if n <= 32:
        return 0.35, "local_smoke_n_le_32"
    if n <= 64:
        return 0.30, "local_smoke_n_le_64"
    if n <= 128:
        return 0.25, "local_smoke_n_le_128"
    return 0.22, "larger_local_or_server"


def _table2_jsd_warning_threshold(metric: str, num_compared: Any) -> tuple[float, str]:
    """Return sample-aware JSD warning thresholds for local Table-2 diagnostics."""

    n = _as_float(num_compared)
    if metric == "connection_jsd":
        if n is None:
            return 0.30, "unknown_n_default"
        if n <= 16:
            return 0.55, "local_smoke_n_le_16"
        if n <= 32:
            return 0.48, "local_smoke_n_le_32"
        if n <= 64:
            return 0.35, "local_smoke_n_le_64"
        if n <= 128:
            return 0.26, "local_smoke_n_le_128"
        return 0.24, "larger_local_or_server"
    if metric == "link_jsd":
        if n is None:
            return 0.25, "unknown_n_default"
        if n <= 16:
            return 0.36, "local_smoke_n_le_16"
        if n <= 32:
            return 0.24, "local_smoke_n_le_32"
        if n <= 64:
            return 0.17, "local_smoke_n_le_64"
        if n <= 128:
            return 0.10, "local_smoke_n_le_128"
        return 0.09, "larger_local_or_server"
    return 0.30, "unknown_metric_default"


def _check_finite_metric(
    issues: list[dict[str, Any]],
    table: str,
    metric: str,
    value: Any,
    message: str,
    severity: str = "hard_fail",
) -> None:
    if _has_value(value) and not _is_finite_number(value):
        issues.append(_metric_issue(table, metric, severity, value, message))


def _check_required_finite_metric(
    issues: list[dict[str, Any]],
    table: str,
    metric: str,
    value: Any,
    message: str,
    severity: str = "hard_fail",
) -> None:
    if not _has_value(value):
        issues.append(_metric_issue(table, metric, severity, value, message))
        return
    _check_finite_metric(issues, table, metric, value, message, severity=severity)


def summarize_eval_root(eval_root: Path, dataset: str = "porto") -> dict[str, Any]:
    table1 = _load_json(eval_root / "table1" / dataset / "metrics.json") or {}
    table2_raw = _load_json(eval_root / "table2" / dataset / "metrics.json") or {}
    table2 = table2_raw.get("table2_semantic_planning", table2_raw)
    table3_raw = _load_json(eval_root / "table3" / dataset / "metrics.json") or {}
    table4_raw = _load_json(eval_root / "table4" / dataset / "metrics.json") or {}

    table3 = table3_raw.get("table3_instruction_following", table3_raw)
    table4 = table4_raw.get("table4_counterfactual_controllability", table4_raw)
    endpoint_marginal_jsd = table1.get("Endpoint Marginal JSD")
    endpoint_marginal_protocol = table1.get("Endpoint Marginal Protocol")
    route_transition_jsd = table1.get("Route Transition JSD")
    route_transition_protocol = table1.get("Route Transition Protocol")
    rejected_deprecated_route_metric_names = [
        key for key in ("Trip Error", "trip_error") if key in table1
    ]

    summary = {
        "eval_root": str(eval_root),
        "table1": {
            "num_compared": table1.get("num_compared"),
            "density_error": _round(table1.get("Density Error")),
            "endpoint_marginal_jsd": _round(endpoint_marginal_jsd),
            "endpoint_marginal_protocol": endpoint_marginal_protocol,
            "route_transition_jsd": _round(route_transition_jsd),
            "route_transition_protocol": route_transition_protocol,
            "od_pair_jsd": _round(table1.get("OD Pair JSD")),
            "length_error": _round(table1.get("Length Error")),
            "rejected_deprecated_route_metric_names": rejected_deprecated_route_metric_names,
        },
        "table2": {
            "num_samples": table2.get("num_samples", table2_raw.get("num_samples")),
            "mean_accuracy": _round(table2.get("mean_accuracy")),
            "mean_macro_f1": _round(table2.get("mean_macro_f1")),
            "exact_match": _round(table2.get("exact_match")),
            "raw_exact_match": _round(table2.get("raw_exact_match")),
            "distance_accuracy": _round(table2.get("distance_accuracy")),
            "direction_accuracy": _round(table2.get("direction_accuracy")),
            "curvature_accuracy": _round(table2.get("curvature_accuracy")),
            "progress_accuracy": _round(table2.get("progress_accuracy")),
            "reference_adherence_accuracy": _round(table2.get("reference_adherence_accuracy")),
            "distance_f1": _round(table2.get("distance_f1")),
            "direction_f1": _round(table2.get("direction_f1")),
            "curvature_f1": _round(table2.get("curvature_f1")),
            "progress_f1": _round(table2.get("progress_f1")),
            "reference_adherence_f1": _round(table2.get("reference_adherence_f1")),
        },
        "table3": {
            "distance_f1": _round(table3.get("distance_f1")),
            "direction_f1": _round(table3.get("direction_f1")),
            "curvature_f1": _round(table3.get("curvature_f1")),
            "progress_f1": _round(table3.get("progress_f1")),
            "reference_adherence_f1": _round(table3.get("reference_adherence_f1")),
            "mean_f1": _round(
                table3.get("mean_instruction_following_f1", table3.get("mean_f1"))
            ),
        },
        "table4": {
            "target_effect": _round(
                table4.get("counterfactual_target_effect", table4.get("target_effect"))
            ),
            "success_rate": _round(
                table4.get("counterfactual_success_rate", table4.get("success_rate"))
            ),
            "collateral_drift": _round(
                table4.get("independent_collateral_drift", table4.get("collateral_drift"))
            ),
            "raw_collateral_drift": _round(
                table4.get("raw_collateral_drift", table4.get("collateral_drift"))
            ),
            "num_pairs": table4_raw.get("num_pairs", table4.get("num_pairs")),
            "fields": table4_raw.get("fields", table4.get("fields")),
            "transition_filter": table4_raw.get("transition_filter"),
        },
    }
    summary["diagnosis"] = assess_summary(summary)
    return summary


def assess_summary(summary: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    table1 = summary["table1"]
    table2 = summary["table2"]
    table3 = summary["table3"]
    table4 = summary["table4"]

    density = table1.get("density_error")
    endpoint_marginal = table1.get("endpoint_marginal_jsd")
    route_transition = table1.get("route_transition_jsd")
    route_transition_threshold, route_transition_threshold_source = _route_transition_warning_threshold(
        table1.get("num_compared")
    )
    length = table1.get("length_error")
    rejected_deprecated_route_metric_names = (
        table1.get("rejected_deprecated_route_metric_names")
        or table1.get("legacy_route_metric_aliases")
        or table1.get("deprecated_fields")
        or []
    )
    if rejected_deprecated_route_metric_names:
        issues.append(
            _metric_issue(
                "table1",
                "rejected_deprecated_route_metric_names",
                "hard_fail",
                rejected_deprecated_route_metric_names,
                (
                    "Deprecated Table 1 route metric alias detected. The official metric "
                    "name is Route Transition JSD for full-route grid-transition divergence."
                ),
            )
        )
    for metric in ("density_error", "endpoint_marginal_jsd", "route_transition_jsd", "length_error"):
        _check_required_finite_metric(
            issues,
            "table1",
            metric,
            table1.get(metric),
            "Table 1 metric is non-finite; the distribution evaluation is not trustworthy.",
        )
    if table1.get("endpoint_marginal_protocol") != "endpoint_marginal_js":
        issues.append(
            _metric_issue(
                "table1",
                "endpoint_marginal_protocol",
                "hard_fail",
                table1.get("endpoint_marginal_protocol"),
                "Endpoint Marginal JSD must use the endpoint_marginal_js protocol.",
            )
        )
    if table1.get("route_transition_protocol") != "grid_transition_js":
        issues.append(
            _metric_issue(
                "table1",
                "route_transition_protocol",
                "hard_fail",
                table1.get("route_transition_protocol"),
                "Route Transition JSD must use full-route grid_transition_js under OD-chord conditioning.",
            )
        )
    if _has_value(route_transition) and route_transition <= 1e-9 and (
        (_has_value(density) and density > 1e-6) or (_has_value(length) and length > 1e-6)
    ):
        issues.append(
            _metric_issue(
                "table1",
                "route_transition_jsd",
                "hard_fail",
                route_transition,
                "Route Transition JSD is degenerate: it is ~0 while Density/Length Error is non-zero.",
            )
        )
    if _has_value(route_transition) and route_transition > route_transition_threshold:
        issues.append(
            _metric_issue(
                "table1",
                "route_transition_jsd",
                "warning",
                route_transition,
                (
                    "Route transition distribution is high for this evaluation size "
                    f"(sample-aware threshold={route_transition_threshold:.3f}, "
                    f"source={route_transition_threshold_source})."
                ),
            )
        )
    if _has_value(density) and density > 0.2:
        issues.append(
            _metric_issue("table1", "density_error", "warning", density, "Density Error is high.")
        )
    if _has_value(length) and length > 0.2:
        issues.append(
            _metric_issue("table1", "length_error", "warning", length, "Length Error is high.")
        )

    for metric in ("mean_accuracy", "mean_macro_f1", "exact_match"):
        value = table2.get(metric)
        _check_required_finite_metric(
            issues,
            "table2",
            metric,
            value,
            "Table 2 metric is non-finite; semantic-planner evaluation is not trustworthy.",
        )
        if _has_value(value) and value < 0.5:
            issues.append(
                _metric_issue(
                    "table2",
                    metric,
                    "warning",
                    value,
                    "Semantic-planner score is low; check Qwen planner checkpoint or instruction labels.",
                )
            )
    for metric in (
        "distance_f1",
        "direction_f1",
        "curvature_f1",
        "progress_f1",
        "reference_adherence_f1",
    ):
        _check_required_finite_metric(
            issues,
            "table2",
            metric,
            table2.get(metric),
            "Table 2 field F1 is non-finite; semantic-planner evaluation is incomplete.",
        )

    for metric in (
        "distance_f1",
        "direction_f1",
        "curvature_f1",
        "progress_f1",
        "reference_adherence_f1",
    ):
        value = table3.get(metric)
        _check_required_finite_metric(
            issues,
            "table3",
            metric,
            value,
            "Instruction-following F1 is non-finite; label or geometry decoding may be malformed.",
        )
        if _has_value(value) and value < 0.3:
            issues.append(
                _metric_issue(
                    "table3",
                    metric,
                    "warning",
                    value,
                    "Instruction-following F1 is below the local minimum target.",
                )
            )
    mean_f1 = table3.get("mean_f1")
    _check_required_finite_metric(
        issues,
        "table3",
        "mean_f1",
        mean_f1,
        "Mean instruction-following F1 is non-finite; Table 3 is not trustworthy.",
    )
    if _has_value(mean_f1) and mean_f1 <= 0.46:
        issues.append(
            _metric_issue(
                "table3",
                "mean_f1",
                "warning",
                mean_f1,
                "Mean instruction-following F1 has not improved beyond the current baseline.",
            )
        )

    num_pairs = table4.get("num_pairs")
    if isinstance(num_pairs, str):
        try:
            num_pairs = int(num_pairs)
        except ValueError:
            num_pairs = None
    if _has_value(num_pairs) and num_pairs < 20:
        issues.append(
            _metric_issue(
                "table4",
                "num_pairs",
                "hard_fail",
                num_pairs,
                "Counterfactual evaluation has too few valid pairs for a reliable local conclusion.",
            )
        )
    collateral = table4.get("collateral_drift")
    focused_direction_diagnostic = _is_focused_direction_diagnostic(table4)
    if focused_direction_diagnostic and not _is_finite_number(collateral):
        issues.append(
            _metric_issue(
                "table4",
                "independent_collateral_drift",
                "warning",
                collateral,
                (
                    "Independent collateral is not applicable for direction-only "
                    "focused diagnostics because structurally independent fields "
                    "can be empty. Use raw_collateral_drift for this diagnostic; "
                    "official multi-field Table 4 still requires finite independent collateral."
                ),
            )
        )
    else:
        _check_required_finite_metric(
            issues,
            "table4",
            "independent_collateral_drift",
            collateral,
            "Independent counterfactual collateral drift is non-finite; Table 4 guardrail cannot be assessed.",
        )
    if _has_value(collateral) and collateral > 0.12:
        issues.append(
            _metric_issue(
                "table4",
                "independent_collateral_drift",
                "warning",
                collateral,
                "Independent counterfactual collateral drift is above the local guardrail.",
            )
        )
    raw_collateral = table4.get("raw_collateral_drift")
    _check_required_finite_metric(
        issues,
        "table4",
        "raw_collateral_drift",
        raw_collateral,
        "Raw counterfactual collateral drift is non-finite; Table 4 is not trustworthy.",
    )
    if _has_value(raw_collateral) and raw_collateral > 0.25:
        issues.append(
            _metric_issue(
                "table4",
                "raw_collateral_drift",
                "warning",
                raw_collateral,
                "Raw all-field collateral drift is high; inspect structurally coupled fields.",
            )
        )
    success = table4.get("success_rate")
    _check_required_finite_metric(
        issues,
        "table4",
        "success_rate",
        success,
        "Counterfactual success rate is non-finite; Table 4 is not trustworthy.",
    )
    if _has_value(success) and success < 0.7:
        issues.append(
            _metric_issue(
                "table4",
                "success_rate",
                "warning",
                success,
                "Counterfactual success rate is below the local guardrail.",
            )
        )
    target_effect = table4.get("target_effect")
    _check_required_finite_metric(
        issues,
        "table4",
        "target_effect",
        target_effect,
        "Counterfactual target effect is missing or non-finite; Table 4 is not trustworthy.",
    )
    if not _has_value(table4.get("num_pairs")):
        issues.append(
            _metric_issue(
                "table4",
                "num_pairs",
                "hard_fail",
                table4.get("num_pairs"),
                "Counterfactual evaluation did not report num_pairs; Table 4 may be missing.",
            )
        )

    hard_fail_count = sum(1 for issue in issues if issue["severity"] == "hard_fail")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    status = "hard_fail" if hard_fail_count else ("warning" if warning_count else "ok")
    return {
        "status": status,
        "hard_fail_count": hard_fail_count,
        "warning_count": warning_count,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_roots", nargs="+", type=Path)
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of compact text.")
    parser.add_argument("--fail-on-hard", action="store_true", help="Exit non-zero when any hard-fail issue is found.")
    parser.add_argument("--fail-on-warning", action="store_true", help="Exit non-zero when any warning or hard-fail issue is found.")
    args = parser.parse_args()

    summaries = [summarize_eval_root(root, dataset=args.dataset) for root in args.eval_roots]
    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return

    for summary in summaries:
        table1 = summary["table1"]
        table2 = summary["table2"]
        table3 = summary["table3"]
        table4 = summary["table4"]
        diagnosis = summary["diagnosis"]
        print(
            "[eval] "
            f"{summary['eval_root']} | "
            f"T1 density={table1.get('density_error')} route_jsd={table1.get('route_transition_jsd')} "
            f"length={table1.get('length_error')} | "
            f"T2 jaccard={table2.get('jaccard')} link_jsd={table2.get('link_jsd')} | "
            f"T3 mean_f1={table3.get('mean_f1')} ref_f1={table3.get('reference_adherence_f1')} | "
            f"T4 effect={table4.get('target_effect')} success={table4.get('success_rate')} "
            f"collateral={table4.get('collateral_drift')} pairs={table4.get('num_pairs')} | "
            f"status={diagnosis['status']} hard={diagnosis['hard_fail_count']} warn={diagnosis['warning_count']}"
        )
        for issue in diagnosis["issues"]:
            print(
                f"  [{issue['severity']}] {issue['table']}.{issue['metric']}="
                f"{issue['value']}: {issue['message']}"
            )

    has_hard = any(summary["diagnosis"]["hard_fail_count"] > 0 for summary in summaries)
    has_warning = any(summary["diagnosis"]["warning_count"] > 0 for summary in summaries)
    if args.fail_on_hard and has_hard:
        sys.exit(2)
    if args.fail_on_warning and (has_hard or has_warning):
        sys.exit(3)


if __name__ == "__main__":
    main()
