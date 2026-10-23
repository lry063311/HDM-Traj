from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ORDER = ("train", "val", "test")
OFFICIAL_COUNTERFACTUAL_FIELDS = [
    "distance",
    "direction",
    "curvature",
    "progress",
    "reference_adherence",
]
SCALAR_EFFECT_ONLY_TOKEN_MATCH_OVERRIDES = [
    "distance=0",
    "progress=0",
]
SCALAR_EFFECT_ONLY_MIN_EFFECT_OVERRIDES = [
    "distance=0.05",
    "progress=0.05",
]


def run_command(args: List[str], dry_run: bool = False) -> None:
    print(" ".join(args), flush=True)
    if dry_run:
        return
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    subprocess.run(args, cwd=REPO_ROOT, check=True, env=env)


def announce_stage(index: int, total: int, message: str, output_path: Path | None = None) -> None:
    suffix = f" -> {output_path}" if output_path is not None else ""
    print(f"\n[controllability {index}/{total}] {message}{suffix}", flush=True)


def should_run(path: Path, overwrite: bool) -> bool:
    return overwrite or not path.exists()


def should_run_all(paths: Iterable[Path], overwrite: bool) -> bool:
    return overwrite or any(not Path(path).exists() for path in paths)


def split_path(path: Path, split: str) -> Path:
    return path.with_name(f"{path.stem}.{split}{path.suffix}")


def parse_fields(raw: Iterable[str]) -> List[str]:
    fields: List[str] = []
    for item in raw:
        for field in str(item).replace(",", " ").split():
            if field and field not in fields:
                fields.append(field)
    return fields


def merge_field_overrides(user_values: Iterable[str], default_values: Iterable[str]) -> List[str]:
    """Merge field=value CLI overrides with user values taking precedence."""

    merged: dict[str, str] = {}
    order: list[str] = []
    for item in [*default_values, *user_values]:
        text = str(item).strip()
        if not text:
            continue
        field = text.split("=", 1)[0].strip()
        if not field:
            continue
        if field not in merged:
            order.append(field)
        merged[field] = text
    return [merged[field] for field in order]


def summarize_transition_diagnostics(
    path: Path,
    output_json: Path,
    output_csv: Path,
    min_success_rate: float,
    min_effect_z: float,
    min_token_match_rate: float,
    min_pairs: int,
    dry_run: bool = False,
) -> dict:
    if dry_run:
        summary = {
            "transition_diagnostics_json": str(path),
            "summary_json": str(output_json),
            "summary_csv": str(output_csv),
            "dry_run": True,
            "thresholds": {
                "min_success_rate": float(min_success_rate),
                "min_effect_z": float(min_effect_z),
                "min_token_match_rate": float(min_token_match_rate),
                "min_pairs": int(min_pairs),
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    kept_by_field: Counter[str] = Counter()
    rejected_by_field: Counter[str] = Counter()
    for key, stats in sorted(payload.get("by_transition", {}).items()):
        field = key.split(":", 1)[0]
        num_pairs = int(stats.get("num_pairs", 0))
        success_rate = float(stats.get("pseudo_target_success_rate", 0.0))
        effect_mean = float(stats.get("pseudo_target_effect_z", {}).get("mean", 0.0))
        collateral_stats = stats.get("independent_collateral_drift_z", stats.get("collateral_drift_z", {}))
        collateral_mean = float(collateral_stats.get("mean", 0.0)) if isinstance(collateral_stats, dict) else 0.0
        token_match_rate = float(stats.get("predicted_target_match_rate", 1.0))
        keep = (
            num_pairs >= int(min_pairs)
            and success_rate >= float(min_success_rate)
            and effect_mean >= float(min_effect_z)
            and token_match_rate >= float(min_token_match_rate)
        )
        if keep:
            kept_by_field[field] += 1
        else:
            rejected_by_field[field] += 1
        rows.append(
            {
                "transition": key,
                "field": field,
                "num_pairs": num_pairs,
                "success_rate": success_rate,
                "effect_z_mean": effect_mean,
                "independent_collateral_z_mean": collateral_mean,
                "token_match_rate": token_match_rate,
                "kept_for_training": bool(keep),
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "transition",
                "field",
                "num_pairs",
                "success_rate",
                "effect_z_mean",
                "independent_collateral_z_mean",
                "token_match_rate",
                "kept_for_training",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "transition_diagnostics_json": str(path),
        "summary_csv": str(output_csv),
        "thresholds": {
            "min_success_rate": float(min_success_rate),
            "min_effect_z": float(min_effect_z),
            "min_token_match_rate": float(min_token_match_rate),
            "min_pairs": int(min_pairs),
        },
        "num_transitions": int(len(rows)),
        "num_kept_for_training": int(sum(1 for row in rows if row["kept_for_training"])),
        "kept_by_field": dict(sorted(kept_by_field.items())),
        "rejected_by_field": dict(sorted(rejected_by_field.items())),
        "kept_transitions": [row for row in rows if row["kept_for_training"]],
        "rejected_transitions": [row for row in rows if not row["kept_for_training"]],
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def add_transition_coverage_check(
    summary: dict,
    fields: Iterable[str],
    min_kept_per_field: int,
) -> dict:
    """Attach a hard coverage check for official five-field controllability data."""

    fields = list(fields)
    kept_by_field = {
        str(field): int(count)
        for field, count in dict(summary.get("kept_by_field", {})).items()
    }
    min_kept_per_field = int(min_kept_per_field)
    failed_fields = {
        field: int(kept_by_field.get(field, 0))
        for field in fields
        if min_kept_per_field > 0 and int(kept_by_field.get(field, 0)) < min_kept_per_field
    }
    check = {
        "enabled": bool(min_kept_per_field > 0),
        "min_kept_transitions_per_field": min_kept_per_field,
        "fields": fields,
        "kept_by_field": {field: int(kept_by_field.get(field, 0)) for field in fields},
        "failed_fields": failed_fields,
        "passed": not failed_fields,
    }
    summary["transition_coverage_check"] = check
    summary["hard_fail_count"] = int(len(failed_fields))
    return check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the server-scale controllability data package: balanced "
            "counterfactual instruction data, pseudo-target filtering, token "
            "geometry calibration, and transition diagnostics."
        )
    )
    parser.add_argument("--dataset", default="porto")
    parser.add_argument(
        "--instruction-input",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "local_porto_od_chord_256_verify"
        / "instruction_data"
        / "porto_instruction_samples.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "controllability_data" / "porto",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--fields",
        nargs="+",
        default=OFFICIAL_COUNTERFACTUAL_FIELDS,
        help=(
            "Counterfactual fields for the official Table-4 package. The "
            "porto_od_chord_v1 default covers all five instruction fields."
        ),
    )
    parser.add_argument("--variants-per-field", type=int, default=2)
    parser.add_argument(
        "--direction-target-policy",
        choices=["legacy", "transition_aware"],
        default="transition_aware",
        help="Direction counterfactual pseudo-target policy passed to build_counterfactual_instruction_dataset.",
    )
    parser.add_argument(
        "--counterfactual-geometry-policy",
        choices=["field_isolated", "changed_field_last", "composed"],
        default="field_isolated",
        help=(
            "Pseudo-target composition policy passed to build_counterfactual_instruction_dataset. "
            "Use field_isolated for the official Table-4 controllability protocol."
        ),
    )
    parser.add_argument("--include-original", action="store_true", default=True)
    parser.add_argument("--no-include-original", action="store_false", dest="include_original")
    parser.add_argument("--balance-fields", action="store_true")
    parser.add_argument("--balance-target", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-len", type=int, default=128)
    parser.add_argument(
        "--direction-metric-mode",
        choices=["fixed", "auto", "signed_lr"],
        default="signed_lr",
        help="Direction metric mode for token-geometry calibration.",
    )
    parser.add_argument(
        "--curvature-order-mode",
        choices=["none", "ordered"],
        default="ordered",
        help="Curvature calibration center order. Official porto_od_chord_v1 uses smooth < moderate < winding.",
    )
    parser.add_argument("--calibration-max-samples", type=int, default=0)
    parser.add_argument("--transition-max-records", type=int, default=0)
    parser.add_argument("--transition-max-pairs", type=int, default=0)
    parser.add_argument("--transition-min-success-rate", type=float, default=0.6)
    parser.add_argument("--transition-min-effect-z", type=float, default=0.05)
    parser.add_argument(
        "--transition-min-token-match-rate",
        type=float,
        default=0.0,
        help=(
            "Minimum rate at which pseudo targets are decoded as the requested target token. "
            "Keep 0.0 for exploratory diagnostics; raise it for training-quality packages."
        ),
    )
    parser.add_argument("--transition-min-pairs", type=int, default=2)
    parser.add_argument(
        "--transition-min-kept-per-field",
        type=int,
        default=2,
        help=(
            "Hard coverage gate for official Table-4 controllability packages. "
            "Each requested field must keep at least this many reliable "
            "counterfactual transitions after filtering; set 0 to disable."
        ),
    )
    parser.add_argument("--min-length-ratio", type=float, default=0.60)
    parser.add_argument("--max-length-ratio", type=float, default=1.80)
    parser.add_argument("--max-curvature-ratio", type=float, default=4.00)
    parser.add_argument("--max-mean-reference-distance-m", type=float, default=250.0)
    parser.add_argument(
        "--field-max-mean-reference-distance-m",
        nargs="+",
        default=["direction=600"],
        help="Optional field-specific pseudo-target reference gates.",
    )
    parser.add_argument("--max-endpoint-drift-m", type=float, default=5.0)
    parser.add_argument("--max-step-m", type=float, default=300.0)
    parser.add_argument(
        "--calibration-aware-filter",
        action="store_true",
        default=True,
        help=(
            "Run the official second-stage pseudo-target filter against a bootstrap token-geometry "
            "calibration before writing the final .filtered.jsonl."
        ),
    )
    parser.add_argument("--no-calibration-aware-filter", action="store_false", dest="calibration_aware_filter")
    parser.add_argument("--min-calibration-effect-z", type=float, default=0.0)
    parser.add_argument(
        "--min-calibration-token-match-rate",
        type=float,
        default=1.0,
        help="Per-record target-token match gate for the official second-stage filter.",
    )
    parser.add_argument(
        "--max-calibration-collateral-z",
        type=float,
        default=0.8,
        help="Per-record non-target metric-drift gate for the official second-stage filter. Use 0 to disable.",
    )
    parser.add_argument(
        "--field-min-calibration-effect-z",
        nargs="*",
        default=[],
        help=(
            "Field-specific calibration effect gates. Defaults add distance=0.05 "
            "and progress=0.05 because scalar OD-preserving edits are judged by "
            "Table-4 effect rather than exact token-bin recovery."
        ),
    )
    parser.add_argument(
        "--field-min-calibration-token-match-rate",
        nargs="*",
        default=[],
        help=(
            "Field-specific token-match gates. Defaults add distance=0 and "
            "progress=0 so useful scalar counterfactuals are not dropped only "
            "because calibration bins are coarse."
        ),
    )
    parser.add_argument("--field-max-calibration-collateral-z", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional base-sample split manifest. When set, train/val/test counterfactual files are emitted.",
    )
    parser.add_argument(
        "--no-split-outputs",
        action="store_true",
        help="Disable split-specific Stage-2 outputs even if a split manifest exists.",
    )
    parser.add_argument(
        "--storage-format",
        choices=["expanded", "compact"],
        default="expanded",
        help="expanded keeps legacy JSONL outputs; compact additionally writes .npy-backed final filtered metadata.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def filter_command(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    token_calibration_path: Path | None = None,
    compact_split: str | None = None,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "langdirecttraj.data.filter_pseudo_target_dataset",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--summary-output",
        str(summary_path),
        "--min-length-ratio",
        str(args.min_length_ratio),
        "--max-length-ratio",
        str(args.max_length_ratio),
        "--max-curvature-ratio",
        str(args.max_curvature_ratio),
        "--max-mean-reference-distance-m",
        str(args.max_mean_reference_distance_m),
        "--field-max-mean-reference-distance-m",
        *[str(item) for item in args.field_max_mean_reference_distance_m],
        "--max-endpoint-drift-m",
        str(args.max_endpoint_drift_m),
        "--max-step-m",
        str(args.max_step_m),
    ]
    if token_calibration_path is not None:
        cmd.extend(
            [
                "--token-calibration-path",
                str(token_calibration_path),
                "--target-len",
                str(args.target_len),
                "--min-calibration-effect-z",
                str(args.min_calibration_effect_z),
                "--min-calibration-token-match-rate",
                str(args.min_calibration_token_match_rate),
                "--max-calibration-collateral-z",
                str(args.max_calibration_collateral_z),
            ]
        )
        if args.field_min_calibration_effect_z:
            cmd.extend(["--field-min-calibration-effect-z", *[str(item) for item in args.field_min_calibration_effect_z]])
        if args.field_min_calibration_token_match_rate:
            cmd.extend(
                [
                    "--field-min-calibration-token-match-rate",
                    *[str(item) for item in args.field_min_calibration_token_match_rate],
                ]
            )
        if args.field_max_calibration_collateral_z:
            cmd.extend(
                [
                    "--field-max-calibration-collateral-z",
                    *[str(item) for item in args.field_max_calibration_collateral_z],
                ]
            )
    if str(getattr(args, "storage_format", "expanded")) == "compact" and compact_split:
        cmd.extend(
            [
                "--compact-output-dir",
                str(args.output_dir / "compact"),
                "--compact-dataset",
                str(args.dataset),
                "--compact-split",
                str(compact_split),
                "--compact-prefix",
                "filtered",
            ]
        )
    return cmd


def calibration_command(args: argparse.Namespace, input_path: Path, output_path: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "langdirecttraj.data.build_token_geometry_calibration",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--target-key",
        "pseudo_trajectory",
        "--target-len",
        str(args.target_len),
        "--max-samples",
        str(args.calibration_max_samples),
        "--active-field-only",
        "--direction-metric-mode",
        args.direction_metric_mode,
        "--reference-order-mode",
        "ordered",
        "--progress-order-mode",
        "ordered",
        "--curvature-order-mode",
        args.curvature_order_mode,
    ]


def main() -> None:
    args = parse_args()
    fields = parse_fields(args.fields)
    args.field_min_calibration_effect_z = merge_field_overrides(
        args.field_min_calibration_effect_z,
        SCALAR_EFFECT_ONLY_MIN_EFFECT_OVERRIDES,
    )
    args.field_min_calibration_token_match_rate = merge_field_overrides(
        args.field_min_calibration_token_match_rate,
        SCALAR_EFFECT_ONLY_TOKEN_MATCH_OVERRIDES,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl = output_dir / f"{args.dataset}_counterfactual_instruction_samples.jsonl"
    raw_summary = output_dir / f"{args.dataset}_counterfactual_instruction_summary.json"
    prefiltered_jsonl = output_dir / f"{args.dataset}_counterfactual_instruction_samples.prefiltered.jsonl"
    prefiltered_summary = output_dir / f"{args.dataset}_counterfactual_instruction_samples.prefiltered_summary.json"
    bootstrap_calibration_json = output_dir / f"{args.dataset}_token_geometry_calibration.prefiltered.json"
    filtered_jsonl = output_dir / f"{args.dataset}_counterfactual_instruction_samples.filtered.jsonl"
    filtered_summary = output_dir / f"{args.dataset}_counterfactual_instruction_samples.filtered_summary.json"
    calibration_json = output_dir / f"{args.dataset}_token_geometry_calibration.json"
    transition_json = output_dir / f"{args.dataset}_counterfactual_transition_diagnostics.json"
    transition_summary_json = output_dir / f"{args.dataset}_counterfactual_transition_training_summary.json"
    transition_summary_csv = output_dir / f"{args.dataset}_counterfactual_transition_training_summary.csv"
    manifest_json = output_dir / f"{args.dataset}_controllability_data_manifest.json"
    total_stages = 7
    split_enabled = bool(args.split_manifest) and not bool(args.no_split_outputs)
    raw_split_jsonl = {split: split_path(raw_jsonl, split) for split in SPLIT_ORDER}
    prefiltered_split_jsonl = {split: split_path(prefiltered_jsonl, split) for split in SPLIT_ORDER}
    prefiltered_split_summary = {
        split: split_path(prefiltered_summary.with_suffix(".json"), split)
        for split in SPLIT_ORDER
    }
    filtered_split_jsonl = {split: split_path(filtered_jsonl, split) for split in SPLIT_ORDER}
    filtered_split_summary = {
        split: split_path(filtered_summary.with_suffix(".json"), split)
        for split in SPLIT_ORDER
    }
    calibration_source_jsonl = prefiltered_split_jsonl["train"] if split_enabled else prefiltered_jsonl
    transition_source_jsonl = filtered_split_jsonl["train"] if split_enabled else filtered_jsonl

    raw_gate_path = raw_split_jsonl["train"] if split_enabled else raw_jsonl
    raw_needed = (
        should_run_all(raw_split_jsonl.values(), args.overwrite)
        if split_enabled
        else should_run(raw_gate_path, args.overwrite)
    )
    if raw_needed:
        announce_stage(1, total_stages, "build counterfactual instruction dataset", raw_gate_path)
        cmd = [
            sys.executable,
            "-m",
            "langdirecttraj.data.build_counterfactual_instruction_dataset",
            "--dataset",
            args.dataset,
            "--input",
            str(args.instruction_input),
            "--output-dir",
            str(output_dir),
            "--max-samples",
            str(args.max_samples),
            "--fields",
            *fields,
            "--variants-per-field",
            str(args.variants_per_field),
            "--direction-target-policy",
            args.direction_target_policy,
            "--counterfactual-geometry-policy",
            args.counterfactual_geometry_policy,
            "--seed",
            str(args.seed),
            "--streaming-write",
        ]
        if args.include_original:
            cmd.append("--include-original")
        if split_enabled:
            cmd.extend(["--split-manifest", str(args.split_manifest), "--split-only-output"])
        if args.balance_fields:
            cmd.append("--balance-fields")
        if args.balance_target > 0:
            cmd.extend(["--balance-target", str(args.balance_target)])
        run_command(cmd, dry_run=args.dry_run)
    else:
        announce_stage(1, total_stages, "skip existing counterfactual instruction dataset", raw_gate_path)

    prefilter_gate_path = prefiltered_split_jsonl["train"] if split_enabled else prefiltered_jsonl
    prefilter_needed = (
        should_run_all(prefiltered_split_jsonl.values(), args.overwrite)
        if split_enabled
        else should_run(prefilter_gate_path, args.overwrite)
    )
    if prefilter_needed:
        if not split_enabled:
            announce_stage(2, total_stages, "prefilter pseudo-target dataset", prefiltered_jsonl)
            run_command(
                filter_command(args, raw_jsonl, prefiltered_jsonl, prefiltered_summary),
                dry_run=args.dry_run,
            )
        else:
            for split in SPLIT_ORDER:
                announce_stage(2, total_stages, f"prefilter pseudo-target dataset ({split})", prefiltered_split_jsonl[split])
                run_command(
                    filter_command(
                        args,
                        raw_split_jsonl[split],
                        prefiltered_split_jsonl[split],
                        prefiltered_split_summary[split],
                    ),
                    dry_run=args.dry_run,
                )
    else:
        announce_stage(2, total_stages, "skip existing prefiltered pseudo-target dataset", prefiltered_jsonl)

    if should_run(bootstrap_calibration_json, args.overwrite):
        announce_stage(3, total_stages, "build bootstrap token-geometry calibration", bootstrap_calibration_json)
        run_command(
            calibration_command(args, calibration_source_jsonl, bootstrap_calibration_json),
            dry_run=args.dry_run,
        )
    else:
        announce_stage(3, total_stages, "skip existing bootstrap token-geometry calibration", bootstrap_calibration_json)

    filtered_gate_path = filtered_split_jsonl["train"] if split_enabled else filtered_jsonl
    filter_needed = (
        should_run_all(filtered_split_jsonl.values(), args.overwrite)
        if split_enabled
        else should_run(filtered_gate_path, args.overwrite)
    )
    if filter_needed:
        calibration_path = bootstrap_calibration_json if args.calibration_aware_filter else None
        if not split_enabled:
            announce_stage(4, total_stages, "calibration-aware pseudo-target filter", filtered_jsonl)
            filter_input = raw_jsonl
            run_command(
                filter_command(
                    args,
                    filter_input,
                    filtered_jsonl,
                    filtered_summary,
                    calibration_path,
                    compact_split="all",
                ),
                dry_run=args.dry_run,
            )
        else:
            for split in SPLIT_ORDER:
                announce_stage(4, total_stages, f"calibration-aware pseudo-target filter ({split})", filtered_split_jsonl[split])
                run_command(
                    filter_command(
                        args,
                        raw_split_jsonl[split],
                        filtered_split_jsonl[split],
                        filtered_split_summary[split],
                        calibration_path,
                        compact_split=split,
                    ),
                    dry_run=args.dry_run,
                )
    else:
        announce_stage(4, total_stages, "skip existing filtered pseudo-target dataset", filtered_jsonl)

    if should_run(calibration_json, args.overwrite):
        announce_stage(5, total_stages, "final token-geometry calibration", calibration_json)
        if args.calibration_aware_filter:
            print(f"copy {bootstrap_calibration_json} {calibration_json}", flush=True)
            if not args.dry_run:
                shutil.copyfile(bootstrap_calibration_json, calibration_json)
        else:
            run_command(
                calibration_command(args, filtered_jsonl, calibration_json),
                dry_run=args.dry_run,
            )
    else:
        announce_stage(5, total_stages, "skip existing final token-geometry calibration", calibration_json)

    if should_run(transition_json, args.overwrite):
        announce_stage(6, total_stages, "diagnose counterfactual transitions", transition_json)
        run_command(
            [
                sys.executable,
                "-m",
                "langdirecttraj.data.diagnose_counterfactual_transitions",
                "--input",
                str(transition_source_jsonl),
                "--output",
                str(transition_json),
                "--target-key",
                "pseudo_trajectory",
                "--target-len",
                str(args.target_len),
                "--max-records",
                str(args.transition_max_records),
                "--max-pairs",
                str(args.transition_max_pairs),
                "--fields",
                *fields,
                "--token-calibration-path",
                str(calibration_json),
                "--device",
                "cpu",
            ],
            dry_run=args.dry_run,
        )
    else:
        announce_stage(6, total_stages, "skip existing transition diagnostics", transition_json)

    if should_run(transition_summary_json, args.overwrite):
        announce_stage(7, total_stages, "summarize transition training gate", transition_summary_json)
        transition_summary = summarize_transition_diagnostics(
            transition_json,
            transition_summary_json,
            transition_summary_csv,
            min_success_rate=args.transition_min_success_rate,
            min_effect_z=args.transition_min_effect_z,
            min_token_match_rate=args.transition_min_token_match_rate,
            min_pairs=args.transition_min_pairs,
            dry_run=args.dry_run,
        )
        coverage_check = add_transition_coverage_check(
            transition_summary,
            fields=fields,
            min_kept_per_field=args.transition_min_kept_per_field,
        )
        if not args.dry_run:
            transition_summary_json.write_text(
                json.dumps(transition_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if (
            not args.dry_run
            and args.transition_min_kept_per_field > 0
            and not coverage_check["passed"]
        ):
            failed = ", ".join(
                f"{field}={count}"
                for field, count in sorted(coverage_check["failed_fields"].items())
            )
            raise RuntimeError(
                "Counterfactual transition coverage is too weak for official "
                f"five-field training: {failed}. Increase samples/balance target "
                "or fix pseudo-target generation before training."
            )
    else:
        announce_stage(7, total_stages, "skip existing transition training summary", transition_summary_json)

    manifest = {
        "dataset": args.dataset,
        "instruction_input": str(args.instruction_input),
        "output_dir": str(output_dir),
        "fields": fields,
        "max_samples": int(args.max_samples),
        "variants_per_field": int(args.variants_per_field),
        "direction_target_policy": str(args.direction_target_policy),
        "counterfactual_geometry_policy": str(args.counterfactual_geometry_policy),
        "direction_metric_mode": str(args.direction_metric_mode),
        "curvature_order_mode": str(args.curvature_order_mode),
        "field_max_mean_reference_distance_m": [str(item) for item in args.field_max_mean_reference_distance_m],
        "include_original": bool(args.include_original),
        "balance_fields": bool(args.balance_fields),
        "balance_target": int(args.balance_target),
        "storage_format": str(args.storage_format),
        "raw_jsonl": None if split_enabled else str(raw_jsonl),
        "raw_summary": str(raw_summary),
        "prefiltered_jsonl": None if split_enabled else str(prefiltered_jsonl),
        "prefiltered_summary": None if split_enabled else str(prefiltered_summary),
        "bootstrap_calibration_json": str(bootstrap_calibration_json),
        "filtered_jsonl": None if split_enabled else str(filtered_jsonl),
        "filtered_summary": None if split_enabled else str(filtered_summary),
        "calibration_json": str(calibration_json),
        "transition_diagnostics_json": str(transition_json),
        "transition_training_summary_json": str(transition_summary_json),
        "transition_training_summary_csv": str(transition_summary_csv),
        "split": {
            "enabled": bool(split_enabled),
            "manifest_path": str(args.split_manifest) if args.split_manifest else None,
            "raw_jsonl": {split: str(raw_split_jsonl[split]) for split in SPLIT_ORDER} if split_enabled else {},
            "prefiltered_jsonl": {
                split: str(prefiltered_split_jsonl[split]) for split in SPLIT_ORDER
            } if split_enabled else {},
            "filtered_jsonl": {
                split: str(filtered_split_jsonl[split]) for split in SPLIT_ORDER
            } if split_enabled else {},
            "filtered_compact_jsonl": {
                split: str(output_dir / "compact" / f"{args.dataset}_counterfactual_instruction_samples.filtered.{split}.compact.jsonl")
                for split in SPLIT_ORDER
            } if split_enabled and str(args.storage_format) == "compact" else {},
            "calibration_source_split": "train" if split_enabled else "all",
            "transition_diagnostics_source_split": "train" if split_enabled else "all",
        },
        "transition_training_thresholds": {
            "min_success_rate": float(args.transition_min_success_rate),
            "min_effect_z": float(args.transition_min_effect_z),
            "min_token_match_rate": float(args.transition_min_token_match_rate),
            "min_pairs": int(args.transition_min_pairs),
            "min_kept_per_field": int(args.transition_min_kept_per_field),
        },
        "calibration_filter": {
            "enabled": bool(args.calibration_aware_filter),
            "calibration_source": "prefiltered_bootstrap" if args.calibration_aware_filter else "filtered",
            "min_effect_z": float(args.min_calibration_effect_z),
            "min_token_match_rate": float(args.min_calibration_token_match_rate),
            "max_collateral_z": float(args.max_calibration_collateral_z),
            "field_min_effect_z": [str(item) for item in args.field_min_calibration_effect_z],
            "field_min_token_match_rate": [str(item) for item in args.field_min_calibration_token_match_rate],
            "field_max_collateral_z": [str(item) for item in args.field_max_calibration_collateral_z],
        },
        "target_key": "pseudo_trajectory",
        "target_len": int(args.target_len),
    }
    if not args.dry_run:
        manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "manifest_json": str(manifest_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
