#!/usr/bin/env python
"""Run the official dual-mode four-table evaluation.

The paper model is hierarchical rather than a single checkpoint used for every
metric family:

* Stage 2 route-first selector -> Table I continuous realism.
* Qwen semantic planner -> Table II language-to-mobility semantic planning.
* Stage 1 local-edit decoder -> Table III instruction following and Table IV
  same-OD counterfactual controllability.

This runner keeps that division explicit while writing a single eval directory
that is compatible with scripts/summarize_four_tables.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def log(message: str) -> None:
    print(f"[dual-four-tables] {message}", flush=True)


def run_command(cmd: Sequence[str], *, dry_run: bool = False) -> None:
    log("$ " + " ".join(str(part) for part in cmd))
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, check=True)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def first_npz_key(path: Path, keys: Sequence[str]) -> str:
    import numpy as np

    payload = np.load(path)
    for key in keys:
        if key in payload.files:
            return key
    raise KeyError(f"{path} missing all keys: {', '.join(keys)}")


def copy_json(src: Path, dst: Path) -> None:
    require_file(src, "metrics source")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def build_stage2_command(args: argparse.Namespace, stage2_dir: Path) -> list[str]:
    required = {
        "stage2_reranker_checkpoint": args.stage2_reranker_checkpoint,
        "memory_real_npz": args.memory_real_npz,
        "train_query_real_npz": args.train_query_real_npz,
        "val_real_npz": args.val_real_npz,
        "test_real_npz": args.test_real_npz,
        "train_input": args.train_input,
        "val_input": args.val_input,
        "test_input": args.test_input,
        "val_counterfactual_input": args.val_counterfactual_input,
        "test_counterfactual_input": args.test_counterfactual_input,
    }
    for label, path in required.items():
        if path is None:
            raise ValueError(f"--{label.replace('_', '-')} is required when --stage2-generated-npz is not provided.")
        require_file(path, label)

    cmd = [
        sys.executable,
        "scripts/train_semantic_route_memory_reranker.py",
        "--eval-only",
        "--eval-split",
        args.eval_split,
        "--skip-table3",
        "--skip-table4",
        "--checkpoint",
        str(args.stage2_reranker_checkpoint),
        "--memory-real-npz",
        str(args.memory_real_npz),
        "--train-query-real-npz",
        str(args.train_query_real_npz),
        "--val-real-npz",
        str(args.val_real_npz),
        "--test-real-npz",
        str(args.test_real_npz),
        "--train-input",
        str(args.train_input),
        "--val-input",
        str(args.val_input),
        "--test-input",
        str(args.test_input),
        "--val-counterfactual-input",
        str(args.val_counterfactual_input),
        "--test-counterfactual-input",
        str(args.test_counterfactual_input),
        "--output-dir",
        str(stage2_dir),
        "--eval-samples",
        str(args.stage2_eval_samples),
        "--memory-samples",
        str(args.stage2_memory_samples),
        "--train-query-samples",
        str(args.stage2_train_query_samples),
        "--prior-samples",
        str(args.stage2_prior_samples),
        "--candidate-topm",
        str(args.stage2_candidate_topm),
        "--target-length-scale",
        str(args.stage2_target_length_scale),
        "--target-length-knn",
        str(args.stage2_target_length_knn),
        "--device",
        args.device,
    ]
    if args.stage2_disable_semantic_costs:
        cmd.append("--disable-semantic-costs")
    if args.route_memory_feature_cache is not None:
        cmd += ["--route-memory-feature-cache", str(args.route_memory_feature_cache)]
        if args.reuse_route_memory_feature_cache:
            cmd.append("--reuse-route-memory-feature-cache")
    if args.route_memory_prior_cache is not None:
        cmd += ["--route-memory-prior-cache", str(args.route_memory_prior_cache)]
        if args.reuse_route_memory_prior_cache:
            cmd.append("--reuse-route-memory-prior-cache")
    if args.route_memory_candidate_cache is not None:
        cmd += ["--route-memory-candidate-cache", str(args.route_memory_candidate_cache)]
        if args.reuse_route_memory_candidate_cache:
            cmd.append("--reuse-route-memory-candidate-cache")
    cmd.extend(args.stage2_extra_args or [])
    return cmd


def build_stage1_table3_command(args: argparse.Namespace, table3_root: Path) -> list[str]:
    require_file(args.stage1_checkpoint, "stage1 checkpoint")
    require_file(args.test_input, "test input")
    cmd = [
        sys.executable,
        "-m",
        "langdirecttraj.eval.eval_instruction_following",
        "--dataset",
        args.dataset,
        "--input",
        str(args.test_input),
        "--target-key",
        args.stage1_eval_target_key,
        "--checkpoint",
        str(args.stage1_checkpoint),
        "--max-samples",
        str(args.stage1_eval_samples),
        "--batch-size",
        str(args.stage1_batch_size),
        "--device",
        args.device,
        "--output-dir",
        str(table3_root),
    ]
    if args.calibration_path is not None:
        require_file(args.calibration_path, "token calibration")
        cmd += ["--token-calibration-path", str(args.calibration_path)]
    if args.disable_qwen:
        cmd.append("--disable-qwen")
    else:
        if args.qwen_planner_checkpoint is not None:
            require_file(args.qwen_planner_checkpoint, "Qwen planner checkpoint")
            cmd += ["--qwen-planner-checkpoint", str(args.qwen_planner_checkpoint)]
        if args.qwen_model_name:
            cmd += ["--qwen-model-name", args.qwen_model_name]
        if args.qwen_local_files_only:
            cmd.append("--qwen-local-files-only")
    cmd.extend(args.stage1_table3_extra_args or [])
    return cmd


def build_table2_planner_command(args: argparse.Namespace, table2_dir: Path) -> list[str]:
    if args.qwen_planner_checkpoint is None:
        raise ValueError("--qwen-planner-checkpoint is required for Table II semantic planner evaluation.")
    require_file(args.qwen_planner_checkpoint, "Qwen planner checkpoint")
    require_file(args.test_input, "test input")
    cmd = [
        sys.executable,
        "scripts/eval_qwen_semantic_planner.py",
        "--dataset",
        args.dataset,
        "--input",
        str(args.test_input),
        "--checkpoint",
        str(args.qwen_planner_checkpoint),
        "--output-dir",
        str(table2_dir),
        "--max-samples",
        str(args.table2_max_samples),
        "--batch-size",
        str(args.table2_batch_size),
        "--device",
        args.device,
    ]
    if args.qwen_model_name:
        cmd += ["--model-name", args.qwen_model_name]
    if args.qwen_local_files_only:
        cmd.append("--local-files-only")
    if args.table2_no_rule_override:
        cmd.append("--no-rule-override")
    cmd.extend(args.table2_extra_args or [])
    return cmd


def build_stage1_table4_command(args: argparse.Namespace, table4_root: Path) -> list[str]:
    require_file(args.stage1_checkpoint, "stage1 checkpoint")
    require_file(args.test_counterfactual_input, "test counterfactual input")
    cmd = [
        sys.executable,
        "-m",
        "langdirecttraj.eval.eval_counterfactual_controllability",
        "--dataset",
        args.dataset,
        "--input",
        str(args.test_counterfactual_input),
        "--target-key",
        "pseudo_trajectory",
        "--checkpoint",
        str(args.stage1_checkpoint),
        "--max-records",
        str(args.table4_max_records),
        "--max-pairs",
        str(args.table4_max_pairs),
        "--pair-sampling",
        args.table4_pair_sampling,
        "--device",
        args.device,
        "--output-dir",
        str(table4_root),
    ]
    if args.calibration_path is not None:
        require_file(args.calibration_path, "token calibration")
        cmd += ["--token-calibration-path", str(args.calibration_path)]
    if args.transition_diagnostics_path is not None:
        require_file(args.transition_diagnostics_path, "transition diagnostics")
        cmd += ["--transition-diagnostics-path", str(args.transition_diagnostics_path)]
    if args.disable_qwen:
        cmd.append("--disable-qwen")
    else:
        if args.qwen_planner_checkpoint is not None:
            require_file(args.qwen_planner_checkpoint, "Qwen planner checkpoint")
            cmd += ["--qwen-planner-checkpoint", str(args.qwen_planner_checkpoint)]
        if args.qwen_model_name:
            cmd += ["--qwen-model-name", args.qwen_model_name]
        if args.qwen_local_files_only:
            cmd.append("--qwen-local-files-only")
    if args.table4_fields:
        cmd += ["--fields", *args.table4_fields]
    cmd.extend(args.stage1_table4_extra_args or [])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage2-only", action="store_true", help="Only compute Stage-2 Table I/II outputs.")

    parser.add_argument("--test-input", required=True, type=Path)
    parser.add_argument("--train-input", type=Path, default=None)
    parser.add_argument("--val-input", type=Path, default=None)
    parser.add_argument("--test-counterfactual-input", type=Path, default=None)
    parser.add_argument("--val-counterfactual-input", type=Path, default=None)
    parser.add_argument("--calibration-path", type=Path, default=None)
    parser.add_argument("--transition-diagnostics-path", type=Path, default=None)

    parser.add_argument("--stage2-generated-npz", type=Path, default=None)
    parser.add_argument("--stage2-reranker-checkpoint", type=Path, default=None)
    parser.add_argument("--memory-real-npz", type=Path, default=None)
    parser.add_argument("--train-query-real-npz", type=Path, default=None)
    parser.add_argument("--val-real-npz", type=Path, default=None)
    parser.add_argument("--test-real-npz", type=Path, default=None)
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--stage2-output-name", default="test1000")
    parser.add_argument("--stage2-eval-samples", type=int, default=0)
    parser.add_argument("--stage2-memory-samples", type=int, default=50000)
    parser.add_argument("--stage2-train-query-samples", type=int, default=1000)
    parser.add_argument("--stage2-prior-samples", type=int, default=10000)
    parser.add_argument("--stage2-candidate-topm", type=int, default=64)
    parser.add_argument("--stage2-target-length-scale", type=float, default=1.15)
    parser.add_argument("--stage2-target-length-knn", type=int, default=16)
    parser.add_argument("--stage2-disable-semantic-costs", action="store_true")
    parser.add_argument("--route-memory-feature-cache", type=Path, default=None)
    parser.add_argument("--reuse-route-memory-feature-cache", action="store_true")
    parser.add_argument("--route-memory-prior-cache", type=Path, default=None)
    parser.add_argument("--reuse-route-memory-prior-cache", action="store_true")
    parser.add_argument("--route-memory-candidate-cache", type=Path, default=None)
    parser.add_argument("--reuse-route-memory-candidate-cache", action="store_true")
    parser.add_argument(
        "--skip-existing-table1",
        action="store_true",
        help="Reuse eval/table1/<dataset>/metrics.json when it already exists.",
    )
    parser.add_argument(
        "--stage2-extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments appended to train_semantic_route_memory_reranker.py. Put this option last.",
    )

    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--stage1-eval-samples", type=int, default=0)
    parser.add_argument("--stage1-eval-target-key", choices=["trajectory", "pseudo_trajectory"], default="trajectory")
    parser.add_argument("--stage1-batch-size", type=int, default=64)
    parser.add_argument("--table4-max-records", type=int, default=0)
    parser.add_argument("--table4-max-pairs", type=int, default=0)
    parser.add_argument("--table4-pair-sampling", choices=["sequential", "balanced_field"], default="balanced_field")
    parser.add_argument("--table4-fields", nargs="*", default=[])
    parser.add_argument("--disable-qwen", action="store_true")
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default="")
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--stage1-table3-extra-args", nargs="*", default=[])
    parser.add_argument("--stage1-table4-extra-args", nargs="*", default=[])
    parser.add_argument(
        "--skip-existing-table3",
        action="store_true",
        help="Reuse eval/table3/<dataset>/metrics.json when it already exists.",
    )
    parser.add_argument(
        "--skip-existing-table4",
        action="store_true",
        help="Reuse eval/table4/<dataset>/metrics.json when it already exists.",
    )
    parser.add_argument("--table2-max-samples", type=int, default=0)
    parser.add_argument("--table2-batch-size", type=int, default=16)
    parser.add_argument("--table2-no-rule-override", action="store_true")
    parser.add_argument(
        "--skip-existing-table2",
        action="store_true",
        help="Reuse eval/table2/<dataset>/metrics.json when it already exists.",
    )
    parser.add_argument("--table2-extra-args", nargs="*", default=[])

    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--fail-on-summary-hard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    eval_root = output_root / "eval"
    table1_dir = eval_root / "table1" / args.dataset
    table2_dir = eval_root / "table2" / args.dataset
    table3_root = eval_root / "table3"
    table4_root = eval_root / "table4"
    stage2_dir = output_root / "stage2_route_first"

    require_file(args.test_input, "test input")
    if not args.stage2_only:
        if args.test_counterfactual_input is None:
            raise ValueError("--test-counterfactual-input is required unless --stage2-only is set.")
        if args.stage1_checkpoint is None:
            raise ValueError("--stage1-checkpoint is required unless --stage2-only is set.")
        require_file(args.test_counterfactual_input, "test counterfactual input")

    table1_metrics = table1_dir / "metrics.json"
    if args.skip_existing_table1 and table1_metrics.exists():
        log(f"skip existing Table I metrics: {table1_metrics}")
        stage2_npz = None
        stage2_table1 = table1_metrics
    elif args.stage2_generated_npz is None:
        stage2_cmd = build_stage2_command(args, stage2_dir)
        run_command(stage2_cmd, dry_run=args.dry_run)
        stage2_split_dir = stage2_dir / args.stage2_output_name
        stage2_npz = stage2_split_dir / "semantic_reranker_trajectories.npz"
        stage2_table1 = stage2_split_dir / "table1_metrics.json"
    else:
        require_file(args.stage2_generated_npz, "stage2 generated npz")
        stage2_npz = args.stage2_generated_npz
        stage2_table1 = output_root / "stage2_route_first" / "table1_metrics_from_npz.json"
        real_key = first_npz_key(stage2_npz, ("real_trajectories", "real", "trajectories_real"))
        generated_key = first_npz_key(stage2_npz, ("generated_trajectories", "trajectories", "generated"))
        run_command(
            [
                sys.executable,
                "-m",
                "langdirecttraj.eval.distribution_metrics",
                "--real",
                str(stage2_npz),
                "--real-key",
                real_key,
                "--generated",
                str(stage2_npz),
                "--generated-key",
                generated_key,
                "--output",
                str(stage2_table1),
            ],
            dry_run=args.dry_run,
        )

    if args.dry_run:
        table2_metrics = table2_dir / "metrics.json"
        if args.skip_existing_table2 and table2_metrics.exists():
            log(f"skip existing Table II metrics: {table2_metrics}")
        else:
            run_command(build_table2_planner_command(args, table2_dir), dry_run=True)
        if not args.stage2_only:
            table3_metrics = table3_root / args.dataset / "metrics.json"
            table4_metrics = table4_root / args.dataset / "metrics.json"
            if args.skip_existing_table3 and table3_metrics.exists():
                log(f"skip existing Table III metrics: {table3_metrics}")
            else:
                run_command(build_stage1_table3_command(args, table3_root), dry_run=True)
            if args.skip_existing_table4 and table4_metrics.exists():
                log(f"skip existing Table IV metrics: {table4_metrics}")
            else:
                run_command(build_stage1_table4_command(args, table4_root), dry_run=True)
        summary_cmd = [
            sys.executable,
            "scripts/summarize_four_tables.py",
            str(eval_root),
            "--dataset",
            args.dataset,
        ]
        if args.fail_on_summary_hard:
            summary_cmd.append("--fail-on-hard")
        if not args.stage2_only:
            run_command(summary_cmd, dry_run=True)
        log("dry-run complete; no files were written.")
        return

    if not (args.skip_existing_table1 and table1_metrics.exists()):
        copy_json(stage2_table1, table1_metrics)

    table2_metrics = table2_dir / "metrics.json"
    if args.skip_existing_table2 and table2_metrics.exists():
        log(f"skip existing Table II metrics: {table2_metrics}")
    else:
        run_command(build_table2_planner_command(args, table2_dir), dry_run=args.dry_run)

    if not args.stage2_only:
        table3_metrics = table3_root / args.dataset / "metrics.json"
        table4_metrics = table4_root / args.dataset / "metrics.json"
        if args.skip_existing_table3 and table3_metrics.exists():
            log(f"skip existing Table III metrics: {table3_metrics}")
        else:
            run_command(build_stage1_table3_command(args, table3_root), dry_run=args.dry_run)
        if args.skip_existing_table4 and table4_metrics.exists():
            log(f"skip existing Table IV metrics: {table4_metrics}")
        else:
            run_command(build_stage1_table4_command(args, table4_root), dry_run=args.dry_run)

    summary_cmd = [
        sys.executable,
        "scripts/summarize_four_tables.py",
        str(eval_root),
        "--dataset",
        args.dataset,
    ]
    if args.fail_on_summary_hard:
        summary_cmd.append("--fail-on-hard")
    if not args.stage2_only:
        run_command(summary_cmd, dry_run=args.dry_run)

    manifest = {
        "dataset": args.dataset,
        "eval_root": str(eval_root),
        "stage2_npz": str(stage2_npz),
        "stage2_table1_source": str(stage2_table1),
        "stage1_checkpoint": str(args.stage1_checkpoint) if args.stage1_checkpoint is not None else None,
        "table1": str(table1_dir / "metrics.json"),
        "table2": str(table2_dir / "metrics.json"),
        "table3": None if args.stage2_only else str(table3_root / args.dataset / "metrics.json"),
        "table4": None if args.stage2_only else str(table4_root / args.dataset / "metrics.json"),
        "dual_mode_contract": {
            "table1": "stage2_route_first",
            "table2": "qwen_semantic_planner",
            "table3": "stage1_local_edit",
            "table4": "stage1_local_edit",
        },
    }
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "dual_mode_four_tables_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    log(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
