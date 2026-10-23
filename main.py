from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")
    return payload


def parse_scalar(raw: str) -> Any:
    stripped = raw.strip()
    if (stripped.startswith("[") and stripped.endswith("]")) or (
        stripped.startswith("{") and stripped.endswith("}")
    ):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def apply_override(config: dict[str, Any], assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"Override must use key=value syntax, got: {assignment}")
    key_path, raw_value = assignment.split("=", 1)
    keys = [piece for piece in key_path.split(".") if piece]
    if not keys:
        raise ValueError(f"Override has an empty key path: {assignment}")
    cursor: dict[str, Any] = config
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[keys[-1]] = parse_scalar(raw_value)


def powershell_executable(preferred: str | None = None) -> str:
    candidates = [preferred] if preferred else []
    if os.name == "nt":
        candidates.extend(["powershell.exe", "powershell"])
    else:
        candidates.extend(["pwsh", "powershell"])
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return candidate
    raise RuntimeError(
        "No PowerShell executable found. Install PowerShell 7 on Linux or run on Windows PowerShell."
    )


def python_executable(config: dict[str, Any]) -> str:
    return str(config.get("python", sys.executable))


def append_ps_arg(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append("-" + key)
        return
    command.append("-" + key)
    if isinstance(value, list):
        if key == "EvalEpochs":
            command.append(",".join(str(item) for item in value))
        else:
            command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def flatten_ps_args(*sections: dict[str, Any]) -> list[str]:
    command: list[str] = []
    for section in sections:
        for key, value in section.items():
            append_ps_arg(command, key, value)
    return command


def format_command_summary(command: list[str]) -> str:
    if "-File" in command:
        script_index = command.index("-File") + 1
        if script_index < len(command):
            script = Path(str(command[script_index])).name
            interesting = []
            for key in ("-Dataset", "-OutputRoot", "-TotalEpochs", "-EvalEvery", "-EvalEpochs", "-EvalOnly"):
                if key in command:
                    idx = command.index(key)
                    if idx + 1 < len(command) and not str(command[idx + 1]).startswith("-"):
                        interesting.append(f"{key.lstrip('-')}={command[idx + 1]}")
                    else:
                        interesting.append(key.lstrip("-"))
            suffix = " ".join(interesting)
            return f"+ pwsh {script}" + (f" ({suffix})" if suffix else "")
    return "+ " + " ".join(str(part) for part in command[:8]) + (" ..." if len(command) > 8 else "")


def run_command(command: list[str], dry_run: bool) -> None:
    print(format_command_summary(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_epoch_from_chunk(path: Path) -> int:
    match = re.search(r"_epoch_(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def find_loop_checkpoint(output_root: Path, dataset: str, checkpoint_kind: str) -> Path:
    chunks = sorted(
        [path for path in output_root.glob("chunk_*_epoch_*") if path.is_dir()],
        key=parse_epoch_from_chunk,
        reverse=True,
    )
    for chunk in chunks:
        candidate = chunk / "train" / "dit_anchor" / dataset / f"{checkpoint_kind}.pt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {checkpoint_kind}.pt under {output_root}/chunk_*/train/dit_anchor/{dataset}"
    )


def expected_last_chunk_root(output_root: Path, train_loop: dict[str, Any]) -> Path:
    total_epochs = int(train_loop.get("TotalEpochs", 1))
    eval_epochs = train_loop.get("EvalEpochs")
    if isinstance(eval_epochs, list) and eval_epochs:
        milestones = sorted(
            {
                int(epoch)
                for epoch in eval_epochs
                if isinstance(epoch, (int, float, str)) and str(epoch).strip()
            }
        )
        milestones = [epoch for epoch in milestones if 0 < epoch <= total_epochs]
        if not milestones or milestones[-1] != total_epochs:
            milestones.append(total_epochs)
        return output_root / f"chunk_{len(milestones):03d}_epoch_{total_epochs:04d}"
    eval_every = max(int(train_loop.get("EvalEvery", 1)), 1)
    chunk_id = (total_epochs + eval_every - 1) // eval_every
    return output_root / f"chunk_{chunk_id:03d}_epoch_{total_epochs:04d}"


def completed_stage_checkpoint(
    output_root: Path,
    train_loop: dict[str, Any],
    dataset: str,
    checkpoint_kind: str,
) -> Path | None:
    chunk_root = expected_last_chunk_root(output_root, train_loop)
    checkpoint = chunk_root / "train" / "dit_anchor" / dataset / f"{checkpoint_kind}.pt"
    eval_root = chunk_root / "eval"
    required_metrics = [
        eval_root / "table1" / dataset / "metrics.json",
        eval_root / "table2" / dataset / "metrics.json",
        eval_root / "table3" / dataset / "metrics.json",
        eval_root / "table4" / dataset / "metrics.json",
    ]
    if not checkpoint.exists() or not all(path.exists() for path in required_metrics):
        return None
    gate_path = eval_root / "reference_corridor_gate.json"
    if gate_path.exists():
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if str(gate.get("status", "")).upper() == "FAIL":
            return None
    return checkpoint


def run_candidate_gate(eval_root: Path, dataset: str, config: dict[str, Any], dry_run: bool) -> None:
    gate_cfg = config.get("gate", {})
    if not gate_cfg.get("enabled", False):
        return
    rows_path = eval_root / "table4" / "candidate_rows.jsonl"
    if not rows_path.exists() and not dry_run:
        print(f"[gate] skip, candidate rows not found: {rows_path}")
        return

    py = python_executable(config)
    run_command(
        [
            py,
            "-u",
            "scripts/summarize_candidate_rows.py",
            str(rows_path),
            "--output",
            str(eval_root / "table4" / "candidate_rows_summary.json"),
        ],
        dry_run=dry_run,
    )
    run_command(
        [
            py,
            "-u",
            "scripts/check_reference_corridor_gate.py",
            str(eval_root),
            "--dataset",
            dataset,
            "--output",
            str(eval_root / "reference_corridor_gate.json"),
        ],
        dry_run=dry_run,
    )


def run_powershell_loop(config: dict[str, Any], dry_run: bool, skip_final: bool) -> None:
    runner = config.get("runner", {})
    common = deepcopy(config.get("common_args", {}))
    train_loop = deepcopy(config.get("train_loop", {}))
    final_eval = deepcopy(config.get("final_eval", {}))
    final_eval_enabled = bool(final_eval.pop("enabled", True))
    dataset = str(common.get("Dataset", "porto"))

    loop_script = Path(runner.get("loop_script", "scripts/run_local_goal_loop.ps1"))
    if not loop_script.is_absolute():
        loop_script = REPO_ROOT / loop_script
    ps = powershell_executable(runner.get("powershell"))

    loop_command = [
        ps,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(loop_script),
        *flatten_ps_args(common, train_loop),
    ]
    run_command(loop_command, dry_run=dry_run)

    output_root = REPO_ROOT / str(common.get("OutputRoot", "outputs/main_loop"))
    quick_chunks = sorted(
        [path for path in output_root.glob("chunk_*_epoch_*") if path.is_dir()],
        key=parse_epoch_from_chunk,
    )
    if quick_chunks or dry_run:
        quick_eval_root = (
            quick_chunks[-1] / "eval"
            if quick_chunks
            else expected_last_chunk_root(output_root, train_loop) / "eval"
        )
        run_candidate_gate(quick_eval_root, dataset, config, dry_run=dry_run)

    if skip_final or not final_eval_enabled:
        return

    if train_loop.get("EvalOnly", False):
        current_checkpoint = Path(str(common.get("InitCheckpoint", "")))
    elif dry_run:
        current_checkpoint = output_root / "chunk_LAST" / "train" / "dit_anchor" / dataset / (
            str(runner.get("final_checkpoint", "latest")) + ".pt"
        )
    else:
        current_checkpoint = find_loop_checkpoint(
            output_root, dataset, str(runner.get("final_checkpoint", "latest"))
        )

    final_output_root = Path(str(final_eval.pop("OutputRoot", output_root / "final_eval")))
    if not final_output_root.is_absolute():
        final_output_root = REPO_ROOT / final_output_root
    final_args = deepcopy(common)
    final_args.update(final_eval)
    final_args.pop("NoInit", None)
    final_args.update(
        {
            "EvalOnly": True,
            "InitCheckpoint": str(current_checkpoint),
            "OutputRoot": str(final_output_root),
            "TotalEpochs": 1,
            "EvalEvery": 1,
        }
    )
    final_command = [
        ps,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(loop_script),
        *flatten_ps_args(final_args),
    ]
    run_command(final_command, dry_run=dry_run)
    run_candidate_gate(final_output_root / "chunk_001_epoch_0001" / "eval", dataset, config, dry_run=dry_run)


def run_staged_powershell_loop(config: dict[str, Any], dry_run: bool, skip_final: bool) -> None:
    stages = config.get("stages", [])
    if not isinstance(stages, list) or not stages:
        raise ValueError("runner backend staged_powershell_loop requires a non-empty 'stages' list.")

    runner = config.get("runner", {})
    common_base = deepcopy(config.get("common_args", {}))
    train_base = deepcopy(config.get("train_loop", {}))
    final_eval = deepcopy(config.get("final_eval", {}))
    final_eval_enabled = bool(final_eval.pop("enabled", True))
    dataset = str(common_base.get("Dataset", "porto"))

    loop_script = Path(runner.get("loop_script", "scripts/run_local_goal_loop.ps1"))
    if not loop_script.is_absolute():
        loop_script = REPO_ROOT / loop_script
    ps = powershell_executable(runner.get("powershell"))
    output_root_base = REPO_ROOT / str(common_base.get("OutputRoot", "outputs/main_loop"))
    initial_checkpoint = common_base.get("InitCheckpoint")
    current_checkpoint = Path(str(initial_checkpoint)) if initial_checkpoint else None
    last_stage_output_root = output_root_base

    for idx, stage in enumerate(stages, start=1):
        if not isinstance(stage, dict):
            raise ValueError(f"Stage {idx} must be a JSON object.")
        name = str(stage.get("name", f"stage_{idx:02d}"))
        stage_common = deepcopy(common_base)
        stage_train = deepcopy(train_base)
        stage_common.update(deepcopy(stage.get("common_args", {})))
        stage_train.update(deepcopy(stage.get("train_loop", {})))
        stage_output_root = output_root_base / f"{idx:02d}_{name}"
        stage_common["OutputRoot"] = str(stage_output_root)
        if current_checkpoint is not None:
            stage_common["InitCheckpoint"] = str(current_checkpoint)

        print(f"[stage {idx}/{len(stages)}] {name}", flush=True)
        if not dry_run:
            completed_checkpoint = completed_stage_checkpoint(
                stage_output_root,
                stage_train,
                dataset,
                str(runner.get("final_checkpoint", "latest")),
            )
            if completed_checkpoint is not None:
                print(f"[stage {idx}/{len(stages)}] skip completed stage, checkpoint={completed_checkpoint}", flush=True)
                current_checkpoint = completed_checkpoint
                last_stage_output_root = stage_output_root
                continue
        command = [
            ps,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(loop_script),
            *flatten_ps_args(stage_common, stage_train),
        ]
        run_command(command, dry_run=dry_run)

        eval_root = expected_last_chunk_root(stage_output_root, stage_train) / "eval"
        run_candidate_gate(eval_root, dataset, config, dry_run=dry_run)
        if dry_run:
            current_checkpoint = stage_output_root / "chunk_LAST" / "train" / "dit_anchor" / dataset / (
                str(runner.get("final_checkpoint", "latest")) + ".pt"
            )
        else:
            current_checkpoint = find_loop_checkpoint(
                stage_output_root,
                dataset,
                str(runner.get("final_checkpoint", "latest")),
            )
        last_stage_output_root = stage_output_root

    if skip_final or not final_eval_enabled:
        return
    if current_checkpoint is None:
        raise RuntimeError("No checkpoint available for final evaluation.")

    final_output_root = Path(str(final_eval.pop("OutputRoot", output_root_base / "final_eval")))
    if not final_output_root.is_absolute():
        final_output_root = REPO_ROOT / final_output_root
    final_args = deepcopy(common_base)
    final_args.update(final_eval)
    final_args.pop("NoInit", None)
    final_args.update(
        {
            "EvalOnly": True,
            "InitCheckpoint": str(current_checkpoint),
            "OutputRoot": str(final_output_root),
            "TotalEpochs": 1,
            "EvalEvery": 1,
        }
    )
    final_command = [
        ps,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(loop_script),
        *flatten_ps_args(final_args),
    ]
    print(f"[final_eval] checkpoint={current_checkpoint}", flush=True)
    run_command(final_command, dry_run=dry_run)
    run_candidate_gate(final_output_root / "chunk_001_epoch_0001" / "eval", dataset, config, dry_run=dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LangDirectTraj structured train/eval entrypoint."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "reference_corridor_server_train_full_compact_qwen.json",
        help="Path to a JSON experiment config.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a value, e.g. --set train_loop.TotalEpochs=20.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    parser.add_argument("--skip-final", action="store_true", help="Skip final eval.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_json(config_path)
    for override in args.overrides:
        apply_override(config, override)

    print(f"[main] config: {config_path}")
    print(f"[main] experiment: {config.get('experiment_name', config_path.stem)}")
    backend = str(config.get("runner", {}).get("backend", "powershell_loop"))
    if backend == "powershell_loop":
        run_powershell_loop(config, dry_run=args.dry_run, skip_final=args.skip_final)
    elif backend == "staged_powershell_loop":
        run_staged_powershell_loop(config, dry_run=args.dry_run, skip_final=args.skip_final)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


if __name__ == "__main__":
    main()
