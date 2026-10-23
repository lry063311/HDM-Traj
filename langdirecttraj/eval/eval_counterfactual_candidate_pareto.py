from __future__ import annotations

import argparse
import atexit
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from langdirecttraj.eval.eval_anchor_decoder import build_model, load_checkpoint
from langdirecttraj.eval.eval_counterfactual_controllability import (
    build_pair_calibrator,
    collateral_drift_z_by_field,
    generate_candidate_geometries,
    group_counterfactual_pairs,
    independent_collateral_fields,
    read_records,
)
from langdirecttraj.eval.instruction_geometry import GeometryTokenCalibrator
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.train.control_losses import load_counterfactual_transition_table
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    parse_field_epochs,
    parse_field_float_map,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose counterfactual candidate-pool ceilings before final "
            "reranker/selection. Reports whether any mode can achieve target "
            "effect under a collateral threshold."
        )
    )
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "counterfactual_candidate_pareto")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--pair-sampling", choices=["sequential", "balanced_field"], default="balanced_field")
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["distance", "direction", "curvature", "progress", "reference_adherence"],
    )
    parser.add_argument("--success-margin-z", type=float, default=0.0)
    parser.add_argument("--collateral-margin-z", type=float, default=0.08)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument(
        "--skip-qwen",
        action="store_true",
        help=(
            "Do not load the Qwen planner. This is intended for thermal-safe "
            "local candidate-pool ceiling diagnostics where we only need to "
            "separate generator candidate capacity from Qwen semantic planning."
        ),
    )
    parser.add_argument(
        "--local-cuda-lock-path",
        type=Path,
        default=REPO_ROOT / "outputs" / ".local_cuda.lock",
        help="Lock file used to prevent parallel local CUDA diagnostics on the RTX 2060.",
    )
    parser.add_argument(
        "--disable-local-cuda-lock",
        action="store_true",
        help="Disable the local CUDA lock. Use only on the server or when external scheduling is active.",
    )
    parser.add_argument("--token-calibration-path", type=Path, default=None)
    parser.add_argument("--counterfactual-adapter-scale", type=float, default=0.0)
    parser.add_argument("--transition-diagnostics-path", type=Path, default=None)
    parser.add_argument("--transition-min-success-rate", type=float, default=0.6)
    parser.add_argument("--transition-min-effect-z", type=float, default=0.05)
    parser.add_argument("--transition-min-token-match-rate", type=float, default=0.5)
    parser.add_argument("--transition-min-pairs", type=int, default=2)
    parser.add_argument("--transition-field-min-success-rates", default="")
    parser.add_argument("--transition-field-min-effect-z", default="")
    parser.add_argument("--transition-field-min-token-match-rates", default="")
    parser.add_argument("--transition-field-min-pairs", default="")
    parser.add_argument("--transition-max-collateral-z", type=float, default=0.0)
    parser.add_argument("--transition-field-max-collateral-z", default="")
    parser.add_argument("--transition-allowlist", default="")
    parser.add_argument("--save-pairs", action="store_true")
    return parser.parse_args()


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            int(pid),
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    except PermissionError:
        return True
    return True


def acquire_local_cuda_lock(path: Path, enabled: bool) -> None:
    if not enabled:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", -1))
        except Exception:
            pid = -1
        if _pid_exists(pid):
            raise RuntimeError(
                f"Local CUDA lock is active at {path} for pid={pid}. "
                "Do not run parallel Qwen/DiT diagnostics on the RTX 2060."
            )
        path.unlink(missing_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Local CUDA lock was acquired by another process: {path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "tool": "eval_counterfactual_candidate_pareto"}, f)

    def _release() -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("pid", -1)) == os.getpid():
                path.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_release)


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _safe_rate(values: Iterable[float]) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _quantiles(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
    }


def summarize_rows(rows: List[Dict[str, float]]) -> Dict[str, object]:
    if not rows:
        return {
            "num_pairs": 0,
            "oracle_success_rate": float("nan"),
            "pareto_success_rate": float("nan"),
        }
    return {
        "num_pairs": len(rows),
        "oracle_target_effect": _quantiles(row["oracle_target_effect"] for row in rows),
        "pareto_target_effect": _quantiles(row["pareto_target_effect"] for row in rows),
        "best_collateral_at_success": _quantiles(row["best_collateral_at_success"] for row in rows),
        "oracle_success_rate": _safe_rate(row["oracle_success"] for row in rows),
        "pareto_success_rate": _safe_rate(row["pareto_success"] for row in rows),
        "any_collateral_feasible_rate": _safe_rate(row["any_collateral_feasible"] for row in rows),
        "avg_num_modes": _safe_mean(row["num_modes"] for row in rows),
    }


def evaluate_candidate_pool(
    model,
    qwen_provider,
    pairs: List[tuple[Dict, Dict]],
    attr_dim: int,
    target_key: str,
    calibrator: GeometryTokenCalibrator,
    device: torch.device,
    success_margin_z: float,
    collateral_margin_z: float,
    counterfactual_adapter_scale: float,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    by_field: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    for pair_index, (original, counterfactual) in enumerate(pairs):
        cf_meta = counterfactual.get("counterfactual", {})
        field_name = str(cf_meta.get("field", ""))
        source_token = str(cf_meta.get("from", original.get("labels", {}).get(field_name, "unknown")))
        target_token = str(cf_meta.get("to", counterfactual.get("labels", {}).get(field_name, "unknown")))
        if field_name not in DEFAULT_MOBILITY_VOCAB.fields or target_token not in calibrator.stats.get(field_name, {}):
            continue

        base = generate_candidate_geometries(
            model,
            qwen_provider,
            original,
            attr_dim,
            target_key,
            device,
        )
        cf = generate_candidate_geometries(
            model,
            qwen_provider,
            counterfactual,
            attr_dim,
            target_key,
            device,
            counterfactual_adapter_scale=float(counterfactual_adapter_scale),
        )
        num_modes = min(len(base["geoms"]), len(cf["geoms"]))
        if num_modes <= 0:
            continue

        mode_rows: List[Dict[str, float]] = []
        for mode_idx in range(num_modes):
            base_geom = base["geoms"][mode_idx]
            cf_geom = cf["geoms"][mode_idx]
            base_target_score = calibrator.score_token(field_name, target_token, base_geom)
            cf_target_score = calibrator.score_token(field_name, target_token, cf_geom)
            target_effect = float(base_target_score - cf_target_score)
            drift_by_field = collateral_drift_z_by_field(
                calibrator,
                field_name,
                original.get("labels", {}),
                base_geom,
                cf_geom,
            )
            independent_drift_by_field = independent_collateral_fields(field_name, drift_by_field)
            if independent_drift_by_field:
                collateral = float(np.mean(list(independent_drift_by_field.values())))
            elif drift_by_field:
                collateral = float(np.mean(list(drift_by_field.values())))
            else:
                collateral = 0.0
            if not np.isfinite(collateral):
                collateral = 0.0
            mode_rows.append(
                {
                    "mode": float(mode_idx),
                    "target_effect": target_effect,
                    "collateral": collateral,
                    "collateral_by_field": {
                        key: float(value) for key, value in sorted(drift_by_field.items())
                    },
                    "independent_collateral_by_field": {
                        key: float(value) for key, value in sorted(independent_drift_by_field.items())
                    },
                    "cf_target_score": float(cf_target_score),
                    "base_target_score": float(base_target_score),
                    "success": float(target_effect > float(success_margin_z)),
                    "collateral_feasible": float(collateral <= float(collateral_margin_z)),
                    "pareto_feasible": float(
                        target_effect > float(success_margin_z)
                        and collateral <= float(collateral_margin_z)
                    ),
                }
            )

        oracle = max(mode_rows, key=lambda row: row["target_effect"])
        collateral_feasible = [
            row for row in mode_rows if row["collateral"] <= float(collateral_margin_z)
        ]
        if collateral_feasible:
            pareto = max(collateral_feasible, key=lambda row: row["target_effect"])
            any_collateral_feasible = 1.0
        else:
            pareto = min(mode_rows, key=lambda row: row["collateral"])
            any_collateral_feasible = 0.0
        success_modes = [row for row in mode_rows if row["target_effect"] > float(success_margin_z)]
        best_collateral_at_success = (
            min(row["collateral"] for row in success_modes)
            if success_modes
            else float("nan")
        )
        pair_row: Dict[str, object] = {
            "pair_index": pair_index,
            "sample_id": int(original.get("sample_id", -1)),
            "counterfactual_id": int(counterfactual.get("counterfactual_id", counterfactual.get("sample_id", -1))),
            "field": field_name,
            "source_token": source_token,
            "target_token": target_token,
            "num_modes": float(num_modes),
            "oracle_mode": int(oracle["mode"]),
            "oracle_target_effect": float(oracle["target_effect"]),
            "oracle_collateral": float(oracle["collateral"]),
            "oracle_success": float(oracle["target_effect"] > float(success_margin_z)),
            "pareto_mode": int(pareto["mode"]),
            "pareto_target_effect": float(pareto["target_effect"]),
            "pareto_collateral": float(pareto["collateral"]),
            "pareto_success": float(
                pareto["target_effect"] > float(success_margin_z)
                and pareto["collateral"] <= float(collateral_margin_z)
            ),
            "any_collateral_feasible": any_collateral_feasible,
            "best_collateral_at_success": float(best_collateral_at_success),
            "modes": mode_rows,
        }
        rows.append(pair_row)
        by_field[field_name].append(pair_row)  # type: ignore[arg-type]

    metrics: Dict[str, object] = {
        "overall": summarize_rows(rows),  # type: ignore[arg-type]
        "by_field": {
            field: summarize_rows(field_rows)  # type: ignore[arg-type]
            for field, field_rows in sorted(by_field.items())
        },
    }
    return metrics, rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    acquire_local_cuda_lock(
        args.local_cuda_lock_path,
        enabled=(device.type == "cuda" and not bool(args.disable_local_cuda_lock)),
    )
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint, device, token_calibration_path=args.token_calibration_path)
    saved_args = checkpoint.get("args", {})
    attr_dim = int(saved_args.get("attr_dim", 6))
    if bool(args.skip_qwen):
        args.qwen_planner_checkpoint = None
    elif args.qwen_planner_checkpoint is None:
        saved_qwen_checkpoint = saved_args.get("qwen_planner_checkpoint")
        if saved_qwen_checkpoint:
            args.qwen_planner_checkpoint = Path(saved_qwen_checkpoint)
    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device) if args.qwen_planner_checkpoint else None

    jsonl_path = args.input or (
        REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_counterfactual_instruction_samples.filtered.jsonl"
    )
    records = read_records(jsonl_path, max_records=args.max_records)
    transition_table = load_counterfactual_transition_table(
        args.transition_diagnostics_path,
        min_success_rate=args.transition_min_success_rate,
        min_effect_z=args.transition_min_effect_z,
        min_pairs=args.transition_min_pairs,
        min_token_match_rate=args.transition_min_token_match_rate,
        max_collateral_z=args.transition_max_collateral_z,
        field_min_success_rate=parse_field_float_map(
            args.transition_field_min_success_rates,
            name="transition-field-min-success-rates",
        ),
        field_min_effect_z=parse_field_float_map(
            args.transition_field_min_effect_z,
            name="transition-field-min-effect-z",
            allow_negative=True,
        ),
        field_min_pairs=parse_field_epochs(args.transition_field_min_pairs, name="transition-field-min-pairs"),
        field_min_token_match_rate=parse_field_float_map(
            args.transition_field_min_token_match_rates,
            name="transition-field-min-token-match-rates",
        ),
        field_max_collateral_z=parse_field_float_map(
            args.transition_field_max_collateral_z,
            name="transition-field-max-collateral-z",
        ),
        allow_transitions=args.transition_allowlist,
    )
    pairs, transition_filter_stats = group_counterfactual_pairs(
        records,
        fields=set(args.fields),
        max_pairs=args.max_pairs,
        transition_table=transition_table,
        pair_sampling=args.pair_sampling,
    )
    if not pairs:
        raise ValueError("No original/counterfactual pairs found for candidate-pool Pareto diagnosis.")
    if args.token_calibration_path is not None and args.token_calibration_path.exists():
        calibrator = GeometryTokenCalibrator.from_json(str(args.token_calibration_path))
        calibration_source = str(args.token_calibration_path)
    else:
        calibrator = build_pair_calibrator(pairs, args.target_key, model.decoder.target_len, device)
        calibration_source = "paired target trajectories"

    metrics, rows = evaluate_candidate_pool(
        model=model,
        qwen_provider=qwen_provider,
        pairs=pairs,
        attr_dim=attr_dim,
        target_key=args.target_key,
        calibrator=calibrator,
        device=device,
        success_margin_z=args.success_margin_z,
        collateral_margin_z=args.collateral_margin_z,
        counterfactual_adapter_scale=float(args.counterfactual_adapter_scale),
    )
    result = {
        **metrics,
        "dataset": args.dataset,
        "input": str(jsonl_path),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "target_key": args.target_key,
        "fields": args.fields,
        "num_pairs": len(rows),
        "success_margin_z": float(args.success_margin_z),
        "collateral_margin_z": float(args.collateral_margin_z),
        "counterfactual_adapter_scale": float(args.counterfactual_adapter_scale),
        "qwen_planner_checkpoint": str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None,
        "qwen_semantic_conditioning": args.qwen_planner_checkpoint is not None,
        "qwen_skipped": bool(args.skip_qwen),
        "calibration_source": calibration_source,
        "transition_filter": {
            "diagnostics_path": str(args.transition_diagnostics_path) if args.transition_diagnostics_path else None,
            "enabled": bool(transition_table),
            "stats": transition_filter_stats,
            "pair_sampling": args.pair_sampling,
        },
    }
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    if args.save_pairs:
        pair_path = output_dir / "pairs.jsonl"
        with pair_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        result["pairs_path"] = str(pair_path)
        metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
