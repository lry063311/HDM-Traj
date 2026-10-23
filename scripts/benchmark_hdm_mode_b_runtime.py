#!/usr/bin/env python
"""Unified end-to-end runtime benchmark for HDM-Traj Mode B (local-edit decoder).

Mode B is the Stage-1 local-edit DiT path, benchmarked separately from Mode A
because ordinary instruction following and counterfactual editing use different
selection paths and are not directly comparable with population-level route
generation (see docs/runtime_efficiency_protocol.md).

Two paths, mirroring the reported Table 3 / Table 4 inference configurations:

  --path ordinary        -> "HDM-Traj Mode B (ordinary)"
      OD chord + attr conditioning + structured mobility tokens -> single-step
      decoder forward (K candidate modes) -> ordinary candidate scorer ->
      reranker selection. This is the 0.540 instruction-following-F1 path.

  --path counterfactual  -> "HDM-Traj Mode B (counterfactual)"
      base decoder candidate generation -> counterfactual decoder candidate
      generation conditioned on the selected base -> target-effect / collateral
      computation -> feasible-first weighted (pareto) selection. This is the
      0.712 target-improvement-success path.

Reported path (default) does NOT run the online Qwen planner. The reported
Table 3/4 runs pass semantic_plan_tokens=None (qwen_semantic_conditioning=false
in the saved metrics); use --with-qwen to additionally load the Qwen planner
and time the end-to-end path that includes an online Qwen forward. That is a
supplementary breakdown and must not be merged into the main efficiency table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
for path in (REPO_ROOT, WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from runtime_benchmark_common import (  # noqa: E402
    RuntimeProtocol,
    module_parameter_counts,
    run_runtime_protocol,
)
from langdirecttraj.data.instruction_dataset import (  # noqa: E402
    InstructionTrajectoryDataset,
    collate_instruction_batch,
)
from langdirecttraj.eval.eval_anchor_decoder import (  # noqa: E402
    build_model,
    load_checkpoint,
    prepare_eval_batch,
)
from langdirecttraj.eval.instruction_geometry import GeometryTokenCalibrator  # noqa: E402
from langdirecttraj.eval.route_selection import (  # noqa: E402
    load_selection_prior,
    select_prediction_with_policy,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB  # noqa: E402
from langdirecttraj.train.train_anchor_decoder import (  # noqa: E402
    build_qwen_provider,
    make_attr_cond,
    move_batch,
    qwen_semantic_bundle_for_batch,
    resolve_device,
    road_condition_kwargs,
)

# Counterfactual helpers are imported lazily inside the counterfactual branch so
# that the ordinary path has no dependency on the compact-store pair machinery.

ORDINARY_METHOD = "HDM-Traj Mode B (ordinary)"
COUNTERFACTUAL_METHOD = "HDM-Traj Mode B (counterfactual)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--path", choices=["ordinary", "counterfactual"], required=True)
    parser.add_argument("--input", required=True, type=Path,
                        help="Ordinary: Table-3 instruction jsonl; counterfactual: Table-4 compact pair jsonl.")
    parser.add_argument("--target-key", default="pseudo_trajectory",
                        choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--with-qwen", action="store_true",
                        help="Also load the online Qwen planner and time it inside each request.")
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    # Protocol knobs (identical defaults to the Mode A benchmark).
    parser.add_argument("--latency-samples", type=int, default=1000)
    parser.add_argument("--throughput-samples", type=int, default=9984)
    parser.add_argument("--throughput-batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--throughput-warmup-batches", type=int, default=5)
    # Counterfactual selection weights (final Table-4 configuration).
    parser.add_argument("--cf-success-margin-z", type=float, default=0.0)
    parser.add_argument("--cf-adapter-scale", type=float, default=0.0)
    parser.add_argument("--cf-instruction-weight", type=float, default=0.0)
    parser.add_argument("--cf-target-effect-weight", type=float, default=1.2)
    parser.add_argument("--cf-collateral-weight", type=float, default=3.0)
    parser.add_argument("--cf-collateral-margin", type=float, default=0.08)
    parser.add_argument("--cf-raw-collateral-margin", type=float, default=0.15)
    parser.add_argument("--cf-logit-weight", type=float, default=0.0)
    parser.add_argument("--cf-priority", default="pareto_feasible_first")
    parser.add_argument("--cf-pair-sampling", default="balanced_field",
                        choices=["sequential", "balanced_field"])
    return parser.parse_args()


def _build_qwen_provider(cli: argparse.Namespace, saved_args: dict, device: torch.device):
    """Mirror eval_instruction_following / eval_counterfactual_controllability."""
    if not cli.with_qwen:
        return None
    qwen_ckpt = cli.qwen_planner_checkpoint
    if qwen_ckpt is None:
        saved_qwen = saved_args.get("qwen_planner_checkpoint")
        if saved_qwen:
            qwen_ckpt = Path(saved_qwen)
    if qwen_ckpt is None:
        raise SystemExit("--with-qwen requires a Qwen planner checkpoint "
                         "(--qwen-planner-checkpoint or saved in the stage1 checkpoint)")
    return build_qwen_provider(qwen_ckpt, cli, device)


def run_ordinary(cli: argparse.Namespace) -> None:
    device = resolve_device(cli.device)
    checkpoint = load_checkpoint(cli.checkpoint, device)
    model = build_model(checkpoint, device, token_calibration_path=cli.calibration_path)
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    attr_dim = int(saved_args.get("attr_dim", 6))
    qwen_provider = _build_qwen_provider(cli, saved_args, device)
    selection_prior = load_selection_prior(None, device)

    protocol = RuntimeProtocol(
        latency_samples=cli.latency_samples,
        throughput_samples=cli.throughput_samples,
        throughput_batch_size=cli.throughput_batch_size,
        repeats=cli.repeats,
        latency_warmup=cli.latency_warmup,
        throughput_warmup_batches=cli.throughput_warmup_batches,
    )
    sample_count = max(protocol.latency_samples, protocol.throughput_samples)
    dataset = InstructionTrajectoryDataset(
        cli.input,
        max_samples=sample_count,
        target_key=cli.target_key,
    )
    if len(dataset) < sample_count:
        raise ValueError(f"ordinary input has only {len(dataset)} samples, need {sample_count}")

    def infer(start: int, count: int):
        items = [dataset[index] for index in range(start, start + count)]
        batch = move_batch(collate_instruction_batch(items), device)
        _, reference, route_scales = prepare_eval_batch(batch, model.decoder.target_len)
        attr_cond = make_attr_cond(batch, attr_dim, device)
        semantic_plan_tokens = qwen_semantic_bundle_for_batch(qwen_provider, batch, device)
        pred_out = model(
            reference,
            batch["mobility_tokens"],
            attr_cond,
            semantic_plan_tokens=semantic_plan_tokens,
            **road_condition_kwargs(batch),
            return_aux=True,
        )
        pred, _, _ = select_prediction_with_policy(
            pred_out,
            reference=reference,
            reference_lonlat=batch["reference"],
            route_scales=route_scales[:, 0],
            prior=selection_prior,
            policy="reranker",
            mobility_tokens=batch["mobility_tokens"],
            dataset="porto",
        )
        return pred

    total, trainable = module_parameter_counts([model])
    method_name = ORDINARY_METHOD + (" (+online Qwen)" if qwen_provider is not None else "")
    run_runtime_protocol(
        method=method_name,
        infer=infer,
        device=device,
        output=cli.output,
        total_parameters=total,
        trainable_parameters=trainable,
        aux_storage_bytes=0,
        protocol=protocol,
        metadata={
            "checkpoint": str(cli.checkpoint.resolve()),
            "dataset": "porto",
            "split": "test",
            "steps_or_k": "1-step, K=6 candidates",
            "path": "ordinary_instruction_following",
            "selection_policy": "reranker",
            "num_candidates": int(saved_args.get("num_candidates", 1)),
            "target_key": cli.target_key,
            "online_qwen_planner": bool(qwen_provider is not None),
            "semantic_plan_tokens_source": (
                "online_qwen_forward" if qwen_provider is not None else
                "none_reported_config_qwen_semantic_conditioning_false"
            ),
            "inference_boundary": (
                "semantic_controls_absent_or_ready->prepare_eval_batch->attr_conditioning->"
                "single_step_decoder_forward_k_candidates->candidate_scoring->reranker_selection"
            ),
        },
    )


def run_counterfactual(cli: argparse.Namespace) -> None:
    # Imported here so the ordinary path has no compact-store dependency.
    from langdirecttraj.eval.eval_counterfactual_controllability import (
        group_counterfactual_pairs,
        read_records,
        select_hybrid_counterfactual_geometries,
    )
    from langdirecttraj.train.control_losses import load_counterfactual_transition_table

    device = resolve_device(cli.device)
    checkpoint = load_checkpoint(cli.checkpoint, device)
    model = build_model(checkpoint, device, token_calibration_path=cli.calibration_path)
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    attr_dim = int(saved_args.get("attr_dim", 6))
    qwen_provider = _build_qwen_provider(cli, saved_args, device)
    calibrator = GeometryTokenCalibrator.from_json(str(cli.calibration_path))

    protocol = RuntimeProtocol(
        latency_samples=cli.latency_samples,
        throughput_samples=cli.throughput_samples,
        throughput_batch_size=cli.throughput_batch_size,
        repeats=cli.repeats,
        latency_warmup=cli.latency_warmup,
        throughput_warmup_batches=cli.throughput_warmup_batches,
    )
    sample_count = max(protocol.latency_samples, protocol.throughput_samples)

    records = read_records(cli.input, max_records=0)
    transition_table = load_counterfactual_transition_table(None)
    pairs, _ = group_counterfactual_pairs(
        records,
        fields=set(DEFAULT_MOBILITY_VOCAB.fields),
        max_pairs=sample_count,
        transition_table=transition_table,
        pair_sampling=cli.cf_pair_sampling,
    )
    if len(pairs) < sample_count:
        raise ValueError(f"counterfactual input yields only {len(pairs)} pairs, need {sample_count}")
    pairs = pairs[:sample_count]

    def infer(start: int, count: int):
        returned = []
        for pair_index in range(start, start + count):
            original, counterfactual = pairs[pair_index]
            _, cf_geom, _ = select_hybrid_counterfactual_geometries(
                model,
                qwen_provider,
                original,
                counterfactual,
                attr_dim,
                cli.target_key,
                calibrator,
                device,
                success_margin_z=cli.cf_success_margin_z,
                counterfactual_adapter_scale=cli.cf_adapter_scale,
                selection_prior=None,
                selection_route_weight=1.0,
                instruction_weight=cli.cf_instruction_weight,
                target_effect_weight=cli.cf_target_effect_weight,
                collateral_weight=cli.cf_collateral_weight,
                collateral_margin=cli.cf_collateral_margin,
                raw_collateral_margin=cli.cf_raw_collateral_margin,
                logit_weight=cli.cf_logit_weight,
                priority=cli.cf_priority,
            )
            returned.append(cf_geom)
        return returned

    total, trainable = module_parameter_counts([model])
    method_name = COUNTERFACTUAL_METHOD + (" (+online Qwen)" if qwen_provider is not None else "")
    run_runtime_protocol(
        method=method_name,
        infer=infer,
        device=device,
        output=cli.output,
        total_parameters=total,
        trainable_parameters=trainable,
        aux_storage_bytes=0,
        protocol=protocol,
        metadata={
            "checkpoint": str(cli.checkpoint.resolve()),
            "dataset": "porto",
            "split": "test",
            "steps_or_k": "2 decoder forwards (base + counterfactual), K=6 each",
            "path": "counterfactual_editing",
            "selection_policy": "hybrid_counterfactual",
            "selection_priority": cli.cf_priority,
            "pair_sampling": cli.cf_pair_sampling,
            "target_key": cli.target_key,
            "online_qwen_planner": bool(qwen_provider is not None),
            "throughput_batched": False,
            "throughput_note": (
                "Counterfactual inference is inherently two-stage per pair and cannot be "
                "batched to a single decoder call; the measured throughput is single-stream "
                "pairs/sec, NOT a genuine B=128 batched number. The summary table marks it "
                "N/A; only B=1 latency/P95 and peak GPU memory are comparable."
            ),
            "semantic_plan_tokens_source": (
                "online_qwen_forward" if qwen_provider is not None else
                "none_reported_config_qwen_semantic_conditioning_false"
            ),
            "inference_boundary": (
                "semantic_controls_absent_or_ready->base_decoder_candidate_generation->"
                "counterfactual_decoder_candidate_generation_cond_on_base->"
                "target_effect_collateral_computation->feasible_first_weighted_selection"
            ),
        },
    )


def main() -> None:
    cli = parse_args()
    for path in (cli.checkpoint, cli.input, cli.calibration_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available() and cli.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the Mode B runtime benchmark")
    if cli.path == "ordinary":
        run_ordinary(cli)
    else:
        run_counterfactual(cli)


if __name__ == "__main__":
    main()
