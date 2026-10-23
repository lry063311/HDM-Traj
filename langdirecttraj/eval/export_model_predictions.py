from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - keep export usable in minimal envs.
    tqdm = None

from langdirecttraj.data.export_baseline_dataset import tokenize_trajectory
from langdirecttraj.data.instruction_dataset import InstructionTrajectoryDataset, collate_instruction_batch
from langdirecttraj.eval.distribution_metrics import haversine_km, infer_bounds as infer_metric_bounds, to_grid_ids
from langdirecttraj.eval.eval_anchor_decoder import build_model, load_checkpoint
from langdirecttraj.eval.route_selection import load_selection_prior, select_prediction_with_policy
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    extract_qwen_semantic_bundle,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    move_batch,
    counterfactual_route_prior_for_batch,
    load_counterfactual_route_prior,
    load_route_memory_teacher,
    qwen_semantic_bundle_for_batch,
    road_condition_kwargs,
    route_memory_teacher_for_batch,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def meter_scale(points: np.ndarray) -> np.ndarray:
    lat = float(np.nanmean(points[:, 1])) if len(points) else 0.0
    return np.asarray([111_320.0 * math.cos(math.radians(lat)), 110_540.0], dtype=np.float32)


def reference_to_local(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ref = np.asarray(reference, dtype=np.float32)
    if ref.ndim != 2 or ref.shape[-1] != 2 or len(ref) < 2:
        raise ValueError(f"Expected reference [T,2] with T>=2, got {ref.shape}")
    origin = ref[:1].astype(np.float32)
    scale = meter_scale(ref)
    ref_m = (ref - origin) * scale
    route_scale = float(np.linalg.norm(np.diff(ref_m, axis=0), axis=1).sum())
    route_scale = max(route_scale, 100.0)
    return ref_m / route_scale, origin[0], scale, route_scale


def resample_np(points: np.ndarray, target_len: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == target_len:
        return points
    if len(points) <= 1:
        return np.repeat(points[:1], target_len, axis=0)
    old_x = np.linspace(0.0, 1.0, len(points), endpoint=True)
    new_x = np.linspace(0.0, 1.0, target_len, endpoint=True)
    out = np.stack(
        [
            np.interp(new_x, old_x, points[:, 0]),
            np.interp(new_x, old_x, points[:, 1]),
        ],
        axis=1,
    )
    return out.astype(np.float32)


def infer_bounds(records: List[Dict], padding_ratio: float = 0.02) -> Dict[str, List[float]]:
    pts = [np.asarray(record["trajectory"], dtype=np.float32) for record in records]
    all_pts = np.concatenate(pts, axis=0)
    lon_min, lat_min = all_pts.min(axis=0)
    lon_max, lat_max = all_pts.max(axis=0)
    lon_pad = max(float(lon_max - lon_min) * padding_ratio, 1e-6)
    lat_pad = max(float(lat_max - lat_min) * padding_ratio, 1e-6)
    return {
        "lon_bound": [float(lon_min - lon_pad), float(lon_max + lon_pad)],
        "lat_bound": [float(lat_min - lat_pad), float(lat_max + lat_pad)],
    }


def prepare_reference_batch(raw_batch: Dict, target_len: int, device: torch.device):
    refs = []
    meta = []
    for i in range(raw_batch["reference"].shape[0]):
        ref_len = int(raw_batch["reference_mask"][i].sum().item())
        ref = raw_batch["reference"][i, : max(ref_len, 1)].detach().cpu().numpy()
        ref_norm, origin, scale, route_scale = reference_to_local(ref)
        refs.append(torch.from_numpy(resample_np(ref_norm, target_len)).to(device=device, dtype=torch.float32))
        meta.append({"origin": origin, "scale": scale, "route_scale": route_scale})
    return torch.stack(refs, dim=0), meta


def denormalize_predictions(pred: torch.Tensor, meta: List[Dict]) -> np.ndarray:
    pred_np = pred.detach().cpu().float().numpy()
    out = []
    for points, item in zip(pred_np, meta):
        local_m = points * float(item["route_scale"])
        lonlat = local_m / item["scale"][None, :] + item["origin"][None, :]
        out.append(lonlat.astype(np.float32))
    return np.stack(out, axis=0)


def denormalize_candidate_predictions(pred: torch.Tensor, meta: List[Dict]) -> np.ndarray:
    pred_np = pred.detach().cpu().float().numpy()
    if pred_np.ndim != 4:
        raise ValueError(f"Expected candidate predictions [B,K,T,2], got {pred_np.shape}")
    out = []
    for modes, item in zip(pred_np, meta):
        local_m = modes * float(item["route_scale"])
        lonlat = local_m / item["scale"][None, None, :] + item["origin"][None, None, :]
        out.append(lonlat.astype(np.float32))
    return np.stack(out, axis=0)


def _load_npz_array(path: Path, preferred_key: str) -> np.ndarray:
    payload = np.load(path)
    if preferred_key in payload.files:
        return np.asarray(payload[preferred_key], dtype=np.float32)
    for key in ("real_trajectories", "trajectories", "generated", "real"):
        if key in payload.files:
            return np.asarray(payload[key], dtype=np.float32)
    raise ValueError(f"{path} does not contain {preferred_key!r} or a known trajectory key.")


def _train_distribution_prior(path: Path, key: str, grid_num: int, length_bins: int) -> Dict:
    trajs = _load_npz_array(path, key)
    bounds = infer_metric_bounds(trajs, trajs)
    num_cells = int(grid_num) * int(grid_num)
    density = np.zeros(num_cells, dtype=np.float64)
    transition = np.zeros(num_cells * num_cells, dtype=np.float64)
    lengths = haversine_km(trajs[:, :-1], trajs[:, 1:]).sum(axis=1)
    max_len = float(np.ceil(np.percentile(lengths, 99) / 10.0) * 10.0)
    max_len = max(max_len, 1.0)
    length_hist, length_edges = np.histogram(np.clip(lengths, 0.0, max_len), bins=np.linspace(0.0, max_len, int(length_bins) + 1))
    for traj in np.asarray(trajs, dtype=np.float32):
        ids = to_grid_ids(traj, bounds, int(grid_num))
        if ids.size == 0:
            continue
        np.add.at(density, ids, 1.0)
        keep = np.ones(ids.shape[0], dtype=bool)
        keep[1:] = ids[1:] != ids[:-1]
        ids = ids[keep]
        if ids.size >= 2:
            pair_ids = ids[:-1] * num_cells + ids[1:]
            np.add.at(transition, pair_ids, 1.0)
    return {
        "path": str(path),
        "key": key,
        "bounds": bounds,
        "grid_num": int(grid_num),
        "length_bins": int(length_bins),
        "density_prob": (density + 1.0) / float(density.sum() + density.size),
        "transition_prob": (transition + 1.0) / float(transition.sum() + transition.size),
        "length_prob": (length_hist.astype(np.float64) + 1.0) / float(length_hist.sum() + length_hist.size),
        "length_edges": length_edges.astype(np.float64),
    }


def select_candidates_with_train_distribution_prior(
    candidates: np.ndarray,
    prior: Dict,
    *,
    density_weight: float = 1.0,
    transition_weight: float = 2.0,
    length_weight: float = 1.0,
) -> tuple[np.ndarray, Dict[str, float]]:
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 4:
        raise ValueError(f"Expected candidates [N,K,T,2], got {candidates.shape}")
    n, k, _t, _d = candidates.shape
    grid_num = int(prior["grid_num"])
    num_cells = grid_num * grid_num
    density_nll = -np.log(np.asarray(prior["density_prob"], dtype=np.float64))
    transition_nll = -np.log(np.asarray(prior["transition_prob"], dtype=np.float64))
    length_nll = -np.log(np.asarray(prior["length_prob"], dtype=np.float64))
    length_edges = np.asarray(prior["length_edges"], dtype=np.float64)
    scores = np.zeros((n, k), dtype=np.float64)
    density_scores = np.zeros((n, k), dtype=np.float64)
    transition_scores = np.zeros((n, k), dtype=np.float64)
    length_scores = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        for mode in range(k):
            traj = candidates[i, mode]
            ids = to_grid_ids(traj, prior["bounds"], grid_num)
            if ids.size:
                density_scores[i, mode] = float(density_nll[ids].mean())
            else:
                density_scores[i, mode] = float(density_nll.mean())
            if ids.size >= 2:
                keep = np.ones(ids.shape[0], dtype=bool)
                keep[1:] = ids[1:] != ids[:-1]
                ids_changed = ids[keep]
                if ids_changed.size >= 2:
                    pair_ids = ids_changed[:-1] * num_cells + ids_changed[1:]
                    transition_scores[i, mode] = float(transition_nll[pair_ids].mean())
                else:
                    transition_scores[i, mode] = float(transition_nll.mean())
            else:
                transition_scores[i, mode] = float(transition_nll.mean())
            length = float(haversine_km(traj[:-1], traj[1:]).sum())
            bin_idx = int(np.searchsorted(length_edges, np.clip(length, length_edges[0], length_edges[-1]), side="right") - 1)
            bin_idx = int(np.clip(bin_idx, 0, len(length_nll) - 1))
            length_scores[i, mode] = float(length_nll[bin_idx])
    scores = (
        float(density_weight) * density_scores
        + float(transition_weight) * transition_scores
        + float(length_weight) * length_scores
    )
    selected = np.argmin(scores, axis=1).astype(np.int64)
    row = np.arange(n)
    stats = {
        "train_prior_density_score": float(density_scores[row, selected].mean()),
        "train_prior_transition_score": float(transition_scores[row, selected].mean()),
        "train_prior_length_score": float(length_scores[row, selected].mean()),
        "train_prior_total_score": float(scores[row, selected].mean()),
        "train_prior_density_weight": float(density_weight),
        "train_prior_transition_weight": float(transition_weight),
        "train_prior_length_weight": float(length_weight),
    }
    return selected, stats


def load_oracle_selected_k(path: Path | None) -> Dict[int, int]:
    if path is None:
        return {}
    payload = np.load(path)
    if "selected_k" not in payload:
        raise ValueError(f"--oracle-selected-k requires selected_k in {path}")
    selected = np.asarray(payload["selected_k"], dtype=np.int64).reshape(-1)
    if "sample_ids" in payload:
        sample_ids = np.asarray(payload["sample_ids"], dtype=np.int64).reshape(-1)
    elif "sample_id" in payload:
        sample_ids = np.asarray(payload["sample_id"], dtype=np.int64).reshape(-1)
    else:
        sample_ids = np.arange(selected.shape[0], dtype=np.int64)
    if sample_ids.shape[0] != selected.shape[0]:
        raise ValueError(
            f"Oracle selected_k/sample_ids length mismatch in {path}: "
            f"{selected.shape[0]} vs {sample_ids.shape[0]}"
        )
    return {int(sample_id): int(k) for sample_id, k in zip(sample_ids, selected)}


def select_prediction_with_oracle_k(
    pred_out,
    sample_ids: torch.Tensor,
    oracle_selected_k: Dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    if not isinstance(pred_out, tuple):
        pred = pred_out
        idx = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
        return pred, idx, {"oracle_selected_k_applied_rate": 0.0}
    pred, _aux = pred_out
    if pred.ndim != 4:
        idx = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
        return pred, idx, {"oracle_selected_k_applied_rate": 0.0}
    ids = sample_ids.detach().cpu().numpy().astype(np.int64).reshape(-1)
    selected = []
    found = 0
    for sample_id in ids:
        if int(sample_id) not in oracle_selected_k:
            raise KeyError(f"sample_id={int(sample_id)} is missing from --oracle-selected-k.")
        mode_idx = int(oracle_selected_k[int(sample_id)])
        if mode_idx < 0 or mode_idx >= int(pred.shape[1]):
            raise ValueError(
                f"Oracle selected_k={mode_idx} is outside current candidate width {pred.shape[1]} "
                f"for sample_id={int(sample_id)}."
            )
        selected.append(mode_idx)
        found += 1
    idx = torch.tensor(selected, dtype=torch.long, device=pred.device)
    row = torch.arange(pred.shape[0], device=pred.device)
    return pred[row, idx], idx, {
        "oracle_selected_k_applied_rate": float(found / max(len(selected), 1)),
    }


def evaluate_model(args: argparse.Namespace) -> Dict:
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(
        checkpoint,
        device,
        token_calibration_path=args.token_calibration_path,
        arg_overrides={
            "route_proposal_logit_weight": args.route_proposal_logit_weight_override,
            "route_control_selection_adapter_weight": args.route_control_selection_adapter_weight_override,
            "route_control_selection_adapter_clip": args.route_control_selection_adapter_clip_override,
            "road_candidate_logit_weight": args.road_candidate_logit_weight_override,
            "road_candidate_logit_mode": args.road_candidate_logit_mode_override,
            "road_candidate_anchor_weight": args.road_candidate_anchor_weight_override,
            "road_candidate_residual_scale": args.road_candidate_residual_scale_override,
            "route_memory_prior_condition_weight": args.route_memory_prior_condition_weight_override,
            "route_memory_prior_condition_path": args.route_memory_prior_condition_path_override,
            "num_candidates": args.num_candidates_override,
        },
    )
    decoder = getattr(model, "decoder", None)
    if args.reference_positive_candidate_strength_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_positive_candidate_strength"):
            raise ValueError("--reference-positive-candidate-strength-override requires a compatible decoder.")
        decoder.set_reference_positive_candidate_strength(
            float(args.reference_positive_candidate_strength_override)
        )
    if args.reference_positive_candidate_scope_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_positive_candidate_scope"):
            raise ValueError("--reference-positive-candidate-scope-override requires a compatible decoder.")
        decoder.set_reference_positive_candidate_scope(
            str(args.reference_positive_candidate_scope_override)
        )
    if args.reference_distance_candidate_strength_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_distance_candidate_strength"):
            raise ValueError("--reference-distance-candidate-strength-override requires a compatible decoder.")
        decoder.set_reference_distance_candidate_strength(
            float(args.reference_distance_candidate_strength_override)
        )
    if args.reference_distance_candidate_scope_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_distance_candidate_scope"):
            raise ValueError("--reference-distance-candidate-scope-override requires a compatible decoder.")
        decoder.set_reference_distance_candidate_scope(
            str(args.reference_distance_candidate_scope_override)
        )
    if args.reference_distance_candidate_mode_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_distance_candidate_mode"):
            raise ValueError("--reference-distance-candidate-mode-override requires a compatible decoder.")
        decoder.set_reference_distance_candidate_mode(
            str(args.reference_distance_candidate_mode_override)
        )
    saved_args = checkpoint.get("args", {})
    attr_dim = int(saved_args.get("attr_dim", 6))
    if args.disable_qwen or args.qwen_plan_cache is not None:
        args.qwen_planner_checkpoint = None
    elif args.qwen_planner_checkpoint is None:
        saved_qwen_checkpoint = saved_args.get("qwen_planner_checkpoint")
        if saved_qwen_checkpoint:
            args.qwen_planner_checkpoint = Path(saved_qwen_checkpoint)
    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device) if args.qwen_planner_checkpoint else None
    selection_prior = load_selection_prior(args.selection_prior_path, device)
    oracle_selected_k = load_oracle_selected_k(args.oracle_selected_k)
    train_distribution_prior = None
    if args.selection_policy == "train_distribution_prior":
        if args.road_selection_prior_npz is None:
            raise ValueError("--selection-policy train_distribution_prior requires --road-selection-prior-npz.")
        train_distribution_prior = _train_distribution_prior(
            args.road_selection_prior_npz,
            args.road_selection_prior_key,
            args.road_selection_prior_grid_num,
            args.road_selection_prior_length_bins,
        )

    dataset = InstructionTrajectoryDataset(
        args.input,
        max_samples=args.max_samples,
        target_key=args.target_key,
        qwen_plan_cache_path=args.qwen_plan_cache,
        road_corridor_cache_path=args.road_corridor_cache,
    )
    route_memory_prior_lookup = load_counterfactual_route_prior(
        args.counterfactual_route_prior_npz,
        args.counterfactual_route_prior_key,
    )
    route_memory_teacher = load_route_memory_teacher(
        args.route_memory_teacher_npz,
        candidates_key=args.route_memory_teacher_candidates_key,
        scores_key=args.route_memory_teacher_scores_key,
        components_key=args.route_memory_teacher_components_key,
        global_candidates_key=args.route_memory_teacher_global_candidates_key,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_instruction_batch,
    )
    bounds = infer_bounds(dataset.records[: len(dataset)], padding_ratio=args.grid_padding_ratio)

    real_trajs = []
    gen_trajs = []
    real_token_records = []
    token_records = []
    selected_mode_records = []
    selection_stats_accum = []
    candidate_batches = []
    deferred_candidate_batches = []
    deferred_records = []
    with torch.no_grad():
        offset = 0
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(
                loader,
                desc="export predictions",
                total=len(loader),
                dynamic_ncols=True,
                mininterval=5.0,
            )
        for raw_batch in iterator:
            batch = move_batch(raw_batch, device)
            reference, ref_meta = prepare_reference_batch(raw_batch, model.decoder.target_len, device)
            attr_cond = make_attr_cond(batch, attr_dim, device)
            semantic_plan_tokens = qwen_semantic_bundle_for_batch(qwen_provider, batch, device)
            route_scales = torch.tensor(
                [float(item["route_scale"]) for item in ref_meta],
                device=device,
                dtype=torch.float32,
            )
            route_memory_prior = None
            route_memory_prior_valid = None
            if route_memory_prior_lookup and getattr(model, "route_memory_prior_encoder", None) is not None:
                route_memory_prior, route_memory_prior_valid = counterfactual_route_prior_for_batch(
                    batch.get("sample_id"),
                    route_memory_prior_lookup,
                    model.decoder.target_len,
                    device,
                    reference.dtype,
                )
            route_teacher_candidates = None
            route_teacher_scores = None
            route_teacher_components = None
            route_teacher_valid = None
            route_teacher_global_candidates = None
            route_teacher_global_origin = None
            route_teacher_global_scale = None
            route_memory_local_origin = None
            route_memory_local_meter_scale = None
            if route_memory_teacher and float(args.route_memory_teacher_anchor_weight) != 0.0:
                (
                    route_teacher_candidates,
                    route_teacher_scores,
                    route_teacher_components,
                    route_teacher_valid,
                    route_teacher_global_candidates,
                ) = (
                    route_memory_teacher_for_batch(
                        batch.get("sample_id"),
                        route_memory_teacher,
                        model.decoder.target_len,
                        device,
                        reference.dtype,
                    )
                )
                if route_teacher_global_candidates is not None:
                    route_teacher_global_origin = torch.as_tensor(
                        route_memory_teacher.get("global_origin"),
                        device=device,
                        dtype=reference.dtype,
                    )
                    route_teacher_global_scale = torch.as_tensor(
                        route_memory_teacher.get("global_scale"),
                        device=device,
                        dtype=reference.dtype,
                    )
                    route_memory_local_origin = torch.as_tensor(
                        np.stack([item["origin"] for item in ref_meta], axis=0),
                        device=device,
                        dtype=reference.dtype,
                    )
                    route_memory_local_meter_scale = torch.as_tensor(
                        np.stack([item["scale"] for item in ref_meta], axis=0),
                        device=device,
                        dtype=reference.dtype,
                    )
            pred_out = model(
                reference,
                batch["mobility_tokens"],
                attr_cond,
                semantic_plan_tokens=semantic_plan_tokens,
                route_scales=route_scales,
                route_memory_prior=route_memory_prior,
                route_memory_prior_valid=route_memory_prior_valid,
                route_memory_teacher_anchor=route_teacher_candidates,
                route_memory_teacher_global_anchor=route_teacher_global_candidates,
                route_memory_teacher_scores=route_teacher_scores,
                route_memory_teacher_components=route_teacher_components,
                route_memory_teacher_valid=route_teacher_valid,
                route_memory_teacher_global_origin=route_teacher_global_origin,
                route_memory_teacher_global_scale=route_teacher_global_scale,
                route_memory_local_origin=route_memory_local_origin,
                route_memory_local_meter_scale=route_memory_local_meter_scale,
                route_memory_teacher_anchor_weight=args.route_memory_teacher_anchor_weight,
                route_memory_teacher_anchor_residual_scale=args.route_memory_teacher_anchor_residual_scale,
                **road_condition_kwargs(batch),
                return_aux=True,
            )
            if args.selection_policy == "train_distribution_prior":
                pred_modes = pred_out[0] if isinstance(pred_out, tuple) else pred_out
                if pred_modes.ndim != 4:
                    raise ValueError("train_distribution_prior requires multi-candidate predictions [B,K,T,2].")
                denorm_candidates = denormalize_candidate_predictions(pred_modes, ref_meta)
                deferred_candidate_batches.append(denorm_candidates)
                if args.save_candidates:
                    candidate_batches.append(denorm_candidates)
                for j in range(pred_modes.shape[0]):
                    record = dataset.records[offset + j]
                    real = resample_np(np.asarray(record[args.target_key], dtype=np.float32), model.decoder.target_len)
                    real_trajs.append(real)
                    real_tokens = tokenize_trajectory(real, bounds, args.grid_size)
                    real_token_records.append(
                        {
                            "sample_id": int(record["sample_id"]),
                            "dataset": str(record.get("dataset", args.dataset)),
                            "instruction": str(record.get("instruction", "")),
                            "labels": dict(record.get("labels", {})),
                            "grid_tokens": real_tokens,
                            "origin": int(real_tokens[0]) if real_tokens else -1,
                            "destination": int(real_tokens[-1]) if real_tokens else -1,
                            "length": int(len(real_tokens)),
                        }
                    )
                    deferred_records.append(record)
                offset += pred_modes.shape[0]
                continue
            if oracle_selected_k:
                pred, selected_idx, select_stats = select_prediction_with_oracle_k(
                    pred_out,
                    batch["sample_id"],
                    oracle_selected_k,
                )
            else:
                pred, selected_idx, select_stats = select_prediction_with_policy(
                    pred_out,
                    reference=reference,
                    reference_lonlat=raw_batch["reference"].to(device),
                    route_scales=torch.tensor(
                        [float(item["route_scale"]) for item in ref_meta],
                        device=device,
                        dtype=torch.float32,
                    ),
                    prior=selection_prior,
                    policy=args.selection_policy,
                    route_weight=args.selection_route_weight,
                    token_guard_margin=args.selection_token_guard_margin,
                    token_absolute_threshold=args.selection_token_absolute_threshold,
                    min_route_improvement=args.selection_min_route_improvement,
                    logit_guard_margin=args.selection_logit_guard_margin,
                    preserve_direction=args.selection_preserve_direction,
                    preserve_target_direction=args.selection_preserve_target_direction,
                    mobility_tokens=batch["mobility_tokens"],
                    dataset=args.dataset,
                )
            if args.save_candidates:
                pred_modes = pred_out[0] if isinstance(pred_out, tuple) else pred_out
                if pred_modes.ndim == 4:
                    candidate_batches.append(denormalize_candidate_predictions(pred_modes, ref_meta))
            selected_mode_records.append(selected_idx.detach().cpu().numpy().astype(np.int64))
            selection_stats_accum.append(select_stats)
            gen_lonlat = denormalize_predictions(pred, ref_meta)
            gen_trajs.append(gen_lonlat)
            for j in range(gen_lonlat.shape[0]):
                record = dataset.records[offset + j]
                real = resample_np(np.asarray(record[args.target_key], dtype=np.float32), model.decoder.target_len)
                real_trajs.append(real)
                real_tokens = tokenize_trajectory(real, bounds, args.grid_size)
                real_token_records.append(
                    {
                        "sample_id": int(record["sample_id"]),
                        "dataset": str(record.get("dataset", args.dataset)),
                        "instruction": str(record.get("instruction", "")),
                        "labels": dict(record.get("labels", {})),
                        "grid_tokens": real_tokens,
                        "origin": int(real_tokens[0]) if real_tokens else -1,
                        "destination": int(real_tokens[-1]) if real_tokens else -1,
                        "length": int(len(real_tokens)),
                    }
                )
                tokens = tokenize_trajectory(gen_lonlat[j], bounds, args.grid_size)
                token_records.append(
                    {
                        "sample_id": int(record["sample_id"]),
                        "dataset": str(record.get("dataset", args.dataset)),
                        "instruction": str(record.get("instruction", "")),
                        "labels": dict(record.get("labels", {})),
                        "grid_tokens": tokens,
                        "origin": int(tokens[0]) if tokens else -1,
                        "destination": int(tokens[-1]) if tokens else -1,
                        "length": int(len(tokens)),
                    }
                )
            offset += gen_lonlat.shape[0]

    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    continuous_path = output_dir / f"{args.dataset}_model_predictions.npz"
    real_token_path = output_dir / f"{args.dataset}_real_trajgpt_tokens.jsonl"
    token_path = output_dir / f"{args.dataset}_model_trajgpt_tokens.jsonl"
    summary_path = output_dir / "prediction_export_summary.json"
    real_arr = np.stack(real_trajs, axis=0).astype(np.float32)
    if args.selection_policy == "train_distribution_prior":
        candidate_arr = np.concatenate(deferred_candidate_batches, axis=0).astype(np.float32)
        selected_modes, train_prior_stats = select_candidates_with_train_distribution_prior(
            candidate_arr,
            train_distribution_prior,
            density_weight=args.road_selection_prior_density_weight,
            transition_weight=args.road_selection_prior_transition_weight,
            length_weight=args.road_selection_prior_length_weight,
        )
        gen_arr = candidate_arr[np.arange(candidate_arr.shape[0]), selected_modes].astype(np.float32)
        selected_mode_records = [selected_modes.astype(np.int64)]
        selection_stats_accum.append(train_prior_stats)
        for record, traj in zip(deferred_records, gen_arr):
            tokens = tokenize_trajectory(traj, bounds, args.grid_size)
            token_records.append(
                {
                    "sample_id": int(record["sample_id"]),
                    "dataset": str(record.get("dataset", args.dataset)),
                    "instruction": str(record.get("instruction", "")),
                    "labels": dict(record.get("labels", {})),
                    "grid_tokens": tokens,
                    "origin": int(tokens[0]) if tokens else -1,
                    "destination": int(tokens[-1]) if tokens else -1,
                    "length": int(len(tokens)),
                }
            )
    else:
        gen_arr = np.concatenate(gen_trajs, axis=0).astype(np.float32)
    save_payload = {
        "trajectories": gen_arr,
        "real_trajectories": real_arr,
        "sample_ids": np.asarray([record["sample_id"] for record in dataset.records[: len(real_arr)]], dtype=np.int64),
        "selected_modes": np.concatenate(selected_mode_records, axis=0).astype(np.int64) if selected_mode_records else np.zeros(0, dtype=np.int64),
    }
    if args.save_candidates and candidate_batches:
        save_payload["candidate_trajectories"] = np.concatenate(candidate_batches, axis=0).astype(np.float32)
    np.savez_compressed(continuous_path, **save_payload)
    averaged_selection_stats = {}
    if selection_stats_accum:
        keys = sorted({key for stats in selection_stats_accum for key in stats})
        for key in keys:
            values = [float(stats[key]) for stats in selection_stats_accum if key in stats]
            if values:
                averaged_selection_stats[key] = float(np.mean(values))
    selected_modes = np.concatenate(selected_mode_records, axis=0) if selected_mode_records else np.zeros(0, dtype=np.int64)
    with real_token_path.open("w", encoding="utf-8") as f:
        for record in real_token_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with token_path.open("w", encoding="utf-8") as f:
        for record in token_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "dataset": args.dataset,
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "num_samples": int(len(real_arr)),
        "target_len": int(model.decoder.target_len),
        "num_candidates": int(getattr(model.decoder, "num_candidates", getattr(model, "num_candidates", 1))),
        "num_candidates_override": args.num_candidates_override,
        "grid_size": int(args.grid_size),
        "qwen_semantic_conditioning": qwen_provider is not None,
        "continuous_npz": str(continuous_path),
        "real_trajgpt_token_jsonl": str(real_token_path),
        "trajgpt_token_jsonl": str(token_path),
        "bounds": bounds,
        "reference_protocol": "od_chord_only",
        "normalization_protocol": "reference_length_only",
        "selection_policy": args.selection_policy,
        "oracle_selected_k": str(args.oracle_selected_k) if args.oracle_selected_k else None,
        "selection_stats": averaged_selection_stats,
        "selected_mode_hist": (
            np.bincount(selected_modes.astype(np.int64), minlength=max(int(selected_modes.max()) + 1, 1)).tolist()
            if selected_modes.size
            else []
        ),
        "road_corridor_cache": str(args.road_corridor_cache) if args.road_corridor_cache else None,
        "road_candidate_logit_weight": float(getattr(model, "road_candidate_logit_weight", 0.0)),
        "road_candidate_logit_weight_override": args.road_candidate_logit_weight_override,
        "road_candidate_logit_mode": str(getattr(model, "road_candidate_logit_mode", "add")),
        "road_candidate_logit_mode_override": args.road_candidate_logit_mode_override,
        "road_candidate_anchor_weight": float(getattr(model, "road_candidate_anchor_weight", 0.0)),
        "road_candidate_anchor_weight_override": args.road_candidate_anchor_weight_override,
        "road_candidate_residual_scale": float(getattr(model, "road_candidate_residual_scale", 1.0)),
        "road_candidate_residual_scale_override": args.road_candidate_residual_scale_override,
        "route_memory_prior_condition_weight": float(
            getattr(model, "route_memory_prior_condition_weight", 0.0)
        ),
        "route_memory_prior_condition_weight_override": args.route_memory_prior_condition_weight_override,
        "route_memory_prior_condition_path": str(
            getattr(model, "route_memory_prior_condition_path", "separate")
        ),
        "route_memory_prior_condition_path_override": args.route_memory_prior_condition_path_override,
        "counterfactual_route_prior_npz": (
            str(args.counterfactual_route_prior_npz) if args.counterfactual_route_prior_npz else None
        ),
        "counterfactual_route_prior_key": str(args.counterfactual_route_prior_key),
        "counterfactual_route_prior_labels": len(route_memory_prior_lookup),
        "route_memory_teacher_npz": str(args.route_memory_teacher_npz) if args.route_memory_teacher_npz else None,
        "route_memory_teacher_anchor_weight": float(args.route_memory_teacher_anchor_weight),
        "route_memory_teacher_anchor_residual_scale": float(args.route_memory_teacher_anchor_residual_scale),
        "route_memory_teacher_labels": len(route_memory_teacher.get("index", {})) if route_memory_teacher else 0,
        "qwen_plan_cache": str(args.qwen_plan_cache) if args.qwen_plan_cache else None,
        "selection_prior_path": str(args.selection_prior_path) if args.selection_prior_path else None,
        "selection_route_weight": float(args.selection_route_weight),
        "road_selection_prior_npz": str(args.road_selection_prior_npz) if args.road_selection_prior_npz else None,
        "road_selection_prior_key": str(args.road_selection_prior_key),
        "road_selection_prior_grid_num": int(args.road_selection_prior_grid_num),
        "road_selection_prior_length_bins": int(args.road_selection_prior_length_bins),
        "road_selection_prior_density_weight": float(args.road_selection_prior_density_weight),
        "road_selection_prior_transition_weight": float(args.road_selection_prior_transition_weight),
        "road_selection_prior_length_weight": float(args.road_selection_prior_length_weight),
        "reference_positive_candidate_strength_override": args.reference_positive_candidate_strength_override,
        "reference_positive_candidate_scope_override": args.reference_positive_candidate_scope_override,
        "reference_distance_candidate_strength_override": args.reference_distance_candidate_strength_override,
        "reference_distance_candidate_scope_override": args.reference_distance_candidate_scope_override,
        "reference_distance_candidate_mode_override": args.reference_distance_candidate_mode_override,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model predictions for Table 1/2 baseline-compatible eval.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "model_predictions")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--grid-padding-ratio", type=float, default=0.02)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--disable-qwen", action="store_true", help="Do not inherit or load a saved Qwen planner.")
    parser.add_argument("--token-calibration-path", type=Path, default=None)
    parser.add_argument("--qwen-plan-cache", type=Path, default=None)
    parser.add_argument("--road-corridor-cache", type=Path, default=None)
    parser.add_argument(
        "--selection-policy",
        choices=[
            "reranker",
            "od_cell_prior",
            "reranker_plus_od_cell_prior",
            "reranker_guarded_od_cell_prior",
            "transition_prior",
            "realism_guarded_transition_prior",
            "reranker_guarded_transition_prior",
            "target_direction_transition_prior",
            "target_control_transition_prior",
            "reranker_guarded_target_control_transition_prior",
            "train_distribution_prior",
            "fixed_mode_0",
            "fixed_mode_1",
            "fixed_mode_2",
            "fixed_mode_3",
            "fixed_mode_4",
            "fixed_mode_5",
        ],
        default="reranker",
    )
    parser.add_argument("--selection-prior-path", type=Path, default=None)
    parser.add_argument("--selection-route-weight", type=float, default=1.0)
    parser.add_argument("--road-selection-prior-npz", type=Path, default=None)
    parser.add_argument("--road-selection-prior-key", default="real_trajectories")
    parser.add_argument("--road-selection-prior-grid-num", type=int, default=32)
    parser.add_argument("--road-selection-prior-length-bins", type=int, default=32)
    parser.add_argument("--road-selection-prior-density-weight", type=float, default=1.0)
    parser.add_argument("--road-selection-prior-transition-weight", type=float, default=2.0)
    parser.add_argument("--road-selection-prior-length-weight", type=float, default=1.0)
    parser.add_argument("--selection-token-guard-margin", type=float, default=0.0)
    parser.add_argument("--selection-token-absolute-threshold", type=float, default=float("inf"))
    parser.add_argument("--selection-min-route-improvement", type=float, default=0.0)
    parser.add_argument("--selection-logit-guard-margin", type=float, default=float("inf"))
    parser.add_argument("--selection-preserve-direction", action="store_true")
    parser.add_argument("--selection-preserve-target-direction", action="store_true")
    parser.add_argument("--save-candidates", action="store_true")
    parser.add_argument("--counterfactual-route-prior-npz", type=Path, default=None)
    parser.add_argument("--counterfactual-route-prior-key", default="route_prior_norm")
    parser.add_argument("--route-memory-teacher-npz", type=Path, default=None)
    parser.add_argument("--route-memory-teacher-candidates-key", default="route_teacher_candidates_norm")
    parser.add_argument("--route-memory-teacher-global-candidates-key", default="route_teacher_candidates_global")
    parser.add_argument("--route-memory-teacher-scores-key", default="teacher_scores")
    parser.add_argument("--route-memory-teacher-components-key", default="teacher_component_scores")
    parser.add_argument("--route-memory-teacher-anchor-weight", type=float, default=0.0)
    parser.add_argument("--route-memory-teacher-anchor-residual-scale", type=float, default=1.0)
    parser.add_argument("--route-memory-prior-condition-weight-override", type=float, default=None)
    parser.add_argument(
        "--route-memory-prior-condition-path-override",
        choices=["shared", "separate"],
        default=None,
    )
    parser.add_argument(
        "--oracle-selected-k",
        type=Path,
        default=None,
        help="NPZ with selected_k and sample_ids; forces per-sample mode selection for oracle diagnosis.",
    )
    parser.add_argument("--route-proposal-logit-weight-override", type=float, default=None)
    parser.add_argument("--route-control-selection-adapter-weight-override", type=float, default=None)
    parser.add_argument("--route-control-selection-adapter-clip-override", type=float, default=None)
    parser.add_argument("--road-candidate-logit-weight-override", type=float, default=None)
    parser.add_argument("--road-candidate-logit-mode-override", choices=["add", "override", "none"], default=None)
    parser.add_argument("--road-candidate-anchor-weight-override", type=float, default=None)
    parser.add_argument("--road-candidate-residual-scale-override", type=float, default=None)
    parser.add_argument("--num-candidates-override", type=int, default=None)
    parser.add_argument(
        "--road-route-first",
        action="store_true",
        help="Export preset: road scorer selects the road-candidate skeleton directly with residual disabled.",
    )
    parser.add_argument(
        "--road-route-first-residual-scale",
        type=float,
        default=0.0,
        help="Residual scale used by --road-route-first; keep near 0 for route-first exports.",
    )
    parser.add_argument("--reference-positive-candidate-strength-override", type=float, default=None)
    parser.add_argument(
        "--reference-positive-candidate-scope-override",
        choices=["all", "none", "counterfactual_reference_adherence"],
        default=None,
    )
    parser.add_argument("--reference-distance-candidate-strength-override", type=float, default=None)
    parser.add_argument(
        "--reference-distance-candidate-scope-override",
        choices=["all", "none", "instruction", "counterfactual_reference_adherence"],
        default=None,
    )
    parser.add_argument(
        "--reference-distance-candidate-mode-override",
        choices=["global_normal", "local_direction_preserving"],
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.road_route_first:
        args.road_candidate_anchor_weight_override = 1.0
        args.road_candidate_residual_scale_override = float(args.road_route_first_residual_scale)
        args.road_candidate_logit_mode_override = "override"
        args.road_candidate_logit_weight_override = 1.0
    print(json.dumps(evaluate_model(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
