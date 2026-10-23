from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from langdirecttraj.data.export_baseline_dataset import tokenize_trajectory
from langdirecttraj.data.instruction_dataset import InstructionTrajectoryDataset, collate_instruction_batch
from langdirecttraj.eval.distribution_metrics import evaluate_distribution
from langdirecttraj.eval.eval_anchor_decoder import build_model, load_checkpoint, prepare_eval_batch
from langdirecttraj.eval.export_model_predictions import (
    denormalize_predictions,
    infer_bounds as infer_token_bounds,
    prepare_reference_batch,
    resample_np,
)
from langdirecttraj.eval.instruction_geometry import contract_labels_from_tensors
from langdirecttraj.eval.sequence_metrics import evaluate_sequences
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    move_batch,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _select_by_index(candidates: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_idx = torch.arange(candidates.shape[0], device=candidates.device)
    return candidates[batch_idx, indices]


def _haversine_km_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lon1 = np.deg2rad(a[..., 0])
    lat1 = np.deg2rad(a[..., 1])
    lon2 = np.deg2rad(b[..., 0])
    lat2 = np.deg2rad(b[..., 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def _traj_length_km(traj: np.ndarray) -> float:
    traj = np.asarray(traj, dtype=np.float32)
    if len(traj) < 2:
        return 0.0
    return float(_haversine_km_np(traj[:-1], traj[1:]).sum())


def _jaccard(a: List[int], b: List[int]) -> float:
    set_a = set(int(x) for x in a)
    set_b = set(int(x) for x in b)
    if not set_a and not set_b:
        return 1.0
    return float(len(set_a & set_b) / max(len(set_a | set_b), 1))


def _parse_fields(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _label_mismatch_count(
    candidate_labels: Dict[str, str],
    target_labels: Dict[str, str],
    fields: List[str],
) -> int:
    mismatches = 0
    for field in fields:
        target = str(target_labels.get(field, "unknown"))
        if target == "unknown":
            continue
        pred = str(candidate_labels.get(field, "unknown"))
        if pred != target:
            mismatches += 1
    return mismatches


def _select_control_guarded(
    base_idx: int,
    mismatch_counts: np.ndarray,
    objective_scores: np.ndarray,
    maximize: bool,
    max_mismatches: int,
) -> int:
    """Select by objective only among instruction-feasible candidates.

    This is intentionally diagnostic: it tests whether the candidate pool can
    satisfy control constraints and route-token quality at the same time before
    adding another trainable scorer.
    """
    feasible = np.flatnonzero(mismatch_counts <= int(max_mismatches))
    if len(feasible) == 0:
        return int(base_idx)
    feasible_scores = objective_scores[feasible]
    if maximize:
        return int(feasible[int(np.argmax(feasible_scores))])
    return int(feasible[int(np.argmin(feasible_scores))])


def _load_od_prior(path: Path | None) -> Dict | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "od_cell_prior" not in payload:
        raise ValueError(f"Prior file has no od_cell_prior section: {path}")
    return payload


def _od_sector_and_distance_bin(start: np.ndarray, end: np.ndarray, prior: Dict) -> tuple[int, int]:
    od = prior["od_cell_prior"]
    bearing_bins = max(1, int(od.get("bearing_bins", 1)))
    distance_bins = max(1, int(od.get("distance_bins", 1)))
    lat = math.radians(float((start[1] + end[1]) * 0.5))
    dx = (float(end[0]) - float(start[0])) * math.cos(lat)
    dy = float(end[1]) - float(start[1])
    angle = math.atan2(dy, dx)
    sector = int(math.floor(((angle + math.pi) / (2.0 * math.pi)) * bearing_bins))
    sector = int(np.clip(sector, 0, bearing_bins - 1))
    distance_m = float(_haversine_km_np(start[None], end[None])[0] * 1000.0)
    dbin = 0
    for threshold in od.get("distance_thresholds_m", []):
        if distance_m > float(threshold):
            dbin += 1
    dbin = int(np.clip(dbin, 0, distance_bins - 1))
    return sector, dbin


def _od_cell_prior_score(traj: np.ndarray, prior: Dict) -> float:
    traj = np.asarray(traj, dtype=np.float32)
    if len(traj) == 0:
        return float("inf")
    bounds = prior["bounds"]
    grid_size = int(prior["grid_size"])
    tokens = tokenize_trajectory(traj, bounds, grid_size)
    if not tokens:
        return float("inf")
    keep = [int(tokens[0])]
    for token in tokens[1:]:
        if int(token) != keep[-1]:
            keep.append(int(token))
    sector, dbin = _od_sector_and_distance_bin(traj[0], traj[-1], prior)
    group = prior["od_cell_prior"]["groups"].get(f"s{sector}_d{dbin}", {})
    cells = group.get("cells", {})
    default_nll = float(group.get("default_nll", prior.get("cell_default_nll", prior.get("default_nll", 10.0))))
    values = [
        float(cells.get(str(cell), {}).get("nll", default_nll))
        if isinstance(cells.get(str(cell), {}), dict)
        else float(cells.get(str(cell), default_nll))
        for cell in keep
    ]
    return float(np.mean(values)) if values else float("inf")


def _make_token_records(
    records: List[Dict],
    trajs: np.ndarray,
    bounds: Dict,
    grid_size: int,
    dataset: str,
    target_key: str,
    target_len: int,
) -> tuple[List[Dict], List[Dict]]:
    real_records = []
    gen_records = []
    for idx, (record, gen) in enumerate(zip(records, trajs)):
        real = resample_np(np.asarray(record[target_key], dtype=np.float32), target_len)
        real_tokens = tokenize_trajectory(real, bounds, grid_size)
        gen_tokens = tokenize_trajectory(gen, bounds, grid_size)
        base = {
            "sample_id": int(record.get("sample_id", idx)),
            "dataset": str(record.get("dataset", dataset)),
            "instruction": str(record.get("instruction", "")),
            "labels": dict(record.get("labels", {})),
        }
        real_records.append({**base, "grid_tokens": real_tokens, "_sequence": real_tokens})
        gen_records.append({**base, "grid_tokens": gen_tokens, "_sequence": gen_tokens})
    return real_records, gen_records


def _summarize_route_set(
    real: np.ndarray,
    generated: np.ndarray,
    records: List[Dict],
    bounds: Dict,
    args: argparse.Namespace,
    target_len: int,
) -> Dict:
    real_token_records, gen_token_records = _make_token_records(
        records=records,
        trajs=generated,
        bounds=bounds,
        grid_size=args.grid_size,
        dataset=args.dataset,
        target_key=args.target_key,
        target_len=target_len,
    )
    table1 = evaluate_distribution(real, generated, grid_num=args.grid_num, length_bins=args.length_bins)
    table2 = evaluate_sequences(real_token_records, gen_token_records)
    return {"table1": table1, "table2": table2}


def evaluate(args: argparse.Namespace) -> Dict:
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint, device, token_calibration_path=args.token_calibration_path)
    saved_args = checkpoint.get("args", {})
    attr_dim = int(saved_args.get("attr_dim", 6))
    if args.qwen_planner_checkpoint is None:
        saved_qwen_checkpoint = saved_args.get("qwen_planner_checkpoint")
        if saved_qwen_checkpoint:
            args.qwen_planner_checkpoint = Path(saved_qwen_checkpoint)
    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device) if args.qwen_planner_checkpoint else None

    dataset = InstructionTrajectoryDataset(args.input, max_samples=args.max_samples, target_key=args.target_key)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_instruction_batch,
    )
    num_candidates = int(getattr(model.decoder, "num_candidates", 1))
    target_len = int(model.decoder.target_len)
    bounds = infer_token_bounds(dataset.records[: len(dataset)], padding_ratio=args.grid_padding_ratio)
    od_prior = _load_od_prior(args.od_cell_prior_path)

    real_trajs: List[np.ndarray] = []
    selected_trajs: List[np.ndarray] = []
    oracle_trajs: List[np.ndarray] = []
    length_oracle_trajs: List[np.ndarray] = []
    token_jaccard_oracle_trajs: List[np.ndarray] = []
    control_guarded_token_jaccard_oracle_trajs: List[np.ndarray] = []
    od_cell_prior_trajs: List[np.ndarray] = []
    control_guarded_od_cell_prior_trajs: List[np.ndarray] = []
    mode_trajs: List[List[np.ndarray]] = [[] for _ in range(num_candidates)]
    selected_indices: List[int] = []
    oracle_indices: List[int] = []
    length_oracle_indices: List[int] = []
    token_jaccard_oracle_indices: List[int] = []
    control_guarded_token_jaccard_oracle_indices: List[int] = []
    od_cell_prior_indices: List[int] = []
    control_guarded_od_cell_prior_indices: List[int] = []
    control_guard_fields = _parse_fields(args.control_guard_fields)
    control_feasible_counts: List[int] = []
    selected_control_mismatches: List[int] = []
    token_control_mismatches: List[int] = []
    od_control_mismatches: List[int] = []

    with torch.no_grad():
        offset = 0
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            reference, ref_meta = prepare_reference_batch(raw_batch, target_len, device)
            target_norm, _, route_scales = prepare_eval_batch(batch, target_len)
            target_norm = target_norm.to(device)
            route_scale = route_scales[:, 0].to(device)
            attr_cond = make_attr_cond(batch, attr_dim, device)
            semantic_plan_tokens = extract_qwen_semantic_tokens(qwen_provider, batch["instruction"], device)
            pred_out = model(
                reference,
                batch["mobility_tokens"],
                attr_cond,
                semantic_plan_tokens=semantic_plan_tokens,
                return_aux=True,
            )
            if isinstance(pred_out, tuple):
                candidates, aux = pred_out
            else:
                candidates, aux = pred_out, {}
            if candidates.ndim == 3:
                candidates = candidates[:, None]
            logits = aux.get("candidate_logits") if isinstance(aux, dict) else None
            if logits is None:
                selected_idx = torch.zeros(candidates.shape[0], dtype=torch.long, device=device)
            else:
                selected_idx = logits.argmax(dim=1)
            error = torch.linalg.norm(candidates - target_norm[:, None], dim=-1) * route_scale.view(-1, 1, 1)
            ade = error.mean(dim=-1)
            oracle_idx = ade.argmin(dim=1)
            flat_candidates = candidates.reshape(candidates.shape[0] * candidates.shape[1], candidates.shape[2], 2)
            flat_reference = reference[:, None].expand(-1, candidates.shape[1], -1, -1).reshape_as(flat_candidates)
            flat_scales = route_scale[:, None].expand(-1, candidates.shape[1]).reshape(-1)
            flat_candidate_labels = contract_labels_from_tensors(
                flat_candidates.float(),
                flat_reference.float(),
                flat_scales.float(),
                dataset=args.dataset,
            )

            selected_lonlat = denormalize_predictions(_select_by_index(candidates, selected_idx), ref_meta)
            oracle_lonlat = denormalize_predictions(_select_by_index(candidates, oracle_idx), ref_meta)
            mode_lonlat = [
                denormalize_predictions(candidates[:, mode_idx], ref_meta)
                for mode_idx in range(candidates.shape[1])
            ]
            for j in range(candidates.shape[0]):
                record = dataset.records[offset + j]
                real = resample_np(np.asarray(record[args.target_key], dtype=np.float32), target_len)
                real_trajs.append(real)
                selected_trajs.append(selected_lonlat[j])
                oracle_trajs.append(oracle_lonlat[j])
                selected_indices.append(int(selected_idx[j].item()))
                oracle_indices.append(int(oracle_idx[j].item()))
                candidate_lonlat = [mode_lonlat[mode_idx][j] for mode_idx in range(candidates.shape[1])]
                real_length = _traj_length_km(real)
                cand_lengths = np.asarray([_traj_length_km(traj) for traj in candidate_lonlat], dtype=np.float32)
                length_idx = int(np.argmin(np.abs(cand_lengths - real_length)))
                length_oracle_indices.append(length_idx)
                length_oracle_trajs.append(candidate_lonlat[length_idx])

                real_tokens = tokenize_trajectory(real, bounds, args.grid_size)
                cand_jaccard = np.asarray(
                    [
                        _jaccard(real_tokens, tokenize_trajectory(traj, bounds, args.grid_size))
                        for traj in candidate_lonlat
                    ],
                    dtype=np.float32,
                )
                row_label_start = j * candidates.shape[1]
                target_labels = dict(record.get("labels", {}))
                row_candidate_labels = flat_candidate_labels[row_label_start : row_label_start + candidates.shape[1]]
                mismatch_counts = np.asarray(
                    [
                        _label_mismatch_count(
                            candidate_labels=dict(labels),
                            target_labels=target_labels,
                            fields=control_guard_fields,
                        )
                        for labels in row_candidate_labels
                    ],
                    dtype=np.int64,
                )
                control_feasible_counts.append(int((mismatch_counts <= int(args.control_guard_max_mismatches)).sum()))
                selected_control_mismatches.append(int(mismatch_counts[int(selected_idx[j].item())]))

                token_idx = int(np.argmax(cand_jaccard))
                token_jaccard_oracle_indices.append(token_idx)
                token_jaccard_oracle_trajs.append(candidate_lonlat[token_idx])
                guarded_token_idx = _select_control_guarded(
                    base_idx=int(selected_idx[j].item()),
                    mismatch_counts=mismatch_counts,
                    objective_scores=cand_jaccard,
                    maximize=True,
                    max_mismatches=int(args.control_guard_max_mismatches),
                )
                control_guarded_token_jaccard_oracle_indices.append(guarded_token_idx)
                control_guarded_token_jaccard_oracle_trajs.append(candidate_lonlat[guarded_token_idx])
                token_control_mismatches.append(int(mismatch_counts[guarded_token_idx]))

                if od_prior is not None:
                    prior_scores = np.asarray(
                        [_od_cell_prior_score(traj, od_prior) for traj in candidate_lonlat],
                        dtype=np.float32,
                    )
                    prior_idx = int(np.argmin(prior_scores))
                    od_cell_prior_indices.append(prior_idx)
                    od_cell_prior_trajs.append(candidate_lonlat[prior_idx])
                    guarded_prior_idx = _select_control_guarded(
                        base_idx=int(selected_idx[j].item()),
                        mismatch_counts=mismatch_counts,
                        objective_scores=prior_scores,
                        maximize=False,
                        max_mismatches=int(args.control_guard_max_mismatches),
                    )
                    control_guarded_od_cell_prior_indices.append(guarded_prior_idx)
                    control_guarded_od_cell_prior_trajs.append(candidate_lonlat[guarded_prior_idx])
                    od_control_mismatches.append(int(mismatch_counts[guarded_prior_idx]))
                for mode_idx in range(num_candidates):
                    mode_trajs[mode_idx].append(mode_lonlat[mode_idx][j])
            offset += candidates.shape[0]

    records = dataset.records[: len(real_trajs)]
    real_arr = np.stack(real_trajs, axis=0).astype(np.float32)
    route_sets = {
        "selected": np.stack(selected_trajs, axis=0).astype(np.float32),
        "oracle_ade": np.stack(oracle_trajs, axis=0).astype(np.float32),
        "length_oracle": np.stack(length_oracle_trajs, axis=0).astype(np.float32),
        "token_jaccard_oracle": np.stack(token_jaccard_oracle_trajs, axis=0).astype(np.float32),
        "control_guarded_token_jaccard_oracle": np.stack(
            control_guarded_token_jaccard_oracle_trajs,
            axis=0,
        ).astype(np.float32),
    }
    if od_cell_prior_trajs:
        route_sets["od_cell_prior"] = np.stack(od_cell_prior_trajs, axis=0).astype(np.float32)
    if control_guarded_od_cell_prior_trajs:
        route_sets["control_guarded_od_cell_prior"] = np.stack(
            control_guarded_od_cell_prior_trajs,
            axis=0,
        ).astype(np.float32)
    for mode_idx, trajs in enumerate(mode_trajs):
        route_sets[f"mode_{mode_idx}"] = np.stack(trajs, axis=0).astype(np.float32)

    metrics = {
        name: _summarize_route_set(real_arr, generated, records, bounds, args, target_len)
        for name, generated in route_sets.items()
    }
    metrics["selection"] = {
        "selected_index_counts": {str(i): int(selected_indices.count(i)) for i in range(num_candidates)},
        "oracle_index_counts": {str(i): int(oracle_indices.count(i)) for i in range(num_candidates)},
        "length_oracle_index_counts": {
            str(i): int(length_oracle_indices.count(i)) for i in range(num_candidates)
        },
        "token_jaccard_oracle_index_counts": {
            str(i): int(token_jaccard_oracle_indices.count(i)) for i in range(num_candidates)
        },
        "control_guarded_token_jaccard_oracle_index_counts": {
            str(i): int(control_guarded_token_jaccard_oracle_indices.count(i)) for i in range(num_candidates)
        },
        "od_cell_prior_index_counts": {
            str(i): int(od_cell_prior_indices.count(i)) for i in range(num_candidates)
        } if od_cell_prior_indices else None,
        "control_guarded_od_cell_prior_index_counts": {
            str(i): int(control_guarded_od_cell_prior_indices.count(i)) for i in range(num_candidates)
        } if control_guarded_od_cell_prior_indices else None,
        "reranker_accuracy": float(np.mean(np.asarray(selected_indices) == np.asarray(oracle_indices))),
        "control_guard_fields": control_guard_fields,
        "control_guard_max_mismatches": int(args.control_guard_max_mismatches),
        "control_feasible_count_mean": float(np.mean(control_feasible_counts)) if control_feasible_counts else 0.0,
        "control_feasible_rate": float(
            np.mean(np.asarray(control_feasible_counts, dtype=np.float32) > 0)
        ) if control_feasible_counts else 0.0,
        "selected_control_mismatch_mean": float(np.mean(selected_control_mismatches))
        if selected_control_mismatches
        else 0.0,
        "control_guarded_token_mismatch_mean": float(np.mean(token_control_mismatches))
        if token_control_mismatches
        else 0.0,
        "control_guarded_od_mismatch_mean": float(np.mean(od_control_mismatches))
        if od_control_mismatches
        else None,
    }
    metrics["metadata"] = {
        "dataset": args.dataset,
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "target_key": args.target_key,
        "num_samples": int(len(real_arr)),
        "num_candidates": int(num_candidates),
        "grid_size": int(args.grid_size),
        "grid_num": int(args.grid_num),
        "qwen_planner_checkpoint": str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None,
        "od_cell_prior_path": str(args.od_cell_prior_path) if args.od_cell_prior_path else None,
        "diagnostic_note": (
            "control_guarded_token_jaccard_oracle uses GT target tokens and is diagnostic-only, "
            "not an inference-deployable metric."
        ),
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Table1/2 route metrics for selected, oracle, and each candidate mode.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "candidate_route_distribution")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--target-key", default="trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--grid-num", type=int, default=32)
    parser.add_argument("--length-bins", type=int, default=32)
    parser.add_argument("--grid-padding-ratio", type=float, default=0.02)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--token-calibration-path", type=Path, default=None)
    parser.add_argument("--od-cell-prior-path", type=Path, default=None)
    parser.add_argument(
        "--control-guard-fields",
        default="direction,curvature,progress,reference_adherence",
        help=(
            "Comma-separated instruction fields used by diagnostic guarded selectors. "
            "Candidates with too many contract-label mismatches are filtered before route-token ranking."
        ),
    )
    parser.add_argument(
        "--control-guard-max-mismatches",
        type=int,
        default=0,
        help="Maximum allowed instruction-label mismatches for diagnostic guarded selectors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    out_dir = args.output_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "metrics.json"
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**metrics, "metrics_path": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
