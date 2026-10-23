#!/usr/bin/env python
"""Train a semantic-aware route-memory reranker for fair Table I/III/IV checks."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - progress is optional.
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langdirecttraj.counterfactual_policy import independent_collateral_field_names  # noqa: E402
from langdirecttraj.eval.distribution_metrics import (  # noqa: E402
    evaluate_distribution,
    haversine_km,
    load_fixed_bounds,
    to_grid,
)
from langdirecttraj.eval.instruction_geometry import macro_f1  # noqa: E402
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB  # noqa: E402
from scripts.eval_coarse_road_cell_stage import (  # noqa: E402
    _changed_ids,
    _mean_lookup,
    _od_group,
    _transition_ids,
    batch_refine_selection,
    build_coarse_prior,
)
from scripts.eval_route_memory_table3_table4_diagnostic import (  # noqa: E402
    build_calibrator,
    geometry_row,
    group_pairs_streaming,
    load_generated,
    record_map,
    summarize,
)
from scripts.eval_train_route_memory_prior import build_memory_candidates, load_real  # noqa: E402
from scripts.train_route_memory_reranker import (  # noqa: E402
    RouteMemoryReranker,
    compute_prior_scores,
    make_candidate_features,
    make_projection,
    query_features,
    predict,
    train_reranker,
)


def _install_pathlib_pickle_compat() -> None:
    """Allow Linux to load checkpoints pickled with Windows pathlib internals."""
    module = sys.modules.get("pathlib._local")
    if module is None:
        module = types.ModuleType("pathlib._local")
        sys.modules["pathlib._local"] = module
    module.Path = pathlib.Path
    if sys.platform == "win32":
        module.PosixPath = pathlib.WindowsPath
        module.WindowsPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath
        module.PosixPath = pathlib.PosixPath
        module.WindowsPath = pathlib.PosixPath


_install_pathlib_pickle_compat()


def safe_torch_load(path: Path, device: torch.device):
    """Load checkpoints across Windows/Linux when saved args contain pathlib paths."""
    original_posix_path = pathlib.PosixPath
    try:
        if sys.platform == "win32":
            pathlib.PosixPath = pathlib.WindowsPath
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)
    finally:
        pathlib.PosixPath = original_posix_path


CONTROLTRAJ_FAIR_FULL = {
    "Density Error": 0.004415819410470839,
    "Route Transition JSD": 0.03189801434006303,
    "Length Error": 0.00108743125276766,
}


def log_stage(message: str) -> None:
    print(f"[route-memory] {message}", flush=True)


def progress(iterable, *, desc: str, total: int | None = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, mininterval=5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-real-npz", required=True, type=Path)
    parser.add_argument("--train-query-real-npz", required=True, type=Path)
    parser.add_argument("--val-real-npz", required=True, type=Path)
    parser.add_argument("--test-real-npz", required=True, type=Path)
    parser.add_argument("--train-input", required=True, type=Path)
    parser.add_argument("--val-input", required=True, type=Path)
    parser.add_argument("--test-input", required=True, type=Path)
    parser.add_argument("--val-counterfactual-input", required=True, type=Path)
    parser.add_argument("--test-counterfactual-input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--realism-checkpoint", type=Path, default=None)
    parser.add_argument("--train-semantic-candidate-npz", type=Path, default=None)
    parser.add_argument("--val-semantic-candidate-npz", type=Path, default=None)
    parser.add_argument("--test-semantic-candidate-npz", type=Path, default=None)
    parser.add_argument("--memory-samples", type=int, default=50000)
    parser.add_argument("--train-query-samples", type=int, default=1000)
    parser.add_argument("--prior-samples", type=int, default=10000)
    parser.add_argument("--eval-start", type=int, default=0, help="Start offset for chunked eval/export splits.")
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--val-output-name", default="val1000")
    parser.add_argument("--test-output-name", default="test1000")
    parser.add_argument("--candidate-topm", type=int, default=64)
    parser.add_argument("--feature-mode", choices=["od", "od_len_angle"], default="od_len_angle")
    parser.add_argument("--retrieval-method", choices=["kdtree", "dense"], default="kdtree")
    parser.add_argument("--target-length-scale", type=float, default=1.15)
    parser.add_argument("--target-length-knn", type=int, default=16)
    parser.add_argument("--target-length-weight", type=float, default=10.0)
    parser.add_argument("--grid-num", type=int, default=32)
    parser.add_argument("--od-grid", type=int, default=6)
    parser.add_argument("--length-bins", type=int, default=32)
    parser.add_argument(
        "--bounds-json",
        type=Path,
        default=None,
        help="Preprocessing statistics JSON containing fixed lon/lat bounds.",
    )
    parser.add_argument(
        "--bounds-source",
        default=None,
        help="Stable provenance label written to the prior, checkpoint, and metrics.",
    )
    parser.add_argument(
        "--out-of-bounds-policy",
        choices=("discard", "clip", "overflow", "error"),
        default=None,
        help="Defaults to overflow with --bounds-json and discard for legacy inferred bounds.",
    )
    parser.add_argument("--min-group-count", type=int, default=3)
    parser.add_argument("--global-mix", type=float, default=0.1)
    parser.add_argument("--cell-weight", type=float, default=0.25)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--length-ratio-weight", type=float, default=2.0)
    parser.add_argument("--prototype-weight", type=float, default=0.5)
    parser.add_argument("--mode-index-weight", type=float, default=0.005)
    parser.add_argument("--semantic-weight", type=float, default=0.8)
    parser.add_argument("--semantic-score-mode", choices=["calibrator", "contract_mismatch"], default="calibrator")
    parser.add_argument("--semantic-distance-weight", type=float, default=1.0)
    parser.add_argument("--semantic-direction-weight", type=float, default=1.5)
    parser.add_argument("--semantic-curvature-weight", type=float, default=1.0)
    parser.add_argument("--semantic-progress-weight", type=float, default=1.0)
    parser.add_argument("--semantic-reference-weight", type=float, default=1.0)
    parser.add_argument("--semantic-feature-scale", type=float, default=5.0)
    parser.add_argument(
        "--disable-semantic-costs",
        action="store_true",
        help="Disable semantic candidate-cost computation for Table-I-only route-first selector experiments.",
    )
    parser.add_argument(
        "--semantic-contract-candidate-scope",
        choices=["all", "appended", "first_topk", "appended_plus_first_topk"],
        default="all",
        help="Limit expensive contract-mismatch scoring to selected candidate slots; appended means the last candidate.",
    )
    parser.add_argument(
        "--semantic-contract-topk",
        type=int,
        default=8,
        help="Number of leading route-memory candidates scored when semantic-contract-candidate-scope uses top-k.",
    )
    parser.add_argument("--selection-policy", choices=["weighted_sum", "guarded_two_stage"], default="weighted_sum")
    parser.add_argument(
        "--semantic-fallback-policy",
        choices=["none", "direction", "curvature", "dir_or_curv", "all_semantic"],
        default="none",
        help="Optionally switch selected samples to the appended semantic candidate when it repairs contract fields.",
    )
    parser.add_argument("--semantic-fallback-index", type=int, default=-1)
    parser.add_argument("--semantic-fallback-stage", choices=["before_refine", "after_refine"], default="before_refine")
    parser.add_argument("--freeze-semantic-fallback-refine", action="store_true")
    parser.add_argument(
        "--semantic-fallback-max-target-length-error",
        type=float,
        default=-1.0,
        help=(
            "Optional guard for semantic fallback. When positive, the appended semantic "
            "candidate may replace the route-memory candidate only if its relative "
            "target-length error is below this threshold."
        ),
    )
    parser.add_argument("--realism-guard-source", choices=["prior", "realism_logits"], default="prior")
    parser.add_argument("--realism-guard-topk", type=int, default=12)
    parser.add_argument("--realism-guard-score-margin", type=float, default=0.0)
    parser.add_argument("--realism-guard-length-ratio", type=float, default=0.35)
    parser.add_argument("--guarded-route-tie-weight", type=float, default=0.15)
    parser.add_argument("--guarded-invalid-penalty", type=float, default=50.0)
    parser.add_argument("--selection-target-length-weight", type=float, default=0.0)
    parser.add_argument(
        "--refine-target-length-weight",
        type=float,
        default=0.0,
        help=(
            "Add per-sample target-length mismatch to the batch-refinement cost. "
            "This matters when --batch-refine-score-source=realism would otherwise "
            "ignore candidate length compatibility."
        ),
    )
    parser.add_argument("--selection-logit-weight", type=float, default=1.0)
    parser.add_argument("--controltraj-density-threshold", type=float, default=CONTROLTRAJ_FAIR_FULL["Density Error"])
    parser.add_argument("--controltraj-route-threshold", type=float, default=CONTROLTRAJ_FAIR_FULL["Route Transition JSD"])
    parser.add_argument("--controltraj-length-threshold", type=float, default=CONTROLTRAJ_FAIR_FULL["Length Error"])
    parser.add_argument("--batch-refine-cycles", type=int, default=3)
    parser.add_argument("--batch-density-refine-penalty", type=float, default=0.10)
    parser.add_argument("--batch-length-refine-penalty", type=float, default=4.0)
    parser.add_argument("--batch-refine-topk", type=int, default=32)
    parser.add_argument("--batch-density-hot-cells", type=int, default=64)
    parser.add_argument("--batch-refine-length-target", choices=["train", "predicted"], default="train")
    parser.add_argument("--batch-refine-score-source", choices=["selection", "realism", "prior"], default="selection")
    parser.add_argument("--post-fallback-refine-cycles", type=int, default=0)
    parser.add_argument("--post-fallback-density-refine-penalty", type=float, default=-1.0)
    parser.add_argument("--post-fallback-length-refine-penalty", type=float, default=-1.0)
    parser.add_argument("--post-fallback-refine-topk", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--teacher-ce-weight", type=float, default=1.0)
    parser.add_argument("--teacher-soft-weight", type=float, default=1.0)
    parser.add_argument("--teacher-score-weight", type=float, default=0.2)
    parser.add_argument("--teacher-temperature", type=float, default=0.35)
    parser.add_argument(
        "--teacher-score-mode",
        choices=["prior_semantic", "route_first_soft"],
        default="prior_semantic",
        help="Teacher score construction. route_first_soft exposes Density/Route/Length/Semantic costs separately.",
    )
    parser.add_argument("--teacher-density-weight", type=float, default=1.0)
    parser.add_argument("--teacher-density-marginal-weight", type=float, default=1.0)
    parser.add_argument("--teacher-transition-weight", type=float, default=1.0)
    parser.add_argument("--teacher-length-weight", type=float, default=1.0)
    parser.add_argument(
        "--teacher-raw-length-weight",
        type=float,
        default=0.0,
        help=(
            "Additional absolute target-length error weight in the route-first soft teacher. "
            "This complements row-normalized length ranking with an absolute per-sample length constraint."
        ),
    )
    parser.add_argument(
        "--teacher-length-marginal-weight",
        type=float,
        default=0.0,
        help="Batch-level train length-distribution marginal cost used by the route-first soft teacher.",
    )
    parser.add_argument(
        "--teacher-use-real-train-length",
        action="store_true",
        help=(
            "Use the observed train trajectory length when building train-split teacher "
            "labels. This is valid supervision for training and is never used for val/test selection."
        ),
    )
    parser.add_argument("--teacher-semantic-cost-weight", type=float, default=0.5)
    parser.add_argument(
        "--student-density-cost-weight",
        type=float,
        default=0.0,
        help="Train the selector to minimize expected train-density candidate cost under its predicted route distribution.",
    )
    parser.add_argument(
        "--student-transition-cost-weight",
        type=float,
        default=0.0,
        help="Train the selector to minimize expected route-transition candidate cost under its predicted route distribution.",
    )
    parser.add_argument(
        "--student-length-cost-weight",
        type=float,
        default=0.0,
        help="Train the selector to minimize expected target-length candidate cost under its predicted route distribution.",
    )
    parser.add_argument(
        "--student-raw-length-cost-weight",
        type=float,
        default=0.0,
        help="Train the selector to minimize expected absolute target-length error in addition to row-normalized length ranking.",
    )
    parser.add_argument("--prior-score-feature", action="store_true")
    parser.add_argument(
        "--table1-cost-feature",
        action="store_true",
        help=(
            "Append train-prior Table-I cost components to each candidate feature: "
            "density, transition, length, and batch-density marginal costs. These "
            "are computed from train priors, OD, and candidates only."
        ),
    )
    parser.add_argument(
        "--disable-length-transition-calibration",
        action="store_true",
        help=(
            "Matched Stage-2 ablation switch. Keep route memory, density prior, "
            "candidate pool, and selector training intact, but remove route-transition "
            "and target-length calibration evidence from Table-I feature channels, "
            "route-first soft teacher scores, and student expected-cost losses."
        ),
    )
    parser.add_argument("--eval-batch-refine", action="store_true")
    parser.add_argument(
        "--route-memory-feature-cache",
        type=Path,
        default=None,
        help="Optional npz cache for candidates, features, prior scores, and semantic scores.",
    )
    parser.add_argument(
        "--reuse-route-memory-feature-cache",
        action="store_true",
        help="Reuse --route-memory-feature-cache when it exists instead of rebuilding expensive full-eval features.",
    )
    parser.add_argument(
        "--route-memory-prior-cache",
        type=Path,
        default=None,
        help="Optional npz cache for train-prior target lengths and prior scores.",
    )
    parser.add_argument(
        "--reuse-route-memory-prior-cache",
        action="store_true",
        help="Reuse --route-memory-prior-cache when it exists instead of recomputing train-prior scores.",
    )
    parser.add_argument(
        "--route-memory-candidate-cache",
        type=Path,
        default=None,
        help="Optional resumable cache for route-memory candidates before semantic candidate append.",
    )
    parser.add_argument(
        "--reuse-route-memory-candidate-cache",
        action="store_true",
        help="Resume/reuse --route-memory-candidate-cache parts when building route-memory candidates.",
    )
    parser.add_argument("--candidate-cache-chunk-size", type=int, default=2048)
    parser.add_argument("--train-route-memory-candidate-cache", type=Path, default=None)
    parser.add_argument("--reuse-train-route-memory-candidate-cache", action="store_true")
    parser.add_argument("--train-route-memory-prior-cache", type=Path, default=None)
    parser.add_argument("--reuse-train-route-memory-prior-cache", action="store_true")
    parser.add_argument(
        "--stream-train-features",
        action="store_true",
        help=(
            "Build full train-query selector features/teachers as on-disk memmaps in chunks. "
            "This keeps the paper protocol full-data while avoiding large all-at-once RAM tensors."
        ),
    )
    parser.add_argument("--train-feature-cache", type=Path, default=None)
    parser.add_argument("--reuse-train-feature-cache", action="store_true")
    parser.add_argument("--train-feature-chunk-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--counterfactual-scan-limit", type=int, default=200000)
    parser.add_argument("--table4-selection-policy", choices=["model", "target_collateral_guard"], default="model")
    parser.add_argument("--table4-priority", choices=["target_first", "collateral_first"], default="target_first")
    parser.add_argument("--table4-target-margin", type=float, default=0.0)
    parser.add_argument("--table4-collateral-weight", type=float, default=0.25)
    parser.add_argument("--table4-raw-collateral-weight", type=float, default=0.05)
    parser.add_argument("--table4-collateral-margin", type=float, default=0.08)
    parser.add_argument("--table4-skip-collateral-above", type=float, default=-1.0)
    parser.add_argument(
        "--table4-fields",
        default="",
        help="Comma-separated counterfactual fields to evaluate; empty keeps all fields.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=["both", "val", "test"], default="both")
    parser.add_argument("--skip-table3", action="store_true")
    parser.add_argument("--skip-table4", action="store_true")
    parser.add_argument(
        "--save-uncompressed-trajectories",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed for faster large chunk exports.",
    )
    return parser.parse_args()


def disable_length_transition_calibration(args) -> bool:
    return bool(getattr(args, "disable_length_transition_calibration", False))


def prior_out_of_bounds_policy(prior: Mapping) -> str:
    return str(prior.get("out_of_bounds_policy", "discard"))


def prior_num_cells(prior: Mapping) -> int:
    grid_num = int(prior["grid_num"])
    return int(prior.get("num_cells", grid_num * grid_num))


def effective_transition_weight(args) -> float:
    return 0.0 if disable_length_transition_calibration(args) else float(args.teacher_transition_weight)


def effective_length_weight(args) -> float:
    return 0.0 if disable_length_transition_calibration(args) else float(args.teacher_length_weight)


def effective_raw_length_weight(args) -> float:
    return 0.0 if disable_length_transition_calibration(args) else float(args.teacher_raw_length_weight)


def effective_length_marginal_weight(args) -> float:
    return 0.0 if disable_length_transition_calibration(args) else float(getattr(args, "teacher_length_marginal_weight", 0.0))


def load_real_window(path: Path, n: int, start: int = 0) -> np.ndarray:
    payload = np.load(path)
    traj = np.asarray(payload["real_trajectories"], dtype=np.float32)
    start = max(0, int(start))
    if n > 0:
        return traj[start : start + int(n)]
    return traj[start:]


def real_sample_ids(path: Path, n: int, start: int = 0) -> np.ndarray:
    payload = np.load(path)
    ids = np.asarray(payload["sample_id"], dtype=np.int64)
    start = max(0, int(start))
    return ids[start : start + int(n)] if n > 0 else ids[start:]


def onehot_labels(records: List[Mapping]) -> np.ndarray:
    rows: List[List[float]] = []
    for record in records:
        labels = record.get("labels", {})
        row: List[float] = []
        for field in DEFAULT_MOBILITY_VOCAB.fields:
            token_id = DEFAULT_MOBILITY_VOCAB.token_to_id[field].get(str(labels.get(field, "")), 0)
            size = len(DEFAULT_MOBILITY_VOCAB.token_to_id[field])
            row.extend([1.0 if i == token_id else 0.0 for i in range(size)])
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def semantic_scores(
    candidates: np.ndarray,
    records: List[Mapping],
    calibrator,
    *,
    distance_weight: float,
    score_mode: str,
    direction_weight: float,
    curvature_weight: float,
    progress_weight: float,
    reference_weight: float,
    feature_scale: float,
    candidate_scope: str = "all",
    contract_topk: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    n, k = candidates.shape[:2]
    per_field = np.zeros((n, k, len(DEFAULT_MOBILITY_VOCAB.fields)), dtype=np.float32)
    if str(score_mode) == "contract_mismatch":
        from langdirecttraj.eval.export_model_predictions import reference_to_local, resample_np
        from langdirecttraj.eval.instruction_geometry import (
            add_pace_metrics_to_rows,
            apply_pace_progress_labels,
            contract_labels_from_tensors,
            geometry_values,
            tensor_dict_to_rows,
        )

        scope = str(candidate_scope)
        if scope == "all":
            eval_cols = np.arange(k, dtype=np.int64)
        elif scope == "appended":
            eval_cols = np.asarray([k - 1], dtype=np.int64)
            per_field.fill(0.5)
        elif scope == "first_topk":
            topk = max(1, min(int(contract_topk), k))
            eval_cols = np.arange(topk, dtype=np.int64)
            per_field.fill(0.5)
        elif scope == "appended_plus_first_topk":
            topk = max(1, min(int(contract_topk), k))
            eval_cols = np.unique(np.concatenate([np.arange(topk, dtype=np.int64), np.asarray([k - 1], dtype=np.int64)]))
            per_field.fill(0.5)
        else:
            raise ValueError(f"Unknown semantic contract candidate scope: {candidate_scope}")
        eval_k = int(len(eval_cols))
        log_stage(f"semantic contract scope={scope} evaluated_candidates_per_sample={eval_k}/{k}")

        target_len = int(candidates.shape[2])
        flat_n = int(n * eval_k)
        points_norm = np.empty((flat_n, target_len, 2), dtype=np.float32)
        refs_norm = np.empty((flat_n, target_len, 2), dtype=np.float32)
        scales = np.empty((flat_n,), dtype=np.float32)
        flat_rows = np.empty((flat_n,), dtype=np.int64)
        flat_cols = np.empty((flat_n,), dtype=np.int64)
        for i, record in progress(enumerate(records), desc="semantic prep records", total=len(records)):
            reference = np.asarray(record["reference"], dtype=np.float32)
            ref_norm, origin, scale, route_scale = reference_to_local(reference)
            pts = np.asarray(candidates[i, eval_cols], dtype=np.float32)
            offset = i * eval_k
            points_norm[offset : offset + eval_k] = (
                ((pts - origin[None, None, :]) * scale[None, None, :])
                / max(float(route_scale), 1e-6)
            ).astype(np.float32)
            refs_norm[offset : offset + eval_k] = np.repeat(
                resample_np(ref_norm, target_len)[None, :, :],
                eval_k,
                axis=0,
            ).astype(np.float32)
            scales[offset : offset + eval_k] = float(route_scale)
            flat_rows[offset : offset + eval_k] = i
            flat_cols[offset : offset + eval_k] = eval_cols

        chunk_size = 4096
        dataset = str(records[0].get("dataset", "porto") or "porto") if records else "porto"
        for start in progress(range(0, flat_n, chunk_size), desc="semantic contract chunks", total=(flat_n + chunk_size - 1) // chunk_size):
            end = min(flat_n, start + chunk_size)
            points_t = torch.from_numpy(points_norm[start:end]).float()
            reference_t = torch.from_numpy(refs_norm[start:end]).float()
            scale_t = torch.from_numpy(scales[start:end]).float()
            pred_labels = contract_labels_from_tensors(
                points_t,
                reference_t,
                scale_t,
                dataset=dataset,
            )
            geom_rows = tensor_dict_to_rows(geometry_values(points_t, reference_t, scale_t))
            attrs = [records[int(row)].get("trip_attributes", {}) for row in flat_rows[start:end]]
            geom_rows = add_pace_metrics_to_rows(geom_rows, attrs)
            pred_labels = apply_pace_progress_labels(pred_labels, geom_rows, attrs)
            for local_idx, pred in enumerate(pred_labels):
                flat_idx = start + local_idx
                row = int(flat_rows[flat_idx])
                col = int(flat_cols[flat_idx])
                labels = records[row].get("labels", {})
                for f_idx, field in enumerate(DEFAULT_MOBILITY_VOCAB.fields):
                    token = str(labels.get(field, "unknown"))
                    per_field[row, col, f_idx] = (
                        0.0 if str(pred.get(field, "unknown")) == token else 1.0
                    )
    else:
        for i, record in progress(enumerate(records), desc="semantic calibrator records", total=len(records)):
            labels = record.get("labels", {})
            for j in range(k):
                geom = geometry_row(candidates[i, j], record, candidates.shape[2])
                for f_idx, field in enumerate(DEFAULT_MOBILITY_VOCAB.fields):
                    token = str(labels.get(field, "unknown"))
                    per_field[i, j, f_idx] = float(calibrator.score_token(field, token, geom))
    weights = np.ones(len(DEFAULT_MOBILITY_VOCAB.fields), dtype=np.float32)
    field_weights = {
        "distance": float(distance_weight),
        "direction": float(direction_weight),
        "curvature": float(curvature_weight),
        "progress": float(progress_weight),
        "reference_adherence": float(reference_weight),
    }
    for field, weight in field_weights.items():
        weights[DEFAULT_MOBILITY_VOCAB.fields.index(field)] = weight
    total = (np.clip(per_field, 0.0, float(feature_scale)) * weights[None, None, :]).mean(axis=2)
    return total.astype(np.float32), (per_field / max(float(feature_scale), 1e-6)).astype(np.float32)


def add_semantic_features(base: np.ndarray, label_feat: np.ndarray, semantic_per_field: np.ndarray) -> np.ndarray:
    n, k = base.shape[:2]
    return np.concatenate(
        [
            base,
            np.repeat(label_feat[:, None, :], k, axis=1),
            semantic_per_field,
        ],
        axis=2,
    ).astype(np.float32)


def build_records(input_path: Path, real_path: Path, n: int, start: int = 0) -> tuple[np.ndarray, List[Dict]]:
    ids = real_sample_ids(real_path, n, start=start)
    records_by_id = record_map(input_path, ids)
    return ids, [records_by_id[int(sid)] for sid in ids.tolist()]


def append_semantic_candidate(
    candidates: np.ndarray,
    memory_idx: np.ndarray,
    sample_ids: np.ndarray,
    candidate_npz: Path | None,
) -> tuple[np.ndarray, np.ndarray, Dict]:
    if candidate_npz is None:
        return candidates, memory_idx, {"semantic_candidate_appended": False}
    payload = np.load(candidate_npz)
    key = "trajectories" if "trajectories" in payload else "generated_trajectories"
    if key not in payload:
        raise ValueError(f"{candidate_npz} must contain trajectories or generated_trajectories")
    external = np.asarray(payload[key], dtype=np.float32)
    if "sample_ids" in payload:
        ext_ids = np.asarray(payload["sample_ids"], dtype=np.int64)
    elif "sample_id" in payload:
        ext_ids = np.asarray(payload["sample_id"], dtype=np.int64)
    else:
        ext_ids = sample_ids
    by_id = {int(sid): i for i, sid in enumerate(ext_ids.tolist())}
    missing = [int(sid) for sid in sample_ids.tolist() if int(sid) not in by_id]
    if missing:
        raise ValueError(f"Missing {len(missing)} semantic candidates in {candidate_npz}; first missing={missing[:5]}")
    aligned = np.stack([external[by_id[int(sid)]] for sid in sample_ids.tolist()], axis=0).astype(np.float32)
    if aligned.shape[1:] != candidates.shape[2:]:
        raise ValueError(f"Semantic candidate shape {aligned.shape[1:]} does not match candidates {candidates.shape[2:]}")
    augmented_candidates = np.concatenate([candidates, aligned[:, None, :, :]], axis=1)
    sentinel = np.full((memory_idx.shape[0], 1), max(int(memory_idx.max()), 0) + 1, dtype=memory_idx.dtype)
    augmented_memory_idx = np.concatenate([memory_idx, sentinel], axis=1)
    return augmented_candidates, augmented_memory_idx, {
        "semantic_candidate_appended": True,
        "semantic_candidate_npz": str(candidate_npz),
        "candidate_topm_after_append": int(augmented_candidates.shape[1]),
    }


def table3_from_generated(generated: np.ndarray, records: List[Mapping]) -> Dict:
    true_by_field: Dict[str, List[str]] = {field: [] for field in DEFAULT_MOBILITY_VOCAB.fields}
    pred_by_field: Dict[str, List[str]] = {field: [] for field in DEFAULT_MOBILITY_VOCAB.fields}
    for traj, record in progress(zip(generated, records), desc="table3 instruction labels", total=len(records)):
        geom = geometry_row(traj, record, generated.shape[1])
        # Use the same calibrator-free contract labels as the diagnostic script.
        # This keeps the result comparable to the previous route-memory check.
        from langdirecttraj.eval.export_model_predictions import reference_to_local, resample_np
        from langdirecttraj.eval.instruction_geometry import (
            add_pace_metrics_to_rows,
            apply_pace_progress_labels,
            contract_labels_from_tensors,
        )

        ref_norm, origin, scale, route_scale = reference_to_local(np.asarray(record["reference"], dtype=np.float32))
        pts = resample_np(np.asarray(traj, dtype=np.float32), generated.shape[1])
        pts_norm = ((pts - origin[None, :]) * scale[None, :]) / max(float(route_scale), 1e-6)
        pred_labels = contract_labels_from_tensors(
            torch.from_numpy(pts_norm[None]).float(),
            torch.from_numpy(resample_np(ref_norm, generated.shape[1])[None]).float(),
            torch.tensor([route_scale], dtype=torch.float32),
            dataset=str(record.get("dataset", "porto") or "porto"),
        )[0]
        pred_labels = apply_pace_progress_labels([pred_labels], [geom], [record.get("trip_attributes", {})])[0]
        for field in DEFAULT_MOBILITY_VOCAB.fields:
            true_by_field[field].append(str(record.get("labels", {}).get(field, "unknown")))
            pred_by_field[field].append(str(pred_labels[field]))
    field_metrics = {
        field: macro_f1(
            true_by_field[field],
            pred_by_field[field],
            DEFAULT_MOBILITY_VOCAB.token_to_id[field].keys(),
        )
        for field in DEFAULT_MOBILITY_VOCAB.fields
    }
    table3 = {
        "distance_f1": float(field_metrics["distance"]["macro_f1"]),
        "direction_f1": float(field_metrics["direction"]["macro_f1"]),
        "curvature_f1": float(field_metrics["curvature"]["macro_f1"]),
        "progress_f1": float(field_metrics["progress"]["macro_f1"]),
        "reference_adherence_f1": float(field_metrics["reference_adherence"]["macro_f1"]),
    }
    table3["mean_instruction_following_f1"] = float(np.mean(list(table3.values())))
    return {"table3_instruction_following": table3, "field_metrics": field_metrics}


def make_features_for_records(args, real, candidates, memory_idx, records, prior_train, prior, proj, q_stats, calibrator):
    sample_ids = np.asarray(getattr(args, "_current_sample_ids", np.arange(len(real))), dtype=np.int64)
    prior_cache_path = getattr(args, "route_memory_prior_cache", None)
    use_prior_cache = bool(
        getattr(args, "reuse_route_memory_prior_cache", False)
        and prior_cache_path is not None
        and Path(prior_cache_path).exists()
    )
    if use_prior_cache:
        log_stage(f"prior-cache: reuse {prior_cache_path}")
        target_lengths, prior_scores = load_prior_cache(Path(prior_cache_path), sample_ids, candidates.shape[:2])
    else:
        log_stage("features: compute train-prior scores")
        _prior_selected, target_lengths, prior_scores, _prior_stats = compute_prior_scores(args, prior_train, real, candidates, prior)
        if prior_cache_path is not None:
            save_prior_cache(
                Path(prior_cache_path),
                sample_ids=sample_ids,
                candidate_shape=np.asarray(candidates.shape[:2], dtype=np.int64),
                target_lengths=target_lengths,
                prior_scores=prior_scores,
            )
    log_stage("features: build base candidate feature tensor")
    base, _ = make_candidate_features(
        real,
        candidates,
        memory_idx,
        target_lengths,
        proj,
        args.feature_mode,
        q_stats,
        prior_scores if args.prior_score_feature else None,
    )
    if bool(args.disable_semantic_costs):
        log_stage("features: semantic candidate scores disabled")
        sem_total = np.zeros(candidates.shape[:2], dtype=np.float32)
        sem_fields = np.zeros((*candidates.shape[:2], len(DEFAULT_MOBILITY_VOCAB.fields)), dtype=np.float32)
    else:
        log_stage("features: compute semantic candidate scores")
        sem_total, sem_fields = semantic_scores(
            candidates,
            records,
            calibrator,
            distance_weight=float(args.semantic_distance_weight),
            score_mode=str(args.semantic_score_mode),
            direction_weight=float(args.semantic_direction_weight),
            curvature_weight=float(args.semantic_curvature_weight),
            progress_weight=float(args.semantic_progress_weight),
            reference_weight=float(args.semantic_reference_weight),
            feature_scale=float(args.semantic_feature_scale),
            candidate_scope=str(args.semantic_contract_candidate_scope),
            contract_topk=int(args.semantic_contract_topk),
        )
    if bool(getattr(args, "table1_cost_feature", False)):
        log_stage("features: append Table-I cost features")
        table1_features = table1_cost_features(
            args,
            prior_train,
            prior,
            candidates,
            target_lengths,
            sem_total,
        )
        base = np.concatenate([base, table1_features], axis=2).astype(np.float32)
    log_stage("features: build semantic label features")
    label_feat = onehot_labels(records)
    log_stage("features: concatenate final feature tensor")
    return add_semantic_features(base, label_feat, sem_fields), prior_scores, sem_total, target_lengths, sem_fields


def query_feature_stats_only(real: np.ndarray, proj, mode: str) -> np.ndarray:
    raw = query_features(real, proj, mode).astype(np.float32)
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True).clip(1e-4)
    return np.concatenate([mean, std], axis=0).astype(np.float32)


def _table1_primitives(
    args,
    prior: Dict,
    candidates: np.ndarray,
    target_lengths: np.ndarray,
    *,
    keep_density: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, k_paths = candidates.shape[:2]
    grid_num = int(prior["grid_num"])
    out_of_bounds_policy = prior_out_of_bounds_policy(prior)
    num_cells = prior_num_cells(prior)
    bounds = prior["bounds"]
    density_nll = np.zeros((n, k_paths), dtype=np.float32)
    transition_nll = np.zeros((n, k_paths), dtype=np.float32)
    candidate_density = np.zeros((n, k_paths, num_cells), dtype=np.float32) if keep_density else np.zeros((0, 0, 0), dtype=np.float32)

    for i in range(n):
        group = _od_group(candidates[i, 0], bounds, int(prior["od_grid"]), out_of_bounds_policy)
        use_group = int(prior["group_count"].get(group, 0)) >= int(args.min_group_count)
        cell_nll = prior["group_cell_nll"].get(group, prior["global_cell_nll"]) if use_group else prior["global_cell_nll"]
        trans_nll = (
            prior["group_transition_nll"].get(group, prior["global_transition_nll"])
            if use_group
            else prior["global_transition_nll"]
        )
        for k in range(k_paths):
            traj = candidates[i, k]
            ids = _changed_ids(traj, bounds, grid_num, out_of_bounds_policy)
            transitions = _transition_ids(ids, num_cells)
            density_nll[i, k] = _mean_lookup(cell_nll, ids)
            transition_nll[i, k] = _mean_lookup(trans_nll, transitions)
            if keep_density:
                candidate_density[i, k] = to_grid(
                    candidates[i, k : k + 1],
                    bounds,
                    grid_num,
                    out_of_bounds_policy,
                ).astype(np.float32)

    length_cost = candidate_target_length_error(candidates, target_lengths)
    return density_nll, transition_nll, length_cost, candidate_density


def _stream_seed_overfull(
    args,
    prior_train: np.ndarray,
    prior: Dict,
    train_query: np.ndarray,
    train_candidates: np.ndarray,
    *,
    use_real_lengths: bool,
    chunk_size: int,
) -> np.ndarray:
    grid_num = int(prior["grid_num"])
    out_of_bounds_policy = prior_out_of_bounds_policy(prior)
    num_cells = prior_num_cells(prior)
    selected_density = np.zeros((num_cells,), dtype=np.float64)
    ranges = list(range(0, len(train_query), max(1, int(chunk_size))))
    for start in progress(ranges, desc="stream teacher density pass", total=len(ranges)):
        end = min(len(train_query), start + max(1, int(chunk_size)))
        candidates_chunk = np.asarray(train_candidates[start:end], dtype=np.float32)
        if use_real_lengths:
            target_lengths = haversine_km(train_query[start:end, :-1], train_query[start:end, 1:]).sum(axis=1).astype(np.float32)
        else:
            target_lengths, _prior_scores = compute_prior_scores(args, prior_train, train_query[start:end], candidates_chunk, prior)[1:3]
        _density_nll, transition_nll, length_cost, candidate_density = _table1_primitives(
            args,
            prior,
            candidates_chunk,
            target_lengths,
            keep_density=True,
        )
        seed_cost = (
            effective_transition_weight(args) * normalize_rows(transition_nll)
            + effective_length_weight(args) * normalize_rows(length_cost)
        )
        seed_selected = seed_cost.argmin(axis=1)
        selected_density += candidate_density[np.arange(end - start), seed_selected].sum(axis=0).astype(np.float64)
    train_density = to_grid(
        prior_train,
        prior["bounds"],
        grid_num,
        out_of_bounds_policy,
    ).astype(np.float64)
    target_prob = train_density / max(float(train_density.sum()), 1.0)
    selected_prob = selected_density / max(float(selected_density.sum()), 1.0)
    return (selected_prob - target_prob).astype(np.float32)


def _table1_features_from_primitives(
    args,
    density_nll: np.ndarray,
    transition_nll: np.ndarray,
    length_cost: np.ndarray,
    candidate_density: np.ndarray,
    overfull: np.ndarray,
) -> np.ndarray:
    density_marginal = np.einsum("nkd,d->nk", candidate_density, overfull).astype(np.float32)
    if disable_length_transition_calibration(args):
        zero = np.zeros_like(density_nll, dtype=np.float32)
        feature_blocks = [
            normalize_rows(density_nll),
            zero,
            zero,
            normalize_rows(density_marginal),
            zero,
        ]
        return np.stack(feature_blocks, axis=2).astype(np.float32)
    feature_blocks = [
        normalize_rows(density_nll),
        normalize_rows(transition_nll),
        normalize_rows(length_cost),
        normalize_rows(density_marginal),
        np.clip(length_cost, 0.0, 2.0),
    ]
    return np.stack(feature_blocks, axis=2).astype(np.float32)


def _route_first_teacher_from_primitives(
    args,
    density_nll: np.ndarray,
    transition_nll: np.ndarray,
    length_cost: np.ndarray,
    candidate_density: np.ndarray,
    teacher_overfull: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    density_marginal = np.einsum("nkd,d->nk", candidate_density, teacher_overfull).astype(np.float32)
    transition_cost = normalize_rows(transition_nll)
    target_length_cost = normalize_rows(length_cost)
    raw_target_length_cost = np.clip(length_cost, 0.0, 2.0).astype(np.float32)
    density_cost = normalize_rows(density_nll) + float(args.teacher_density_marginal_weight) * normalize_rows(density_marginal)
    if disable_length_transition_calibration(args):
        transition_cost = np.zeros_like(transition_cost, dtype=np.float32)
        target_length_cost = np.zeros_like(target_length_cost, dtype=np.float32)
        raw_target_length_cost = np.zeros_like(raw_target_length_cost, dtype=np.float32)
    length_total_cost = normalize_rows(length_cost) + effective_raw_length_weight(args) * raw_target_length_cost
    if disable_length_transition_calibration(args):
        length_total_cost = np.zeros_like(length_total_cost, dtype=np.float32)
    total = (
        float(args.teacher_density_weight) * density_cost
        + effective_transition_weight(args) * transition_cost
        + effective_length_weight(args) * length_total_cost
    ).astype(np.float32)
    component_scores = np.stack(
        [density_cost, transition_cost, target_length_cost, raw_target_length_cost],
        axis=2,
    ).astype(np.float32)
    selected = total.argmin(axis=1).astype(np.int64)
    row = np.arange(len(selected))
    stats = {
        "teacher_selected_density_cost_mean": float(density_cost[row, selected].mean()),
        "teacher_selected_transition_cost_mean": float(transition_nll[row, selected].mean()),
        "teacher_selected_length_cost_mean": float(length_cost[row, selected].mean()),
        "disable_length_transition_calibration": bool(disable_length_transition_calibration(args)),
        "effective_teacher_transition_weight": float(effective_transition_weight(args)),
        "effective_teacher_length_weight": float(effective_length_weight(args)),
        "effective_teacher_raw_length_weight": float(effective_raw_length_weight(args)),
    }
    return selected, total, component_scores, stats


def build_stream_train_feature_cache(
    args,
    train_query: np.ndarray,
    train_candidates: np.ndarray,
    train_memory_idx: np.ndarray,
    train_records: List[Mapping],
    prior_train: np.ndarray,
    prior: Dict,
    proj,
    q_stats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, Dict]:
    if args.train_feature_cache is None:
        raise ValueError("--stream-train-features requires --train-feature-cache")
    if not bool(args.disable_semantic_costs):
        raise ValueError("--stream-train-features currently supports Table-I training with --disable-semantic-costs")
    if str(args.teacher_score_mode) != "route_first_soft":
        raise ValueError("--stream-train-features currently supports --teacher-score-mode route_first_soft")
    if abs(float(getattr(args, "teacher_length_marginal_weight", 0.0))) > 0.0:
        raise ValueError("--stream-train-features currently requires --teacher-length-marginal-weight 0.0")
    if args.train_semantic_candidate_npz is not None:
        raise ValueError("--stream-train-features does not support appended train semantic candidates")

    cache_dir = Path(str(args.train_feature_cache) + ".parts")
    manifest_path = cache_dir / "manifest.json"
    n, k = train_candidates.shape[:2]
    chunk_size = max(1, int(args.train_feature_chunk_size))
    if bool(args.reuse_train_feature_cache) and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache_ablation_flag = bool(manifest.get("disable_length_transition_calibration", False))
        current_ablation_flag = bool(disable_length_transition_calibration(args))
        if cache_ablation_flag != current_ablation_flag:
            raise ValueError(
                f"{cache_dir} calibration-ablation flag mismatch: "
                f"cache={cache_ablation_flag} current={current_ablation_flag}. "
                "Use a separate --train-feature-cache for this matched ablation."
            )
        if bool(manifest.get("complete", False)) and int(manifest.get("n", -1)) == n and int(manifest.get("k", -1)) == k:
            log_stage(f"stream-cache: reuse complete {cache_dir}")
            feature_shape = tuple(int(v) for v in manifest["feature_shape"])
            component_shape = tuple(int(v) for v in manifest["component_shape"])
            features = np.memmap(cache_dir / "features.float32.dat", mode="r", dtype=np.float32, shape=feature_shape)
            teacher = np.memmap(cache_dir / "teacher.int64.dat", mode="r", dtype=np.int64, shape=(n,))
            teacher_scores = np.memmap(cache_dir / "teacher_scores.float32.dat", mode="r", dtype=np.float32, shape=(n, k))
            teacher_components = np.memmap(cache_dir / "teacher_components.float32.dat", mode="r", dtype=np.float32, shape=component_shape)
            prior_scores = np.memmap(cache_dir / "prior_scores.float32.dat", mode="r", dtype=np.float32, shape=(n, k))
            target_lengths = np.memmap(cache_dir / "target_lengths.float32.dat", mode="r", dtype=np.float32, shape=(n,))
            teacher_stats = json.loads((cache_dir / "teacher_stats.json").read_text(encoding="utf-8"))
            return features, teacher, teacher_scores, teacher_components, prior_scores, target_lengths, teacher_stats, {
                "semantic_candidate_appended": False,
                "stream_train_feature_cache": str(cache_dir),
                "stream_train_feature_cache_reused": True,
            }

    cache_dir.mkdir(parents=True, exist_ok=True)
    ranges = list(range(0, n, chunk_size))
    log_stage("stream-cache: build feature density overfull pass")
    feature_overfull = _stream_seed_overfull(
        args,
        prior_train,
        prior,
        train_query,
        train_candidates,
        use_real_lengths=False,
        chunk_size=chunk_size,
    )
    log_stage("stream-cache: build teacher density overfull pass")
    teacher_overfull = _stream_seed_overfull(
        args,
        prior_train,
        prior,
        train_query,
        train_candidates,
        use_real_lengths=bool(args.teacher_use_real_train_length),
        chunk_size=chunk_size,
    )

    first_end = min(n, chunk_size)
    first_candidates = np.asarray(train_candidates[:first_end], dtype=np.float32)
    _sel, first_target_lengths, first_prior_scores, _stats = compute_prior_scores(
        args,
        prior_train,
        train_query[:first_end],
        first_candidates,
        prior,
    )
    first_base, _ = make_candidate_features(
        train_query[:first_end],
        first_candidates,
        np.asarray(train_memory_idx[:first_end], dtype=np.int64),
        first_target_lengths,
        proj,
        args.feature_mode,
        q_stats,
        first_prior_scores if args.prior_score_feature else None,
    )
    first_density, first_transition, first_length, first_candidate_density = _table1_primitives(
        args,
        prior,
        first_candidates,
        first_target_lengths,
        keep_density=bool(getattr(args, "table1_cost_feature", False)),
    )
    if bool(getattr(args, "table1_cost_feature", False)):
        first_base = np.concatenate(
            [
                first_base,
                _table1_features_from_primitives(
                    args,
                    first_density,
                    first_transition,
                    first_length,
                    first_candidate_density,
                    feature_overfull,
                ),
            ],
            axis=2,
        ).astype(np.float32)
    first_label = onehot_labels(train_records[:first_end])
    first_sem_fields = np.zeros((*first_candidates.shape[:2], len(DEFAULT_MOBILITY_VOCAB.fields)), dtype=np.float32)
    first_features = add_semantic_features(first_base, first_label, first_sem_fields)
    feature_shape = (n, k, first_features.shape[2])
    component_shape = (n, k, 4)
    features = np.memmap(cache_dir / "features.float32.dat", mode="w+", dtype=np.float32, shape=feature_shape)
    teacher = np.memmap(cache_dir / "teacher.int64.dat", mode="w+", dtype=np.int64, shape=(n,))
    teacher_scores = np.memmap(cache_dir / "teacher_scores.float32.dat", mode="w+", dtype=np.float32, shape=(n, k))
    teacher_components = np.memmap(cache_dir / "teacher_components.float32.dat", mode="w+", dtype=np.float32, shape=component_shape)
    prior_scores_mm = np.memmap(cache_dir / "prior_scores.float32.dat", mode="w+", dtype=np.float32, shape=(n, k))
    target_lengths_mm = np.memmap(cache_dir / "target_lengths.float32.dat", mode="w+", dtype=np.float32, shape=(n,))
    teacher_target_lengths_mm = np.memmap(cache_dir / "teacher_target_lengths.float32.dat", mode="w+", dtype=np.float32, shape=(n,))

    aggregate = {
        "teacher_score_mode": "route_first_soft",
        "teacher_target_length_source": "train_real" if bool(args.teacher_use_real_train_length) else "train_prior_predicted",
        "teacher_density_weight": float(args.teacher_density_weight),
        "teacher_density_marginal_weight": float(args.teacher_density_marginal_weight),
        "teacher_transition_weight": float(args.teacher_transition_weight),
        "teacher_length_weight": float(args.teacher_length_weight),
        "teacher_raw_length_weight": float(args.teacher_raw_length_weight),
        "teacher_length_marginal_weight": float(args.teacher_length_marginal_weight),
        "disable_length_transition_calibration": bool(disable_length_transition_calibration(args)),
        "effective_teacher_transition_weight": float(effective_transition_weight(args)),
        "effective_teacher_length_weight": float(effective_length_weight(args)),
        "effective_teacher_raw_length_weight": float(effective_raw_length_weight(args)),
        "effective_teacher_length_marginal_weight": float(effective_length_marginal_weight(args)),
        "teacher_semantic_cost_weight": float(args.teacher_semantic_cost_weight),
        "student_density_cost_weight": float(args.student_density_cost_weight),
        "student_transition_cost_weight": 0.0
        if disable_length_transition_calibration(args)
        else float(args.student_transition_cost_weight),
        "student_length_cost_weight": 0.0
        if disable_length_transition_calibration(args)
        else float(args.student_length_cost_weight),
        "student_raw_length_cost_weight": 0.0
        if disable_length_transition_calibration(args)
        else float(args.student_raw_length_cost_weight),
        "teacher_component_names": ["density", "transition", "target_length", "raw_target_length"],
        "stream_train_features": True,
    }
    weighted_stats: Dict[str, float] = {}
    hist = np.zeros((k,), dtype=np.int64)

    for start in progress(ranges, desc="stream feature cache chunks", total=len(ranges)):
        end = min(n, start + chunk_size)
        candidates_chunk = np.asarray(train_candidates[start:end], dtype=np.float32)
        _sel, target_lengths, prior_scores, _prior_stats = compute_prior_scores(
            args,
            prior_train,
            train_query[start:end],
            candidates_chunk,
            prior,
        )
        base, _ = make_candidate_features(
            train_query[start:end],
            candidates_chunk,
            np.asarray(train_memory_idx[start:end], dtype=np.int64),
            target_lengths,
            proj,
            args.feature_mode,
            q_stats,
            prior_scores if args.prior_score_feature else None,
        )
        density_nll, transition_nll, length_cost, candidate_density = _table1_primitives(
            args,
            prior,
            candidates_chunk,
            target_lengths,
            keep_density=True,
        )
        if bool(getattr(args, "table1_cost_feature", False)):
            base = np.concatenate(
                [
                    base,
                    _table1_features_from_primitives(
                        args,
                        density_nll,
                        transition_nll,
                        length_cost,
                        candidate_density,
                        feature_overfull,
                    ),
                ],
                axis=2,
            ).astype(np.float32)
        label_feat = onehot_labels(train_records[start:end])
        sem_fields = np.zeros((*candidates_chunk.shape[:2], len(DEFAULT_MOBILITY_VOCAB.fields)), dtype=np.float32)
        features[start:end] = add_semantic_features(base, label_feat, sem_fields)
        if bool(args.teacher_use_real_train_length):
            teacher_target_lengths = haversine_km(train_query[start:end, :-1], train_query[start:end, 1:]).sum(axis=1).astype(np.float32)
            t_density_nll, t_transition_nll, t_length_cost, t_candidate_density = _table1_primitives(
                args,
                prior,
                candidates_chunk,
                teacher_target_lengths,
                keep_density=True,
            )
        else:
            teacher_target_lengths = target_lengths
            t_density_nll, t_transition_nll, t_length_cost, t_candidate_density = (
                density_nll,
                transition_nll,
                length_cost,
                candidate_density,
            )
        selected, scores, components, stats = _route_first_teacher_from_primitives(
            args,
            t_density_nll,
            t_transition_nll,
            t_length_cost,
            t_candidate_density,
            teacher_overfull,
        )
        teacher[start:end] = selected
        teacher_scores[start:end] = scores
        teacher_components[start:end] = components
        prior_scores_mm[start:end] = prior_scores
        target_lengths_mm[start:end] = target_lengths
        teacher_target_lengths_mm[start:end] = teacher_target_lengths
        hist += np.bincount(selected, minlength=k)
        for key, value in stats.items():
            weighted_stats[key] = weighted_stats.get(key, 0.0) + float(value) * float(end - start)
        features.flush()
        teacher.flush()
        teacher_scores.flush()
        teacher_components.flush()
        prior_scores_mm.flush()
        target_lengths_mm.flush()
        teacher_target_lengths_mm.flush()

    for key, value in weighted_stats.items():
        aggregate[key] = float(value / max(n, 1))
    aggregate["teacher_hist"] = hist.tolist()
    (cache_dir / "teacher_stats.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "complete": True,
                "n": n,
                "k": k,
                "feature_shape": list(feature_shape),
                "component_shape": list(component_shape),
                "chunk_size": chunk_size,
                "disable_length_transition_calibration": bool(disable_length_transition_calibration(args)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return features, teacher, teacher_scores, teacher_components, prior_scores_mm, target_lengths_mm, aggregate, {
        "semantic_candidate_appended": False,
        "stream_train_feature_cache": str(cache_dir),
        "stream_train_feature_cache_reused": False,
    }


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True).clip(1e-4)
    return ((values - mean) / std).astype(np.float32)


def candidate_target_length_error(candidates: np.ndarray, target_lengths: np.ndarray) -> np.ndarray:
    n, k_paths = candidates.shape[:2]
    cand_lengths = np.zeros((n, k_paths), dtype=np.float32)
    chunk = 1024
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        cand_lengths[start:end] = haversine_km(
            candidates[start:end, :, :-1],
            candidates[start:end, :, 1:],
        ).sum(axis=2).astype(np.float32)
    target = np.maximum(np.asarray(target_lengths, dtype=np.float32)[:, None], 0.1)
    return (np.abs(cand_lengths - target) / target).astype(np.float32)


def candidate_lengths_km(candidates: np.ndarray) -> np.ndarray:
    n, k_paths = candidates.shape[:2]
    lengths = np.zeros((n, k_paths), dtype=np.float32)
    chunk = 1024
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        lengths[start:end] = haversine_km(
            candidates[start:end, :, :-1],
            candidates[start:end, :, 1:],
        ).sum(axis=2).astype(np.float32)
    return lengths


def length_marginal_cost(
    prior_train: np.ndarray,
    candidates: np.ndarray,
    seed_selected: np.ndarray,
    *,
    length_bins: int,
) -> np.ndarray:
    """Penalize candidate choices that worsen train length-distribution balance."""
    bins = max(4, int(length_bins))
    train_lengths = haversine_km(prior_train[:, :-1], prior_train[:, 1:]).sum(axis=1).astype(np.float32)
    cand_lengths = candidate_lengths_km(candidates)
    max_len = max(float(np.percentile(train_lengths, 99.5)), float(np.percentile(cand_lengths, 99.5)), 1.0)
    edges = np.linspace(0.0, max_len, bins + 1, dtype=np.float32)
    train_bins = np.searchsorted(edges, np.clip(train_lengths, edges[0], edges[-1]), side="right") - 1
    train_bins = np.clip(train_bins, 0, bins - 1).astype(np.int64)
    cand_bins = np.searchsorted(edges, np.clip(cand_lengths, edges[0], edges[-1]), side="right") - 1
    cand_bins = np.clip(cand_bins, 0, bins - 1).astype(np.int64)
    target_hist = np.bincount(train_bins, minlength=bins).astype(np.float64)
    selected_bins = cand_bins[np.arange(cand_bins.shape[0]), np.asarray(seed_selected, dtype=np.int64)]
    selected_hist = np.bincount(selected_bins, minlength=bins).astype(np.float64)
    target_prob = target_hist / max(float(target_hist.sum()), 1.0)
    selected_prob = selected_hist / max(float(selected_hist.sum()), 1.0)
    overfull = (selected_prob - target_prob).astype(np.float32)
    return overfull[cand_bins].astype(np.float32)


def table1_cost_features(
    args,
    prior_train: np.ndarray,
    prior: Dict,
    candidates: np.ndarray,
    target_lengths: np.ndarray,
    sem_total: np.ndarray,
) -> np.ndarray:
    """Return per-candidate cost features used by the route-first teacher.

    The features are safe at validation/test time: they use only the train
    prior, OD-compatible route candidates, target-length estimates, and
    optional semantic costs computed from the instruction contract.
    """
    n, k_paths = candidates.shape[:2]
    grid_num = int(prior["grid_num"])
    out_of_bounds_policy = prior_out_of_bounds_policy(prior)
    num_cells = prior_num_cells(prior)
    bounds = prior["bounds"]
    density_nll = np.zeros((n, k_paths), dtype=np.float32)
    transition_nll = np.zeros((n, k_paths), dtype=np.float32)

    for i in progress(range(n), desc="table1 feature costs", total=n):
        group = _od_group(candidates[i, 0], bounds, int(prior["od_grid"]), out_of_bounds_policy)
        use_group = int(prior["group_count"].get(group, 0)) >= int(args.min_group_count)
        cell_nll = prior["group_cell_nll"].get(group, prior["global_cell_nll"]) if use_group else prior["global_cell_nll"]
        trans_nll = (
            prior["group_transition_nll"].get(group, prior["global_transition_nll"])
            if use_group
            else prior["global_transition_nll"]
        )
        for k in range(k_paths):
            traj = candidates[i, k]
            ids = _changed_ids(traj, bounds, grid_num, out_of_bounds_policy)
            transitions = _transition_ids(ids, num_cells)
            density_nll[i, k] = _mean_lookup(cell_nll, ids)
            transition_nll[i, k] = _mean_lookup(trans_nll, transitions)

    length_cost = candidate_target_length_error(candidates, target_lengths)
    semantic_cost = np.asarray(sem_total, dtype=np.float32)
    seed_cost = (
        effective_transition_weight(args) * normalize_rows(transition_nll)
        + effective_length_weight(args) * normalize_rows(length_cost)
        + float(args.teacher_semantic_cost_weight) * normalize_rows(semantic_cost)
    )
    seed_selected = seed_cost.argmin(axis=1)
    train_density = to_grid(prior_train, bounds, grid_num, out_of_bounds_policy).astype(np.float64)
    selected_density = np.zeros(num_cells, dtype=np.float64)
    for i in progress(range(n), desc="table1 selected density", total=n):
        selected_density += to_grid(
            candidates[i, seed_selected[i] : seed_selected[i] + 1],
            bounds,
            grid_num,
            out_of_bounds_policy,
        ).astype(np.float64)
    target_prob = train_density / max(float(train_density.sum()), 1.0)
    selected_prob = selected_density / max(float(selected_density.sum()), 1.0)
    overfull = (selected_prob - target_prob).astype(np.float32)
    density_marginal = np.zeros((n, k_paths), dtype=np.float32)
    for i in progress(range(n), desc="table1 density marginal", total=n):
        for k in range(k_paths):
            density_marginal[i, k] = float(
                np.dot(
                    to_grid(candidates[i, k : k + 1], bounds, grid_num, out_of_bounds_policy).astype(np.float32),
                    overfull,
                )
            )

    feature_blocks = [
        normalize_rows(density_nll),
        np.zeros_like(transition_nll, dtype=np.float32)
        if disable_length_transition_calibration(args)
        else normalize_rows(transition_nll),
        np.zeros_like(length_cost, dtype=np.float32)
        if disable_length_transition_calibration(args)
        else normalize_rows(length_cost),
        normalize_rows(density_marginal),
    ]
    if abs(effective_length_marginal_weight(args)) > 0.0:
        length_marginal = length_marginal_cost(
            prior_train,
            candidates,
            seed_selected,
            length_bins=int(args.length_bins),
        )
        feature_blocks.append(normalize_rows(length_marginal))
    feature_blocks.append(
        np.zeros_like(length_cost, dtype=np.float32)
        if disable_length_transition_calibration(args)
        else np.clip(length_cost, 0.0, 2.0)
    )

    return np.stack(feature_blocks, axis=2).astype(np.float32)


def realism_guard_mask(
    args,
    prior_scores: np.ndarray,
    candidates: np.ndarray,
    target_lengths: np.ndarray,
    realism_scores: np.ndarray | None = None,
) -> np.ndarray:
    scores = np.asarray(realism_scores if realism_scores is not None else prior_scores, dtype=np.float32)
    n, k = scores.shape
    order = np.argsort(scores, axis=1)
    rank = np.empty_like(order)
    rank[np.arange(n)[:, None], order] = np.arange(k, dtype=order.dtype)[None, :]
    valid = rank < max(1, min(int(args.realism_guard_topk), k))
    if float(args.realism_guard_score_margin) > 0.0:
        best = scores.min(axis=1, keepdims=True)
        valid |= scores <= best + float(args.realism_guard_score_margin)
    if float(args.realism_guard_length_ratio) > 0.0:
        cand_lengths = haversine_km(candidates[:, :, :-1], candidates[:, :, 1:]).sum(axis=2)
        target = np.maximum(np.asarray(target_lengths, dtype=np.float32)[:, None], 0.1)
        length_valid = np.abs(cand_lengths - target) / target <= float(args.realism_guard_length_ratio)
        valid &= length_valid
    valid[np.arange(n), scores.argmin(axis=1)] = True
    return valid


def guarded_teacher_scores(
    args,
    prior_scores: np.ndarray,
    sem_total: np.ndarray,
    candidates: np.ndarray,
    target_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict]:
    route_z = normalize_rows(prior_scores)
    sem_z = normalize_rows(sem_total)
    valid = realism_guard_mask(args, prior_scores, candidates, target_lengths)
    scores = float(args.semantic_weight) * sem_z + float(args.guarded_route_tie_weight) * route_z
    penalty = float(args.guarded_invalid_penalty)
    invalid_scores = scores.max(axis=1, keepdims=True) + penalty + np.maximum(route_z, 0.0)
    scores = np.where(valid, scores, invalid_scores).astype(np.float32)
    selected = scores.argmin(axis=1).astype(np.int64)
    stats = {
        "guarded_valid_count_mean": float(valid.sum(axis=1).mean()),
        "guarded_selected_route_rank_mean": float(np.argsort(np.argsort(prior_scores, axis=1), axis=1)[np.arange(len(selected)), selected].mean()),
        "guarded_selected_semantic_cost_mean": float(sem_total[np.arange(len(selected)), selected].mean()),
        "guarded_selected_prior_score_mean": float(prior_scores[np.arange(len(selected)), selected].mean()),
    }
    return selected, scores, stats


def route_first_soft_teacher_scores(
    args,
    prior_train: np.ndarray,
    prior: Dict,
    candidates: np.ndarray,
    target_lengths: np.ndarray,
    sem_total: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Dict, np.ndarray]:
    """Build a teacher whose costs expose the three Table-I realism axes.

    Density is approximated as a batch-level marginal cost: first obtain a
    route/length/semantic seed selection, then penalize candidates that put
    mass in cells already over-represented by that seed batch.
    """
    n, k_paths = candidates.shape[:2]
    grid_num = int(prior["grid_num"])
    out_of_bounds_policy = prior_out_of_bounds_policy(prior)
    num_cells = prior_num_cells(prior)
    bounds = prior["bounds"]

    density_nll = np.zeros((n, k_paths), dtype=np.float32)
    transition_nll = np.zeros((n, k_paths), dtype=np.float32)
    candidate_density = np.zeros((n, k_paths, num_cells), dtype=np.float32)
    for i in progress(range(n), desc="teacher table1 costs", total=n):
        group = _od_group(candidates[i, 0], bounds, int(prior["od_grid"]), out_of_bounds_policy)
        use_group = int(prior["group_count"].get(group, 0)) >= int(args.min_group_count)
        cell_nll = prior["group_cell_nll"].get(group, prior["global_cell_nll"]) if use_group else prior["global_cell_nll"]
        trans_nll = (
            prior["group_transition_nll"].get(group, prior["global_transition_nll"])
            if use_group
            else prior["global_transition_nll"]
        )
        for k in range(k_paths):
            traj = candidates[i, k]
            ids = _changed_ids(traj, bounds, grid_num, out_of_bounds_policy)
            transitions = _transition_ids(ids, num_cells)
            density_nll[i, k] = _mean_lookup(cell_nll, ids)
            transition_nll[i, k] = _mean_lookup(trans_nll, transitions)
            candidate_density[i, k] = to_grid(
                candidates[i, k : k + 1],
                bounds,
                grid_num,
                out_of_bounds_policy,
            ).astype(np.float32)

    length_cost = candidate_target_length_error(candidates, target_lengths)
    seed_cost = (
        effective_transition_weight(args) * normalize_rows(transition_nll)
        + effective_length_weight(args) * normalize_rows(length_cost)
        + float(args.teacher_semantic_cost_weight) * normalize_rows(sem_total)
    )
    seed_selected = seed_cost.argmin(axis=1)

    train_density = to_grid(prior_train, bounds, grid_num, out_of_bounds_policy).astype(np.float64)
    selected_density = candidate_density[np.arange(n), seed_selected].sum(axis=0).astype(np.float64)
    target_prob = train_density / max(float(train_density.sum()), 1.0)
    selected_prob = selected_density / max(float(selected_density.sum()), 1.0)
    overfull = (selected_prob - target_prob).astype(np.float32)
    density_marginal = np.einsum("nkd,d->nk", candidate_density, overfull).astype(np.float32)

    length_marginal = length_marginal_cost(
        prior_train,
        candidates,
        seed_selected,
        length_bins=int(args.length_bins),
    )
    transition_cost = normalize_rows(transition_nll)
    target_length_cost = normalize_rows(length_cost)
    raw_target_length_cost = np.clip(length_cost, 0.0, 2.0).astype(np.float32)
    density_cost = normalize_rows(density_nll) + float(args.teacher_density_marginal_weight) * normalize_rows(density_marginal)
    if disable_length_transition_calibration(args):
        transition_cost = np.zeros_like(transition_cost, dtype=np.float32)
        target_length_cost = np.zeros_like(target_length_cost, dtype=np.float32)
        raw_target_length_cost = np.zeros_like(raw_target_length_cost, dtype=np.float32)
    length_total_cost = (
        normalize_rows(length_cost)
        + effective_length_marginal_weight(args) * normalize_rows(length_marginal)
        + effective_raw_length_weight(args) * raw_target_length_cost
    )
    if disable_length_transition_calibration(args):
        length_total_cost = np.zeros_like(length_total_cost, dtype=np.float32)
    total = (
        float(args.teacher_density_weight) * density_cost
        + effective_transition_weight(args) * transition_cost
        + effective_length_weight(args) * length_total_cost
        + float(args.teacher_semantic_cost_weight) * normalize_rows(sem_total)
    ).astype(np.float32)
    component_scores = np.stack(
        [
            density_cost,
            transition_cost,
            target_length_cost,
            raw_target_length_cost,
        ],
        axis=2,
    ).astype(np.float32)
    selected = total.argmin(axis=1).astype(np.int64)
    row = np.arange(n)
    stats = {
        "teacher_score_mode": "route_first_soft",
        "teacher_selected_density_cost_mean": float(density_cost[row, selected].mean()),
        "teacher_selected_transition_cost_mean": float(transition_nll[row, selected].mean()),
        "teacher_selected_length_cost_mean": float(length_cost[row, selected].mean()),
        "teacher_selected_length_marginal_cost_mean": float(length_marginal[row, selected].mean()),
        "teacher_selected_semantic_cost_mean": float(sem_total[row, selected].mean()),
        "teacher_seed_changed_rate": float((selected != seed_selected).mean()),
        "teacher_density_weight": float(args.teacher_density_weight),
        "teacher_density_marginal_weight": float(args.teacher_density_marginal_weight),
        "teacher_transition_weight": float(args.teacher_transition_weight),
        "teacher_length_weight": float(args.teacher_length_weight),
        "teacher_raw_length_weight": float(args.teacher_raw_length_weight),
        "teacher_length_marginal_weight": float(args.teacher_length_marginal_weight),
        "disable_length_transition_calibration": bool(disable_length_transition_calibration(args)),
        "effective_teacher_transition_weight": float(effective_transition_weight(args)),
        "effective_teacher_length_weight": float(effective_length_weight(args)),
        "effective_teacher_raw_length_weight": float(effective_raw_length_weight(args)),
        "effective_teacher_length_marginal_weight": float(effective_length_marginal_weight(args)),
        "teacher_semantic_cost_weight": float(args.teacher_semantic_cost_weight),
        "student_density_cost_weight": float(args.student_density_cost_weight),
        "student_transition_cost_weight": 0.0
        if disable_length_transition_calibration(args)
        else float(args.student_transition_cost_weight),
        "student_length_cost_weight": 0.0
        if disable_length_transition_calibration(args)
        else float(args.student_length_cost_weight),
        "student_raw_length_cost_weight": 0.0
        if disable_length_transition_calibration(args)
        else float(args.student_raw_length_cost_weight),
        "teacher_component_names": ["density", "transition", "target_length", "raw_target_length"],
    }
    return selected, total, stats, component_scores


def select_candidates(
    args,
    logits: np.ndarray,
    prior_scores: np.ndarray,
    sem_total: np.ndarray,
    candidates: np.ndarray,
    target_lengths: np.ndarray,
    realism_logits: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Dict]:
    if str(args.selection_policy) == "weighted_sum":
        adjusted_logits = float(args.selection_logit_weight) * logits - float(args.semantic_weight) * sem_total
        if float(args.selection_target_length_weight) > 0.0:
            length_abs = candidate_target_length_error(candidates, target_lengths)
            adjusted_logits = adjusted_logits - float(args.selection_target_length_weight) * length_abs
        selected = adjusted_logits.argmax(axis=1).astype(np.int64)
        return selected, adjusted_logits.astype(np.float32), {"selection_policy": "weighted_sum"}

    realism_scores = None
    if str(args.realism_guard_source) == "realism_logits" and realism_logits is not None:
        realism_scores = -np.asarray(realism_logits, dtype=np.float32)
    valid = realism_guard_mask(args, prior_scores, candidates, target_lengths, realism_scores=realism_scores)
    if realism_logits is not None:
        adjusted_logits = (
            float(args.selection_logit_weight) * normalize_rows(logits)
            + float(args.guarded_route_tie_weight) * normalize_rows(realism_logits)
            - float(args.semantic_weight) * normalize_rows(sem_total)
        )
    else:
        adjusted_logits = (
            float(args.selection_logit_weight) * logits.copy().astype(np.float32)
            - float(args.semantic_weight) * normalize_rows(sem_total)
        )
    if float(args.selection_target_length_weight) > 0.0:
        length_abs = candidate_target_length_error(candidates, target_lengths)
        adjusted_logits = adjusted_logits - float(args.selection_target_length_weight) * length_abs
    row_min = adjusted_logits.min(axis=1, keepdims=True)
    adjusted_logits = np.where(valid, adjusted_logits, row_min - float(args.guarded_invalid_penalty))
    selected = adjusted_logits.argmax(axis=1).astype(np.int64)
    rank_scores = realism_scores if realism_scores is not None else prior_scores
    route_rank = np.argsort(np.argsort(rank_scores, axis=1), axis=1)
    stats = {
        "selection_policy": "guarded_two_stage",
        "realism_guard_source": str(args.realism_guard_source),
        "guarded_valid_count_mean": float(valid.sum(axis=1).mean()),
        "guarded_selected_route_rank_mean": float(route_rank[np.arange(len(selected)), selected].mean()),
        "guarded_selected_semantic_cost_mean": float(sem_total[np.arange(len(selected)), selected].mean()),
        "guarded_selected_prior_score_mean": float(prior_scores[np.arange(len(selected)), selected].mean()),
    }
    return selected, adjusted_logits, stats


def apply_semantic_fallback(
    args,
    selected: np.ndarray,
    sem_fields: np.ndarray,
    candidates: np.ndarray | None = None,
    target_lengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Dict]:
    policy = str(getattr(args, "semantic_fallback_policy", "none"))
    selected = np.asarray(selected, dtype=np.int64).copy()
    n, k, _num_fields = sem_fields.shape
    fallback_mask = np.zeros((n,), dtype=bool)
    fallback_idx = int(getattr(args, "semantic_fallback_index", -1))
    if fallback_idx < 0:
        fallback_idx = k - 1
    stats: Dict[str, object] = {
        "semantic_fallback_policy": policy,
        "semantic_fallback_index": int(fallback_idx),
        "semantic_fallback_count": 0,
    }
    if policy == "none":
        return selected, fallback_mask, stats
    if fallback_idx < 0 or fallback_idx >= k:
        stats["semantic_fallback_invalid"] = True
        return selected, fallback_mask, stats

    fields = list(DEFAULT_MOBILITY_VOCAB.fields)
    field_to_idx = {field: idx for idx, field in enumerate(fields)}
    row = np.arange(n)
    selected_fields = sem_fields[row, selected]
    fallback_fields = sem_fields[:, fallback_idx]

    if policy == "direction":
        idx = field_to_idx["direction"]
        fallback_mask = fallback_fields[:, idx] < selected_fields[:, idx]
    elif policy == "curvature":
        idx = field_to_idx["curvature"]
        fallback_mask = fallback_fields[:, idx] < selected_fields[:, idx]
    elif policy == "dir_or_curv":
        d_idx = field_to_idx["direction"]
        c_idx = field_to_idx["curvature"]
        fallback_mask = (
            (fallback_fields[:, d_idx] < selected_fields[:, d_idx])
            | (fallback_fields[:, c_idx] < selected_fields[:, c_idx])
        )
    elif policy == "all_semantic":
        d_idx = field_to_idx["direction"]
        c_idx = field_to_idx["curvature"]
        selected_hits = selected_fields <= 1e-8
        fallback_hits = fallback_fields <= 1e-8
        fallback_mask = (
            (fallback_hits.sum(axis=1) > selected_hits.sum(axis=1))
            & (fallback_hits[:, d_idx] | fallback_hits[:, c_idx])
        )
    else:
        raise ValueError(f"Unknown semantic fallback policy: {policy}")

    max_length_error = float(getattr(args, "semantic_fallback_max_target_length_error", -1.0))
    if max_length_error > 0.0 and candidates is not None and target_lengths is not None:
        length_error = candidate_target_length_error(candidates, target_lengths)
        fallback_length_error = length_error[:, fallback_idx]
        length_ok = fallback_length_error <= max_length_error
        stats["semantic_fallback_max_target_length_error"] = float(max_length_error)
        stats["semantic_fallback_length_guard_rejected"] = int((fallback_mask & ~length_ok).sum())
        stats["semantic_fallback_candidate_length_error_mean"] = float(fallback_length_error.mean())
        fallback_mask = fallback_mask & length_ok

    selected[fallback_mask] = fallback_idx
    if bool(fallback_mask.any()):
        after_fields = sem_fields[row, selected]
        stats.update(
            {
                "semantic_fallback_count": int(fallback_mask.sum()),
                "semantic_fallback_rate": float(fallback_mask.mean()),
                "semantic_fallback_selected_field_cost_before": {
                    field: float(selected_fields[:, idx].mean()) for idx, field in enumerate(fields)
                },
                "semantic_fallback_selected_field_cost_after": {
                    field: float(after_fields[:, idx].mean()) for idx, field in enumerate(fields)
                },
            }
        )
    return selected, fallback_mask.astype(bool), stats


def load_reranker_checkpoint(path: Path, device: torch.device) -> tuple[RouteMemoryReranker, np.ndarray]:
    # Deserialize on CPU first. Loading directly onto CUDA keeps the serialized
    # tensors and the instantiated model resident together, causing a transient
    # memory spike during large evaluation jobs.
    payload = safe_torch_load(path, torch.device("cpu"))
    q_stats = np.asarray(payload["query_feature_stats"], dtype=np.float32).copy()
    first_weight = payload["model_state"]["net.0.weight"]
    hidden_dim = int(first_weight.shape[0])
    input_dim = int(first_weight.shape[1])
    model = RouteMemoryReranker(input_dim, hidden_dim)
    model.load_state_dict(payload["model_state"])
    del payload
    model = model.to(device)
    model.eval()
    return model, q_stats


def load_prior_cache(path: Path, sample_ids: np.ndarray, candidate_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    required = ["sample_id", "candidate_shape", "target_lengths", "prior_scores"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{path} missing prior-cache keys: {missing}")
    cached_ids = np.asarray(payload["sample_id"], dtype=np.int64)
    if cached_ids.shape != sample_ids.shape or not np.array_equal(cached_ids, sample_ids):
        raise ValueError(
            f"{path} sample_id does not match current split: "
            f"cache_shape={cached_ids.shape} current_shape={sample_ids.shape}"
        )
    cached_shape = tuple(np.asarray(payload["candidate_shape"], dtype=np.int64).tolist())
    current_shape = tuple(int(v) for v in candidate_shape)
    if cached_shape != current_shape:
        raise ValueError(f"{path} candidate shape mismatch: cache={cached_shape} current={current_shape}")
    return (
        np.asarray(payload["target_lengths"], dtype=np.float32),
        np.asarray(payload["prior_scores"], dtype=np.float32),
    )


def save_prior_cache(
    path: Path,
    *,
    sample_ids: np.ndarray,
    candidate_shape: np.ndarray,
    target_lengths: np.ndarray,
    prior_scores: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log_stage(f"prior-cache: save {path}")
    np.savez(
        path,
        sample_id=sample_ids.astype(np.int64),
        candidate_shape=np.asarray(candidate_shape, dtype=np.int64),
        target_lengths=target_lengths.astype(np.float32),
        prior_scores=prior_scores.astype(np.float32),
    )
    log_stage(f"prior-cache: saved {path}")


def load_feature_cache(path: Path, sample_ids: np.ndarray) -> Dict[str, np.ndarray]:
    payload = np.load(path)
    required = [
        "sample_id",
        "candidates",
        "memory_idx",
        "prior_scores",
        "target_lengths",
        "sem_total",
        "sem_fields",
        "features",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{path} missing feature-cache keys: {missing}")
    cached_ids = np.asarray(payload["sample_id"], dtype=np.int64)
    if cached_ids.shape != sample_ids.shape or not np.array_equal(cached_ids, sample_ids):
        raise ValueError(
            f"{path} sample_id does not match current split: "
            f"cache_shape={cached_ids.shape} current_shape={sample_ids.shape}"
        )
    return {key: np.asarray(payload[key]) for key in required}


def save_feature_cache(
    path: Path,
    *,
    sample_ids: np.ndarray,
    candidates: np.ndarray,
    memory_idx: np.ndarray,
    prior_scores: np.ndarray,
    target_lengths: np.ndarray,
    sem_total: np.ndarray,
    sem_fields: np.ndarray,
    features: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log_stage(f"feature-cache: save {path}")
    np.savez(
        path,
        sample_id=sample_ids.astype(np.int64),
        candidates=candidates.astype(np.float32),
        memory_idx=memory_idx.astype(np.int64),
        prior_scores=prior_scores.astype(np.float32),
        target_lengths=target_lengths.astype(np.float32),
        sem_total=sem_total.astype(np.float32),
        sem_fields=sem_fields.astype(np.float32),
        features=features.astype(np.float32),
    )
    log_stage(f"feature-cache: saved {path}")


def evaluate_split(
    args,
    name,
    model,
    memory,
    prior_train,
    real,
    real_path,
    input_path,
    prior,
    proj,
    q_stats,
    calibrator,
    device,
    realism_model=None,
    realism_q_stats=None,
    semantic_candidate_npz: Path | None = None,
):
    log_stage(f"{name}: start evaluate_split n={len(real)} topm={int(args.candidate_topm)} device={device}")
    log_stage(f"{name}: build records")
    sample_ids, records = build_records(input_path, real_path, len(real), start=int(args.eval_start))
    cache_path = args.route_memory_feature_cache
    use_cache = bool(args.reuse_route_memory_feature_cache and cache_path is not None and cache_path.exists())
    if use_cache:
        log_stage(f"{name}: reuse feature cache {cache_path}")
        cached = load_feature_cache(cache_path, sample_ids)
        candidates = cached["candidates"].astype(np.float32)
        memory_idx = cached["memory_idx"].astype(np.int64)
        prior_scores = cached["prior_scores"].astype(np.float32)
        target_lengths = cached["target_lengths"].astype(np.float32)
        sem_total = cached["sem_total"].astype(np.float32)
        sem_fields = cached["sem_fields"].astype(np.float32)
        features = cached["features"].astype(np.float32)
        append_stats = {
            "semantic_candidate_appended": bool(semantic_candidate_npz is not None),
            "semantic_candidate_npz": str(semantic_candidate_npz) if semantic_candidate_npz is not None else None,
            "feature_cache_reused": True,
            "feature_cache": str(cache_path),
            "candidate_topm_after_append": int(candidates.shape[1]),
        }
    else:
        log_stage(f"{name}: build memory candidates")
        candidates, memory_idx = build_memory_candidates(
            memory,
            real,
            int(args.candidate_topm),
            args.feature_mode,
            args.retrieval_method,
            cache_path=args.route_memory_candidate_cache,
            reuse_cache=bool(args.reuse_route_memory_candidate_cache),
            chunk_size=int(args.candidate_cache_chunk_size),
        )
        log_stage(f"{name}: append semantic candidate")
        candidates, memory_idx, append_stats = append_semantic_candidate(
            candidates,
            memory_idx,
            sample_ids,
            semantic_candidate_npz,
        )
        log_stage(f"{name}: make candidate features and semantic scores")
        args._current_sample_ids = sample_ids
        features, prior_scores, sem_total, target_lengths, sem_fields = make_features_for_records(
            args, real, candidates, memory_idx, records, prior_train, prior, proj, q_stats, calibrator
        )
        if cache_path is not None:
            save_feature_cache(
                cache_path,
                sample_ids=sample_ids,
                candidates=candidates,
                memory_idx=memory_idx,
                prior_scores=prior_scores,
                target_lengths=target_lengths,
                sem_total=sem_total,
                sem_fields=sem_fields,
                features=features,
            )
    realism_logits = None
    if realism_model is not None and realism_q_stats is not None:
        log_stage(f"{name}: make realism features")
        realism_features, _ = make_candidate_features(
            real,
            candidates,
            memory_idx,
            target_lengths,
            proj,
            args.feature_mode,
            realism_q_stats,
            prior_scores if args.prior_score_feature else None,
        )
        log_stage(f"{name}: predict realism reranker")
        _realism_selected, realism_logits = predict(realism_model, realism_features, device, int(args.batch_size))
    log_stage(f"{name}: predict semantic reranker")
    _raw_selected, logits = predict(model, features, device, int(args.batch_size))
    log_stage(f"{name}: select candidates")
    selected, adjusted_logits, selection_stats = select_candidates(
        args,
        logits,
        prior_scores,
        sem_total,
        candidates,
        target_lengths,
        realism_logits=realism_logits,
    )
    if str(args.batch_refine_score_source) == "realism" and realism_logits is not None:
        refine_scores = -realism_logits
    elif str(args.batch_refine_score_source) == "prior":
        refine_scores = prior_scores
    else:
        refine_scores = -adjusted_logits
    refine_target_length_weight = float(getattr(args, "refine_target_length_weight", 0.0))
    if refine_target_length_weight > 0.0:
        refine_length_error = candidate_target_length_error(candidates, target_lengths)
        refine_scores = np.asarray(refine_scores, dtype=np.float32) + refine_target_length_weight * refine_length_error
    selected_before_fallback = selected.copy()
    semantic_fallback_mask = np.zeros(len(selected), dtype=bool)
    fallback_stats: Dict = {}
    if args.eval_batch_refine:
        if str(args.semantic_fallback_stage) == "before_refine":
            log_stage(f"{name}: semantic fallback before refine")
            selected_before_fallback = selected.copy()
            selected, semantic_fallback_mask, fallback_stats = apply_semantic_fallback(
                args, selected, sem_fields, candidates, target_lengths
            )
        log_stage(f"{name}: batch density/length refinement cycles={int(args.batch_refine_cycles)}")
        selected, refined_stats = batch_refine_selection(
            prior_train,
            candidates,
            selected,
            refine_scores,
            grid_num=int(args.grid_num),
            length_bins=int(args.length_bins),
            cycles=int(args.batch_refine_cycles),
            density_penalty=float(args.batch_density_refine_penalty),
            length_penalty=float(args.batch_length_refine_penalty),
            topk=int(args.batch_refine_topk),
            hot_cells=int(args.batch_density_hot_cells),
            target_lengths=target_lengths if str(args.batch_refine_length_target) == "predicted" else None,
            frozen_mask=(
                semantic_fallback_mask
                if bool(args.freeze_semantic_fallback_refine) and str(args.semantic_fallback_stage) == "before_refine"
                else None
            ),
            bounds=prior["bounds"],
            out_of_bounds_policy=prior_out_of_bounds_policy(prior),
        )
        if str(args.semantic_fallback_stage) == "after_refine":
            log_stage(f"{name}: semantic fallback after refine")
            selected_before_fallback = selected.copy()
            selected, semantic_fallback_mask, fallback_stats = apply_semantic_fallback(
                args, selected, sem_fields, candidates, target_lengths
            )
        post_cycles = int(args.post_fallback_refine_cycles)
        if post_cycles > 0:
            log_stage(f"{name}: post-fallback refinement cycles={post_cycles}")
            post_density = (
                float(args.post_fallback_density_refine_penalty)
                if float(args.post_fallback_density_refine_penalty) >= 0.0
                else float(args.batch_density_refine_penalty)
            )
            post_length = (
                float(args.post_fallback_length_refine_penalty)
                if float(args.post_fallback_length_refine_penalty) >= 0.0
                else float(args.batch_length_refine_penalty)
            )
            post_topk = int(args.post_fallback_refine_topk) if int(args.post_fallback_refine_topk) > 0 else int(args.batch_refine_topk)
            selected, post_refined_stats = batch_refine_selection(
                prior_train,
                candidates,
                selected,
                refine_scores,
                grid_num=int(args.grid_num),
                length_bins=int(args.length_bins),
                cycles=post_cycles,
                density_penalty=post_density,
                length_penalty=post_length,
                topk=post_topk,
                hot_cells=int(args.batch_density_hot_cells),
                target_lengths=target_lengths if str(args.batch_refine_length_target) == "predicted" else None,
                frozen_mask=semantic_fallback_mask if bool(args.freeze_semantic_fallback_refine) else None,
                bounds=prior["bounds"],
                out_of_bounds_policy=prior_out_of_bounds_policy(prior),
            )
            refined_stats = {**refined_stats, "post_fallback_refine": post_refined_stats}
    else:
        log_stage(f"{name}: semantic fallback without batch refinement")
        selected_before_fallback = selected.copy()
        selected, semantic_fallback_mask, fallback_stats = apply_semantic_fallback(
            args, selected, sem_fields, candidates, target_lengths
        )
        refined_stats = {}
    selection_stats.update(fallback_stats)
    log_stage(f"{name}: build generated trajectories")
    generated = candidates[np.arange(len(real)), selected]
    log_stage(f"{name}: evaluate Table I distribution metrics")
    table1 = evaluate_distribution(
        real,
        generated,
        grid_num=int(args.grid_num),
        length_bins=int(args.length_bins),
        bounds=prior["bounds"],
        bounds_source=prior["bounds_source"],
        out_of_bounds_policy=prior_out_of_bounds_policy(prior),
    )
    table1.update(
        {
            "selector": "semantic_route_memory_reranker",
            "num_samples": int(len(real)),
            "candidate_topm": int(args.candidate_topm),
            "selection_policy": str(args.selection_policy),
            "semantic_weight": float(args.semantic_weight),
            "selection_target_length_weight": float(args.selection_target_length_weight),
            "refine_target_length_weight": float(args.refine_target_length_weight),
            "eval_batch_refine": bool(args.eval_batch_refine),
            "semantic_fallback_stage": str(args.semantic_fallback_stage),
            "freeze_semantic_fallback_refine": bool(args.freeze_semantic_fallback_refine),
            "gt_avg_length_km": float(haversine_km(real[:, :-1], real[:, 1:]).sum(axis=1).mean()),
            "gen_avg_length_km": float(haversine_km(generated[:, :-1], generated[:, 1:]).sum(axis=1).mean()),
            "selection_stats": selection_stats,
            "semantic_candidate_stats": append_stats,
            "refined_stats": refined_stats,
            "controltraj_thresholds": {
                "Density Error": float(args.controltraj_density_threshold),
                "Route Transition JSD": float(args.controltraj_route_threshold),
                "Length Error": float(args.controltraj_length_threshold),
            },
            "beats_controltraj_all": bool(
                table1["Density Error"] < float(args.controltraj_density_threshold)
                and table1["Route Transition JSD"] < float(args.controltraj_route_threshold)
                and table1["Length Error"] < float(args.controltraj_length_threshold)
            ),
        }
    )
    if bool(args.skip_table3):
        log_stage(f"{name}: skip Table III instruction following")
        table3 = None
    else:
        log_stage(f"{name}: evaluate Table III instruction following")
        table3 = table3_from_generated(generated, records)
    log_stage(f"{name}: save outputs")
    out = args.output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    save_npz = np.savez if bool(args.save_uncompressed_trajectories) else np.savez_compressed
    save_npz(
        out / "semantic_reranker_trajectories.npz",
        sample_id=sample_ids.astype(np.int64),
        real_trajectories=real.astype(np.float32),
        generated_trajectories=generated.astype(np.float32),
        selected_local=selected.astype(np.int64),
        logits=logits.astype(np.float32),
        adjusted_logits=adjusted_logits.astype(np.float32),
        realism_logits=realism_logits.astype(np.float32) if realism_logits is not None else np.zeros_like(logits, dtype=np.float32),
        selected_before_semantic_fallback=selected_before_fallback.astype(np.int64),
        semantic_fallback_mask=semantic_fallback_mask.astype(bool),
    )
    (out / "table1_metrics.json").write_text(json.dumps(table1, ensure_ascii=False, indent=2), encoding="utf-8")
    if table3 is not None:
        (out / "table3_metrics.json").write_text(json.dumps(table3, ensure_ascii=False, indent=2), encoding="utf-8")
    log_stage(f"{name}: finished evaluate_split")
    return table1, table3, {
        "sample_ids": sample_ids,
        "records": records,
        "real": real,
        "candidates": candidates,
        "memory_idx": memory_idx,
        "logits": logits,
        "realism_logits": realism_logits,
        "prior_scores": prior_scores,
        "target_lengths": target_lengths,
        "realism_model": realism_model,
        "realism_q_stats": realism_q_stats,
        "semantic_candidate_stats": append_stats,
    }


def evaluate_table4(args, name, model, split_state, cf_input, prior_train, prior, proj, q_stats, calibrator, device) -> Dict:
    sample_ids = split_state["sample_ids"]
    real = split_state["real"]
    candidates = split_state["candidates"]
    memory_idx = split_state["memory_idx"]
    realism_model = split_state.get("realism_model")
    realism_q_stats = split_state.get("realism_q_stats")
    id_to_row = {int(sid): i for i, sid in enumerate(sample_ids.tolist())}
    requested_fields = {
        item.strip()
        for item in str(getattr(args, "table4_fields", "")).split(",")
        if item.strip()
    }
    group_limit = int(args.max_pairs)
    if requested_fields and group_limit > 0:
        group_limit *= len(DEFAULT_MOBILITY_VOCAB.fields)
    pairs, pair_stats = group_pairs_streaming(cf_input, set(id_to_row), group_limit, int(args.counterfactual_scan_limit))
    if requested_fields:
        pairs = [
            pair
            for pair in pairs
            if str(pair[1].get("counterfactual", {}).get("field", "")) in requested_fields
        ]
        if int(args.max_pairs) > 0:
            pairs = pairs[: int(args.max_pairs)]
        pair_stats = {
            **pair_stats,
            "field_filter": sorted(requested_fields),
            "matched_pairs_after_field_filter": int(len(pairs)),
        }
    target_effects: List[float] = []
    successes: List[bool] = []
    raw_collaterals: List[float] = []
    independent_collaterals: List[float] = []
    by_field: Dict[str, Dict[str, List[float]]] = {}
    for original, counterfactual in pairs:
        source_id = int(counterfactual.get("counterfactual", {}).get("source_sample_id", original.get("sample_id", -1)))
        row = id_to_row[source_id]
        real_row = real[row : row + 1]
        cand_row = candidates[row : row + 1]
        mem_row = memory_idx[row : row + 1]
        base_features, prior_scores, sem_total, target_lengths, base_sem_fields = make_features_for_records(
            args, real_row, cand_row, mem_row, [original], prior_train, prior, proj, q_stats, calibrator
        )
        base_realism_logits = None
        if realism_model is not None and realism_q_stats is not None:
            base_realism_features, _ = make_candidate_features(
                real_row,
                cand_row,
                mem_row,
                target_lengths,
                proj,
                args.feature_mode,
                realism_q_stats,
                prior_scores if args.prior_score_feature else None,
            )
            _base_realism_selected, base_realism_logits = predict(realism_model, base_realism_features, device, int(args.batch_size))
        _base_selected, base_logits = predict(model, base_features, device, int(args.batch_size))
        base_selected, _base_adjusted, _base_stats = select_candidates(
            args,
            base_logits,
            prior_scores,
            sem_total,
            cand_row,
            target_lengths,
            realism_logits=base_realism_logits,
        )
        base_selected, _base_fallback_mask, _base_fallback_stats = apply_semantic_fallback(
            args, base_selected, base_sem_fields, cand_row, target_lengths
        )
        base_idx = int(base_selected[0])
        base_traj = cand_row[0, base_idx]
        base_geom = geometry_row(base_traj, original, cand_row.shape[2])
        cf_meta = counterfactual.get("counterfactual", {})
        field = str(cf_meta.get("field", ""))
        source_token = str(cf_meta.get("from", original.get("labels", {}).get(field, "unknown")))
        target_token = str(cf_meta.get("to", counterfactual.get("labels", {}).get(field, "unknown")))
        if str(args.table4_selection_policy) == "target_collateral_guard":
            target_values: List[float] = []
            raw_values: List[float] = []
            independent_values: List[float] = []
            for candidate_idx in range(cand_row.shape[1]):
                cand_geom = geometry_row(cand_row[0, candidate_idx], counterfactual, cand_row.shape[2])
                _metric, _base_to_target, _observed, _signed, candidate_signed_z = calibrator.transition_metric_delta(
                    field,
                    source_token,
                    target_token,
                    base_geom,
                    cand_geom,
                )
                drift_by_field = {}
                for other in DEFAULT_MOBILITY_VOCAB.fields:
                    if other == field:
                        continue
                    token = str(original.get("labels", {}).get(other, "unknown"))
                    if token in calibrator.stats[other]:
                        drift_by_field[other] = calibrator.normalized_metric_delta(other, token, base_geom, cand_geom)
                raw_candidate = float(np.mean(list(drift_by_field.values()))) if drift_by_field else 0.0
                independent_candidate = [
                    drift_by_field[x]
                    for x in independent_collateral_field_names(field, fields=drift_by_field.keys())
                    if x in drift_by_field
                ]
                indep_candidate = float(np.mean(independent_candidate)) if independent_candidate else 0.0
                target_values.append(float(candidate_signed_z))
                raw_values.append(raw_candidate)
                independent_values.append(indep_candidate)
            target_arr = np.asarray(target_values, dtype=np.float32)
            raw_arr = np.asarray(raw_values, dtype=np.float32)
            indep_arr = np.asarray(independent_values, dtype=np.float32)
            feasible = (target_arr > float(args.table4_target_margin)) & (indep_arr <= float(args.table4_collateral_margin))
            if str(args.table4_priority) == "collateral_first":
                successful = target_arr > float(args.table4_target_margin)
                collateral_cost = indep_arr + float(args.table4_raw_collateral_weight) * raw_arr
                if bool(feasible.any()):
                    masked = np.where(feasible, collateral_cost, np.inf)
                    cf_idx = int(np.argmin(masked))
                elif bool(successful.any()):
                    masked = np.where(successful, collateral_cost, np.inf)
                    cf_idx = int(np.argmin(masked))
                else:
                    utility = target_arr - float(args.table4_collateral_weight) * indep_arr - float(args.table4_raw_collateral_weight) * raw_arr
                    cf_idx = int(np.argmax(utility))
                if (
                    float(args.table4_skip_collateral_above) >= 0.0
                    and float(collateral_cost[cf_idx]) > float(args.table4_skip_collateral_above)
                ):
                    cf_idx = int(base_idx)
            else:
                utility = target_arr - float(args.table4_collateral_weight) * indep_arr - float(args.table4_raw_collateral_weight) * raw_arr
                if bool(feasible.any()):
                    masked = np.where(feasible, utility, -np.inf)
                    cf_idx = int(np.argmax(masked))
                else:
                    cf_idx = int(np.argmax(utility))
            cf_geom = geometry_row(cand_row[0, cf_idx], counterfactual, cand_row.shape[2])
        else:
            cf_features, cf_prior_scores, cf_sem_total, cf_target_lengths, cf_sem_fields = make_features_for_records(
                args, real_row, cand_row, mem_row, [counterfactual], prior_train, prior, proj, q_stats, calibrator
            )
            cf_realism_logits = None
            if realism_model is not None and realism_q_stats is not None:
                cf_realism_features, _ = make_candidate_features(
                    real_row,
                    cand_row,
                    mem_row,
                    cf_target_lengths,
                    proj,
                    args.feature_mode,
                    realism_q_stats,
                    cf_prior_scores if args.prior_score_feature else None,
                )
                _cf_realism_selected, cf_realism_logits = predict(realism_model, cf_realism_features, device, int(args.batch_size))
            _cf_selected, cf_logits = predict(model, cf_features, device, int(args.batch_size))
            cf_selected, _cf_adjusted, _cf_stats = select_candidates(
                args,
                cf_logits,
                cf_prior_scores,
                cf_sem_total,
                cand_row,
                cf_target_lengths,
                realism_logits=cf_realism_logits,
            )
            cf_selected, _cf_fallback_mask, _cf_fallback_stats = apply_semantic_fallback(
                args, cf_selected, cf_sem_fields, cand_row, cf_target_lengths
            )
            cf_idx = int(cf_selected[0])
            cf_geom = geometry_row(cand_row[0, cf_idx], counterfactual, cand_row.shape[2])
        _metric, _base_to_target, _observed, _signed, signed_z = calibrator.transition_metric_delta(
            field, source_token, target_token, base_geom, cf_geom
        )
        target_effects.append(float(signed_z))
        successes.append(float(signed_z) > 0.0)
        drift_by_field = {}
        for other in DEFAULT_MOBILITY_VOCAB.fields:
            if other == field:
                continue
            token = str(original.get("labels", {}).get(other, "unknown"))
            if token in calibrator.stats[other]:
                drift_by_field[other] = calibrator.normalized_metric_delta(other, token, base_geom, cf_geom)
        raw = float(np.mean(list(drift_by_field.values()))) if drift_by_field else 0.0
        independent = [
            drift_by_field[x]
            for x in independent_collateral_field_names(field, fields=drift_by_field.keys())
            if x in drift_by_field
        ]
        indep = float(np.mean(independent)) if independent else 0.0
        raw_collaterals.append(raw)
        independent_collaterals.append(indep)
        by_field.setdefault(field, {"target": [], "success": [], "raw": [], "independent": []})
        by_field[field]["target"].append(float(signed_z))
        by_field[field]["success"].append(1.0 if float(signed_z) > 0.0 else 0.0)
        by_field[field]["raw"].append(raw)
        by_field[field]["independent"].append(indep)
    table4 = {
        "num_pairs": int(len(pairs)),
        "pair_scan_stats": pair_stats,
        "table4_counterfactual_controllability": {
            "counterfactual_target_effect": float(np.mean(target_effects)) if target_effects else float("nan"),
            "counterfactual_success_rate": float(np.mean(successes)) if successes else float("nan"),
            "raw_collateral_drift": float(np.mean(raw_collaterals)) if raw_collaterals else float("nan"),
            "independent_collateral_drift": float(np.mean(independent_collaterals)) if independent_collaterals else float("nan"),
        },
        "counterfactual_target_effect_z": summarize(target_effects),
        "raw_collateral_drift_z": summarize(raw_collaterals),
        "independent_collateral_drift_z": summarize(independent_collaterals),
        "by_field": {
            field: {
                "counterfactual_target_effect_z": summarize(vals["target"]),
                "counterfactual_success_rate": float(np.mean(vals["success"])) if vals["success"] else float("nan"),
                "raw_collateral_drift_z": summarize(vals["raw"]),
                "independent_collateral_drift_z": summarize(vals["independent"]),
            }
            for field, vals in sorted(by_field.items())
        },
    }
    out = args.output_dir / name
    (out / "table4_metrics.json").write_text(json.dumps(table4, ensure_ascii=False, indent=2), encoding="utf-8")
    return table4


def main() -> None:
    args = parse_args()
    fixed_bounds = load_fixed_bounds(args.bounds_json) if args.bounds_json else None
    out_of_bounds_policy = args.out_of_bounds_policy or ("overflow" if fixed_bounds is not None else "discard")
    bounds_source = args.bounds_source or (str(args.bounds_json.resolve()) if args.bounds_json else None)
    if disable_length_transition_calibration(args):
        args.student_transition_cost_weight = 0.0
        args.student_length_cost_weight = 0.0
        args.student_raw_length_cost_weight = 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    log_stage(f"start device={device} eval_split={args.eval_split} eval_only={bool(args.eval_only)}")
    if disable_length_transition_calibration(args):
        log_stage("ablation: disable length/transition calibration")
    log_stage(f"load memory real npz samples={int(args.memory_samples)}")
    memory = load_real(args.memory_real_npz, int(args.memory_samples))
    log_stage(f"load train query real npz samples={int(args.train_query_samples)}")
    train_query = load_real(args.train_query_real_npz, int(args.train_query_samples))
    log_stage("load eval split real npz")
    val = load_real_window(args.val_real_npz, int(args.eval_samples), int(args.eval_start)) if args.eval_split in {"both", "val"} else None
    test = load_real_window(args.test_real_npz, int(args.eval_samples), int(args.eval_start)) if args.eval_split in {"both", "test"} else None
    log_stage("build train records")
    train_ids, train_records = build_records(args.train_input, args.train_query_real_npz, len(train_query))
    log_stage("make projection")
    proj = make_projection(memory)
    prior_train = memory[: int(args.prior_samples)]
    log_stage(f"build coarse prior samples={len(prior_train)}")
    prior = build_coarse_prior(
        prior_train,
        grid_num=int(args.grid_num),
        od_grid=int(args.od_grid),
        length_bins=int(args.length_bins),
        bounds=fixed_bounds,
        bounds_source=bounds_source,
        out_of_bounds_policy=out_of_bounds_policy,
    )
    log_stage(
        "grid protocol: "
        f"grid={prior['grid_num']}x{prior['grid_num']} "
        f"cells={prior['num_cells']} policy={prior['out_of_bounds_policy']} "
        f"bounds_source={prior['bounds_source']}"
    )
    needs_table4 = not bool(args.skip_table4) and (
        (args.eval_split == "both")
        or (args.eval_split == "val" and val is not None)
        or (args.eval_split == "test" and test is not None)
    )
    if bool(args.disable_semantic_costs) and not needs_table4:
        log_stage("skip semantic calibrator (--disable-semantic-costs)")
        calibrator = None
    else:
        log_stage("build semantic calibrator")
        calibrator = build_calibrator(train_records, train_query.shape[1])
    realism_model = None
    realism_q_stats = None
    if args.realism_checkpoint is not None:
        log_stage(f"load realism checkpoint {args.realism_checkpoint}")
        realism_model, realism_q_stats = load_reranker_checkpoint(args.realism_checkpoint, device)
    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint")
        log_stage(f"load semantic reranker checkpoint {args.checkpoint}")
        model, q_stats = load_reranker_checkpoint(args.checkpoint, device)
    else:
        log_stage("build train memory candidates")
        train_candidates, train_memory_idx = build_memory_candidates(
            memory,
            train_query,
            int(args.candidate_topm),
            args.feature_mode,
            args.retrieval_method,
            cache_path=args.train_route_memory_candidate_cache,
            reuse_cache=bool(args.reuse_train_route_memory_candidate_cache),
            chunk_size=int(args.candidate_cache_chunk_size),
        )
        log_stage("append train semantic candidate")
        train_candidates, train_memory_idx, train_append_stats = append_semantic_candidate(
            train_candidates,
            train_memory_idx,
            train_ids,
            args.train_semantic_candidate_npz,
        )
        log_stage("make train query feature stats")
        q_stats = query_feature_stats_only(train_query, proj, args.feature_mode)
        if bool(args.stream_train_features):
            log_stage("stream full train-query features/teacher cache")
            (
                train_features,
                teacher,
                teacher_scores,
                teacher_component_scores,
                prior_scores,
                target_lengths,
                teacher_stats,
                stream_append_stats,
            ) = build_stream_train_feature_cache(
                args,
                train_query,
                train_candidates,
                train_memory_idx,
                train_records,
                prior_train,
                prior,
                proj,
                q_stats,
            )
            train_append_stats = train_append_stats | stream_append_stats
            teacher_target_lengths = np.memmap(
                Path(str(args.train_feature_cache) + ".parts") / "teacher_target_lengths.float32.dat",
                mode="r",
                dtype=np.float32,
                shape=(len(train_query),),
            )
            sem_total = np.zeros_like(teacher_scores, dtype=np.float32)
        else:
            log_stage("make train semantic candidate features")
            old_prior_cache = args.route_memory_prior_cache
            old_reuse_prior_cache = args.reuse_route_memory_prior_cache
            if args.train_route_memory_prior_cache is not None:
                args.route_memory_prior_cache = args.train_route_memory_prior_cache
                args.reuse_route_memory_prior_cache = bool(args.reuse_train_route_memory_prior_cache)
            try:
                train_features, prior_scores, sem_total, target_lengths, _train_sem_fields = make_features_for_records(
                    args, train_query, train_candidates, train_memory_idx, train_records, prior_train, prior, proj, q_stats, calibrator
                )
            finally:
                args.route_memory_prior_cache = old_prior_cache
                args.reuse_route_memory_prior_cache = old_reuse_prior_cache
            teacher_component_scores = None
            teacher_target_lengths = target_lengths
            if str(args.teacher_score_mode) == "route_first_soft":
                if bool(args.teacher_use_real_train_length):
                    teacher_target_lengths = haversine_km(train_query[:, :-1], train_query[:, 1:]).sum(axis=1).astype(np.float32)
                    log_stage("teacher: use real train lengths for train-split soft labels")
                teacher, teacher_scores, teacher_stats, teacher_component_scores = route_first_soft_teacher_scores(
                    args,
                    prior_train,
                    prior,
                    train_candidates,
                    teacher_target_lengths,
                    sem_total,
                )
                teacher_stats["teacher_target_length_source"] = "train_real" if bool(args.teacher_use_real_train_length) else "train_prior_predicted"
            elif str(args.selection_policy) == "guarded_two_stage":
                teacher, teacher_scores, teacher_stats = guarded_teacher_scores(
                    args,
                    prior_scores,
                    sem_total,
                    train_candidates,
                    target_lengths,
                )
            else:
                teacher_scores = prior_scores + float(args.semantic_weight) * sem_total
                teacher = teacher_scores.argmin(axis=1).astype(np.int64)
                teacher_stats = {
                    "selection_policy": "weighted_sum",
                    "teacher_selected_semantic_cost_mean": float(sem_total[np.arange(len(teacher)), teacher].mean()),
                    "teacher_selected_prior_score_mean": float(prior_scores[np.arange(len(teacher)), teacher].mean()),
                }
        teacher_stats["semantic_candidate_stats"] = train_append_stats
        log_stage("train reranker")
        model, history = train_reranker(
            args,
            train_features,
            teacher,
            teacher_scores,
            device,
            teacher_component_scores=teacher_component_scores,
        )
        log_stage("save reranker checkpoint/history")
        torch.save({"model_state": model.state_dict(), "query_feature_stats": q_stats, "args": vars(args)}, args.output_dir / "model.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output_dir / "teacher_stats.json").write_text(
            json.dumps({"teacher_stats": teacher_stats, "teacher_hist": np.bincount(teacher, minlength=int(args.candidate_topm)).tolist()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.savez_compressed(
            args.output_dir / "teacher_scores.npz",
            teacher=teacher.astype(np.int64),
            teacher_scores=teacher_scores.astype(np.float32),
            prior_scores=prior_scores.astype(np.float32),
            semantic_cost=sem_total.astype(np.float32),
            target_lengths=target_lengths.astype(np.float32),
            teacher_target_lengths=np.asarray(teacher_target_lengths, dtype=np.float32),
            teacher_component_scores=(
                teacher_component_scores.astype(np.float32)
                if teacher_component_scores is not None
                else np.zeros((*teacher_scores.shape, 0), dtype=np.float32)
            ),
            teacher_component_names=np.asarray(["density", "transition", "target_length", "raw_target_length"], dtype="<U24"),
        )
    summary = {}
    if val is not None:
        log_stage("evaluate val split")
        val_t1, val_t3, val_state = evaluate_split(
            args,
            args.val_output_name,
            model,
            memory,
            prior_train,
            val,
            args.val_real_npz,
            args.val_input,
            prior,
            proj,
            q_stats,
            calibrator,
            device,
            realism_model=realism_model,
            realism_q_stats=realism_q_stats,
            semantic_candidate_npz=args.val_semantic_candidate_npz,
        )
        log_stage("evaluate val Table IV" if not args.skip_table4 else "skip val Table IV")
        val_t4 = None if args.skip_table4 else evaluate_table4(args, args.val_output_name, model, val_state, args.val_counterfactual_input, prior_train, prior, proj, q_stats, calibrator, device)
        summary["val"] = {
            "table1": val_t1,
            "table3": val_t3["table3_instruction_following"] if val_t3 is not None else None,
            "table4": val_t4["table4_counterfactual_controllability"] if val_t4 else None,
        }
    if test is not None:
        log_stage("evaluate test split")
        test_t1, test_t3, test_state = evaluate_split(
            args,
            args.test_output_name,
            model,
            memory,
            prior_train,
            test,
            args.test_real_npz,
            args.test_input,
            prior,
            proj,
            q_stats,
            calibrator,
            device,
            realism_model=realism_model,
            realism_q_stats=realism_q_stats,
            semantic_candidate_npz=args.test_semantic_candidate_npz,
        )
        log_stage("evaluate test Table IV" if not args.skip_table4 else "skip test Table IV")
        test_t4 = None if args.skip_table4 else evaluate_table4(args, args.test_output_name, model, test_state, args.test_counterfactual_input, prior_train, prior, proj, q_stats, calibrator, device)
        summary["test"] = {
            "table1": test_t1,
            "table3": test_t3["table3_instruction_following"] if test_t3 is not None else None,
            "table4": test_t4["table4_counterfactual_controllability"] if test_t4 else None,
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_stage("finished all requested splits")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
