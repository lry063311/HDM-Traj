from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from langdirecttraj.data.instruction_dataset import (
    InstructionTrajectoryDataset,
    collate_instruction_batch,
)
from langdirecttraj.eval.eval_anchor_decoder import (
    build_model,
    load_checkpoint,
    prepare_eval_batch,
)
from langdirecttraj.eval.instruction_geometry import (
    add_pace_metrics_to_rows,
    apply_pace_progress_labels,
    contract_labels_from_tensors,
    contract_direction_rule_scores,
    geometry_values,
    macro_f1,
    tensor_dict_to_rows,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.text.qwen_instruction_encoder import batch_encode_qwen_texts
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    make_attr_cond,
    move_batch,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _normalize_path_text(path: str | Path) -> str:
    return str(path).replace("\\", "/").rstrip("/")


def validate_qwen_plan_cache_alignment(input_path: Path, cache_path: Path | None) -> dict[str, Any] | None:
    if cache_path is None:
        return None
    manifest_path = Path(str(cache_path) + ".manifest.json")
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_input = manifest.get("input")
    if manifest_input:
        expected = _normalize_path_text(input_path)
        observed = _normalize_path_text(manifest_input)
        if observed != expected and not expected.endswith(observed) and not observed.endswith(expected):
            raise ValueError(
                "Qwen plan cache manifest input does not match --input. "
                f"cache={cache_path}, manifest_input={manifest_input}, input={input_path}. "
                "Use the matching split cache or omit --qwen-plan-cache to run the planner online."
            )
    return manifest


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _labels_from_token_ids(token_ids: torch.Tensor) -> list[dict[str, str]]:
    ids = token_ids.detach().cpu().long().numpy()
    labels: list[dict[str, str]] = []
    for row in ids:
        item = {}
        for field_idx, field in enumerate(DEFAULT_MOBILITY_VOCAB.fields):
            item[field] = DEFAULT_MOBILITY_VOCAB.id_to_token[field].get(int(row[field_idx]), "unknown")
        labels.append(item)
    return labels


def _qwen_bundle_with_predictions(
    qwen_provider,
    instructions: list[str],
    device: torch.device,
) -> dict[str, Any] | None:
    if qwen_provider is None:
        return None
    tokenizer = qwen_provider["tokenizer"]
    planner = qwen_provider["planner"]
    encoded = batch_encode_qwen_texts(
        tokenizer,
        instructions,
        max_len=int(qwen_provider["max_len"]),
        device=device,
    )
    with torch.no_grad():
        outputs = planner(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    logits = dict(outputs.get("logits", {}))
    pred_ids = torch.stack([logits[field].argmax(dim=-1) for field in DEFAULT_MOBILITY_VOCAB.fields], dim=1)
    confidences = {}
    for field in DEFAULT_MOBILITY_VOCAB.fields:
        probs = torch.softmax(logits[field].float(), dim=-1)
        confidences[field] = probs.amax(dim=-1).detach().cpu().tolist()
    semantic_plan_tokens = outputs["semantic_plan_tokens"].detach()
    return {
        "source": "planner",
        "semantic_plan_tokens": semantic_plan_tokens,
        "pred_ids": pred_ids,
        "pred_labels": _labels_from_token_ids(pred_ids),
        "confidences": confidences,
        "decoder_bundle": {
            "semantic_plan_tokens": semantic_plan_tokens,
            "global_semantic_latent": outputs.get("global_semantic_latent"),
            "field_control_latents": {
                key: value.detach()
                for key, value in dict(outputs.get("field_control_latents", {})).items()
            },
            "field_control_logits": {
                key: value.detach()
                for key, value in dict(outputs.get("field_control_logits", {})).items()
            },
            "semantic_control_enabled": bool(outputs.get("semantic_control_enabled", False)),
            "field_control_fields": list(outputs.get("field_control_fields", [])),
        },
    }


def _cached_qwen_bundle(batch: dict[str, Any], device: torch.device) -> dict[str, Any] | None:
    cached = batch.get("qwen_semantic_plan_tokens")
    if cached is None:
        return None
    semantic_plan_tokens = cached.to(device=device)
    return {
        "source": "cache",
        "semantic_plan_tokens": semantic_plan_tokens,
        "pred_ids": None,
        "pred_labels": None,
        "confidences": {},
        "decoder_bundle": {
            "semantic_plan_tokens": semantic_plan_tokens,
            "global_semantic_latent": None,
            "field_control_latents": {},
            "field_control_logits": {},
            "semantic_control_enabled": False,
            "field_control_fields": [],
        },
    }


def _semantic_plan_stats(plan_tokens: torch.Tensor, row_idx: int, preview_values: int) -> dict[str, Any]:
    row = plan_tokens[row_idx].detach().float().cpu()
    flat = row.reshape(-1)
    stats: dict[str, Any] = {
        "shape": list(row.shape),
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "std": float(flat.std(unbiased=False).item()) if flat.numel() else 0.0,
        "l2": float(torch.linalg.norm(flat).item()) if flat.numel() else 0.0,
    }
    if int(preview_values) > 0:
        stats["preview"] = [float(x) for x in flat[: int(preview_values)].tolist()]
    return stats


def _candidate_errors_m(
    candidates: torch.Tensor,
    target: torch.Tensor,
    route_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    errors = torch.linalg.norm(candidates - target[:, None], dim=-1) * route_scale.view(-1, 1, 1)
    return errors.mean(dim=-1), errors[:, :, -1]


def _field_mismatch_count(candidate_labels: list[dict[str, str]], true_labels: dict[str, str]) -> int:
    return int(
        sum(
            str(candidate_labels[field]) != str(true_labels.get(field, "unknown"))
            for field in DEFAULT_MOBILITY_VOCAB.fields
        )
    )


def _diagnosis_summary(
    target_contract_direction_match_rate: float,
    qwen_direction_accuracy: float | None,
    candidate_has_true_rate: float,
    selected_missed_available_true_rate: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    if target_contract_direction_match_rate < 0.80:
        reasons.append("label_contract_mismatch")
    if qwen_direction_accuracy is not None and qwen_direction_accuracy < 0.40:
        reasons.append("qwen_direction_prediction_weak")
    if candidate_has_true_rate < 0.65:
        reasons.append("candidate_generation_gap")
    if candidate_has_true_rate >= 0.65 and selected_missed_available_true_rate > 0.25:
        reasons.append("reranker_selection_gap")
    if not reasons:
        reasons.append("no_single_dominant_gap")
    return {
        "likely_causes": reasons,
        "interpretation": {
            "label_contract_mismatch": "JSON target labels and Table3 geometry contract disagree often.",
            "qwen_direction_prediction_weak": "Qwen planner predicts the requested direction poorly.",
            "candidate_generation_gap": "The K-candidate pool often lacks a candidate with target direction.",
            "reranker_selection_gap": "A target-direction candidate exists, but reranker often selects another mode.",
            "no_single_dominant_gap": "No thresholded single cause dominates; inspect per-case rows.",
        },
    }


def _geometry_focus(row: dict[str, Any]) -> dict[str, float]:
    keys = [
        "local_dx",
        "local_dy",
        "local_heading_diff_rad",
        "local_abs_heading_diff_rad",
        "local_total_abs_turn_rad",
        "local_max_abs_turn_rad",
        "local_turn_count",
        "contract_net_heading_rad",
        "contract_total_abs_heading_rad",
        "contract_raw_total_abs_heading_rad",
        "mean_reference_distance_norm",
        "mean_reference_distance_m",
        "reference_violation_rate_30m",
        "mean_abs_lateral_offset_m",
        "mean_signed_lateral_offset_m",
        "turn_density_score",
    ]
    out: dict[str, float] = {}
    for key in keys:
        if key in row:
            try:
                out[key] = float(row[key])
            except (TypeError, ValueError):
                pass
    return out


def _reference_metric_rows(
    points: torch.Tensor,
    reference: torch.Tensor,
    route_scale: torch.Tensor,
) -> list[dict[str, float]]:
    route_scale = route_scale.view(-1).to(device=points.device, dtype=points.dtype)
    distance_m = torch.linalg.norm(points - reference, dim=-1) * route_scale[:, None]
    rows: dict[str, torch.Tensor] = {
        "mean_reference_distance_m": distance_m.mean(dim=1),
        "reference_violation_rate_30m": (distance_m > 30.0).to(dtype=points.dtype).mean(dim=1),
    }
    if points.shape[1] < 2 or reference.shape[1] < 2:
        zeros = points.new_zeros(points.shape[0])
        rows["mean_signed_lateral_offset_m"] = zeros
        rows["mean_abs_lateral_offset_m"] = zeros
    else:
        diffs = torch.zeros_like(reference)
        diffs[:, 1:-1] = reference[:, 2:] - reference[:, :-2]
        diffs[:, 0] = reference[:, 1] - reference[:, 0]
        diffs[:, -1] = reference[:, -1] - reference[:, -2]
        tangent = diffs / torch.linalg.norm(diffs, dim=-1, keepdim=True).clamp_min(1e-8)
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
        lateral_m = ((points - reference) * normal).sum(dim=-1) * route_scale[:, None]
        rows["mean_signed_lateral_offset_m"] = lateral_m.mean(dim=1)
        rows["mean_abs_lateral_offset_m"] = lateral_m.abs().mean(dim=1)
    return tensor_dict_to_rows(rows)


def _merge_geometry_rows(
    rows: list[dict[str, Any]],
    reference_rows: list[dict[str, float]],
) -> list[dict[str, Any]]:
    return [{**row, **ref_row} for row, ref_row in zip(rows, reference_rows)]


def _direction_contract_gaps(target_direction: str, geom: dict[str, float]) -> dict[str, float]:
    """Expose why a candidate misses the Table-3 direction contract."""
    scores = contract_direction_rule_scores(geom)
    target_score = float(scores.get(target_direction, float("inf")))
    best_other = min(
        (float(score) for label, score in scores.items() if label != target_direction),
        default=float("inf"),
    )
    score_margin_gap = max(0.0, target_score - best_other)
    local_dx = float(geom.get("local_dx", 0.0))
    abs_heading = float(geom.get("local_abs_heading_diff_rad", abs(float(geom.get("local_heading_diff_rad", 0.0)))))
    total_abs = float(geom.get("local_total_abs_turn_rad", geom.get("contract_total_abs_heading_rad", 0.0)))
    max_abs = float(geom.get("local_max_abs_turn_rad", 0.0))
    turn_count = float(geom.get("local_turn_count", 0.0))
    if target_direction == "u_turn":
        return {
            "score_margin_gap": score_margin_gap,
            "uturn_dx_gap": max(0.0, local_dx - (-0.05)),
            "uturn_heading_gap": max(0.0, 2.0 - abs_heading),
            "uturn_max_abs_gap": max(0.0, 1.0 - max_abs),
        }
    if target_direction == "multi_turn":
        return {
            "score_margin_gap": score_margin_gap,
            "multi_total_abs_gap": max(0.0, 2.0 - total_abs),
            "multi_turn_count_gap": max(0.0, 3.0 - turn_count),
        }
    if target_direction in scores:
        return {"score_margin_gap": score_margin_gap}
    return {}


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    attr_dim: int,
    dataset: str,
    qwen_provider=None,
    max_cases: int = 200,
    qwen_plan_preview_values: int = 0,
) -> dict[str, Any]:
    direction_names = list(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"].keys())
    reference_names = list(DEFAULT_MOBILITY_VOCAB.token_to_id["reference_adherence"].keys())
    selected_true: list[str] = []
    selected_pred: list[str] = []
    direction_oracle_pred: list[str] = []
    reference_true: list[str] = []
    reference_selected_pred: list[str] = []
    reference_oracle_pred: list[str] = []
    target_contract_direction: list[str] = []
    qwen_direction_pred: list[str] = []
    selected_indices: list[int] = []
    direction_oracle_indices: list[int] = []
    ade_oracle_indices: list[int] = []
    control_oracle_indices: list[int] = []
    candidate_has_true_values: list[float] = []
    candidate_has_true_by_label: dict[str, list[float]] = defaultdict(list)
    reference_has_true_values: list[float] = []
    reference_has_true_by_label: dict[str, list[float]] = defaultdict(list)
    selected_missed_available_true = 0
    reference_selected_missed_available_true = 0
    no_true_candidate = 0
    reference_no_true_candidate = 0
    qwen_direction_correct: list[float] = []
    qwen_all_token_exact: list[float] = []
    semantic_plan_available_values: list[float] = []
    target_contract_direction_match: list[float] = []
    true_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    direction_oracle_counts: Counter[str] = Counter()
    reference_true_counts: Counter[str] = Counter()
    reference_selected_counts: Counter[str] = Counter()
    reference_oracle_counts: Counter[str] = Counter()
    target_contract_counts: Counter[str] = Counter()
    candidate_direction_counts: Counter[str] = Counter()
    candidate_reference_counts: Counter[str] = Counter()
    selected_index_counts: Counter[str] = Counter()
    direction_oracle_index_counts: Counter[str] = Counter()
    ade_oracle_index_counts: Counter[str] = Counter()
    control_oracle_index_counts: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    global_sample_idx = 0

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            target, reference, route_scales = prepare_eval_batch(batch, model.decoder.target_len)
            route_scale = route_scales[:, 0]
            attr_cond = make_attr_cond(batch, attr_dim, device)
            qwen_bundle = _cached_qwen_bundle(batch, device)
            if qwen_bundle is None:
                qwen_bundle = _qwen_bundle_with_predictions(qwen_provider, batch["instruction"], device)
            semantic_bundle = qwen_bundle["decoder_bundle"] if qwen_bundle is not None else None
            pred_out = model(
                reference,
                batch["mobility_tokens"],
                attr_cond,
                semantic_plan_tokens=semantic_bundle,
                return_aux=True,
            )
            if isinstance(pred_out, tuple):
                candidates, aux = pred_out
            else:
                candidates, aux = pred_out, {}
            if candidates.ndim == 3:
                candidates = candidates[:, None]
            logits = aux.get("candidate_logits") if isinstance(aux, dict) else None
            if torch.is_tensor(logits):
                selected_idx = logits[:, : candidates.shape[1]].argmax(dim=1)
                candidate_logits = logits[:, : candidates.shape[1]].detach().cpu().float().tolist()
            else:
                selected_idx = torch.zeros(candidates.shape[0], dtype=torch.long, device=device)
                candidate_logits = [[] for _ in range(candidates.shape[0])]

            batch_size, num_candidates, target_len, dim = candidates.shape
            flat_candidates = candidates.reshape(batch_size * num_candidates, target_len, dim)
            flat_reference = reference[:, None].expand_as(candidates).reshape_as(flat_candidates)
            flat_scale = route_scale[:, None].expand(-1, num_candidates).reshape(-1)
            decoded_flat = contract_labels_from_tensors(
                flat_candidates.float(),
                flat_reference.float(),
                flat_scale.float(),
                dataset=dataset,
            )
            flat_geom_rows = tensor_dict_to_rows(
                geometry_values(flat_candidates.float(), flat_reference.float(), flat_scale.float())
            )
            flat_reference_metric_rows = _reference_metric_rows(
                flat_candidates.float(), flat_reference.float(), flat_scale.float()
            )
            flat_geom_rows = _merge_geometry_rows(flat_geom_rows, flat_reference_metric_rows)
            repeated_attrs = []
            for attrs in batch.get("trip_attributes", [{} for _ in range(batch_size)]):
                repeated_attrs.extend([attrs] * num_candidates)
            flat_geom_rows = add_pace_metrics_to_rows(flat_geom_rows, repeated_attrs)
            decoded_flat = apply_pace_progress_labels(decoded_flat, flat_geom_rows, repeated_attrs)
            decoded = np.asarray(decoded_flat, dtype=object).reshape(batch_size, num_candidates)

            target_contract = contract_labels_from_tensors(
                target.float(),
                reference.float(),
                route_scale.float(),
                dataset=dataset,
            )
            target_geom_rows = tensor_dict_to_rows(geometry_values(target.float(), reference.float(), route_scale.float()))
            target_reference_metric_rows = _reference_metric_rows(
                target.float(), reference.float(), route_scale.float()
            )
            target_geom_rows = _merge_geometry_rows(target_geom_rows, target_reference_metric_rows)
            target_geom_rows = add_pace_metrics_to_rows(
                target_geom_rows,
                batch.get("trip_attributes", [{} for _ in range(batch_size)]),
            )
            target_contract = apply_pace_progress_labels(
                target_contract,
                target_geom_rows,
                batch.get("trip_attributes", [{} for _ in range(batch_size)]),
            )

            ade, fde = _candidate_errors_m(candidates, target, route_scale)
            ade_oracle_idx = ade.argmin(dim=1)
            true_label_rows = [dict(labels) for labels in batch["labels"]]
            if qwen_bundle is not None and qwen_bundle.get("pred_labels") is not None:
                qwen_pred_labels = qwen_bundle["pred_labels"]
            else:
                qwen_pred_labels = [None] * batch_size
            qwen_confidences = qwen_bundle["confidences"] if qwen_bundle is not None else {}
            qwen_plan = qwen_bundle["semantic_plan_tokens"] if qwen_bundle is not None else None
            if qwen_plan is not None:
                semantic_plan_available_values.extend([1.0] * batch_size)
            else:
                semantic_plan_available_values.extend([0.0] * batch_size)

            for row_idx in range(batch_size):
                true_labels = true_label_rows[row_idx]
                true_direction = str(true_labels.get("direction", "unknown"))
                row_candidate_labels = [dict(decoded[row_idx, mode_idx]) for mode_idx in range(num_candidates)]
                row_geom_rows = flat_geom_rows[
                    row_idx * num_candidates : (row_idx + 1) * num_candidates
                ]
                row_direction_labels = [str(labels.get("direction", "unknown")) for labels in row_candidate_labels]
                row_reference_labels = [
                    str(labels.get("reference_adherence", "unknown"))
                    for labels in row_candidate_labels
                ]
                row_selected_idx = int(selected_idx[row_idx].detach().cpu().item())
                row_selected_direction = row_direction_labels[row_selected_idx]
                true_reference = str(true_labels.get("reference_adherence", "unknown"))
                row_selected_reference = row_reference_labels[row_selected_idx]
                row_ade_oracle_idx = int(ade_oracle_idx[row_idx].detach().cpu().item())
                match_indices = [idx for idx, label in enumerate(row_direction_labels) if label == true_direction]
                has_true = bool(match_indices)
                if has_true:
                    row_direction_oracle_idx = min(
                        match_indices,
                        key=lambda idx: float(ade[row_idx, idx].detach().cpu().item()),
                    )
                    if row_selected_direction != true_direction:
                        selected_missed_available_true += 1
                else:
                    row_direction_oracle_idx = row_selected_idx
                    no_true_candidate += 1
                reference_match_indices = [
                    idx for idx, label in enumerate(row_reference_labels) if label == true_reference
                ]
                reference_has_true = bool(reference_match_indices)
                if reference_has_true:
                    row_reference_oracle_idx = min(
                        reference_match_indices,
                        key=lambda idx: float(ade[row_idx, idx].detach().cpu().item()),
                    )
                    if row_selected_reference != true_reference:
                        reference_selected_missed_available_true += 1
                else:
                    row_reference_oracle_idx = row_selected_idx
                    reference_no_true_candidate += 1
                mismatch_counts = [
                    _field_mismatch_count(labels, true_labels)
                    for labels in row_candidate_labels
                ]
                row_control_oracle_idx = int(np.argmin(mismatch_counts))
                row_direction_oracle_label = row_direction_labels[row_direction_oracle_idx]
                row_target_contract_direction = str(target_contract[row_idx].get("direction", "unknown"))

                selected_true.append(true_direction)
                selected_pred.append(row_selected_direction)
                direction_oracle_pred.append(row_direction_oracle_label)
                reference_true.append(true_reference)
                reference_selected_pred.append(row_selected_reference)
                reference_oracle_pred.append(row_reference_labels[row_reference_oracle_idx])
                target_contract_direction.append(row_target_contract_direction)
                selected_indices.append(row_selected_idx)
                direction_oracle_indices.append(row_direction_oracle_idx)
                ade_oracle_indices.append(row_ade_oracle_idx)
                control_oracle_indices.append(row_control_oracle_idx)
                candidate_has_true_values.append(float(has_true))
                candidate_has_true_by_label[true_direction].append(float(has_true))
                reference_has_true_values.append(float(reference_has_true))
                reference_has_true_by_label[true_reference].append(float(reference_has_true))
                target_contract_direction_match.append(float(row_target_contract_direction == true_direction))

                true_counts[true_direction] += 1
                selected_counts[row_selected_direction] += 1
                direction_oracle_counts[row_direction_oracle_label] += 1
                reference_true_counts[true_reference] += 1
                reference_selected_counts[row_selected_reference] += 1
                reference_oracle_counts[row_reference_labels[row_reference_oracle_idx]] += 1
                target_contract_counts[row_target_contract_direction] += 1
                for label in row_direction_labels:
                    candidate_direction_counts[label] += 1
                for label in row_reference_labels:
                    candidate_reference_counts[label] += 1
                selected_index_counts[str(row_selected_idx)] += 1
                direction_oracle_index_counts[str(row_direction_oracle_idx)] += 1
                ade_oracle_index_counts[str(row_ade_oracle_idx)] += 1
                control_oracle_index_counts[str(row_control_oracle_idx)] += 1

                qwen_case = None
                if qwen_plan is not None:
                    qwen_case = {
                        "source": str(qwen_bundle.get("source", "unknown")) if qwen_bundle is not None else "none",
                        "semantic_plan": _semantic_plan_stats(
                            qwen_plan,
                            row_idx,
                            preview_values=int(qwen_plan_preview_values),
                        ),
                    }
                if qwen_pred_labels[row_idx] is not None:
                    qwen_labels = dict(qwen_pred_labels[row_idx])
                    qwen_direction = str(qwen_labels.get("direction", "unknown"))
                    qwen_direction_pred.append(qwen_direction)
                    qwen_direction_correct.append(float(qwen_direction == true_direction))
                    qwen_all_token_exact.append(
                        float(
                            all(
                                str(qwen_labels.get(field, "unknown")) == str(true_labels.get(field, "unknown"))
                                for field in DEFAULT_MOBILITY_VOCAB.fields
                            )
                        )
                    )
                    if qwen_case is None:
                        qwen_case = {}
                    qwen_case.update(
                        {
                            "pred_labels": qwen_labels,
                            "direction_correct": bool(qwen_direction == true_direction),
                            "all_token_exact": bool(qwen_all_token_exact[-1]),
                            "confidence": {
                            field: float(qwen_confidences[field][row_idx])
                            for field in DEFAULT_MOBILITY_VOCAB.fields
                            if field in qwen_confidences
                            },
                        }
                    )

                if len(cases) < int(max_cases):
                    cases.append(
                        {
                            "sample_idx": int(global_sample_idx),
                            "instruction": str(batch["instruction"][row_idx]),
                            "target_labels_json": {
                                field: str(true_labels.get(field, "unknown"))
                                for field in DEFAULT_MOBILITY_VOCAB.fields
                            },
                            "target_labels_contract": {
                                field: str(target_contract[row_idx].get(field, "unknown"))
                                for field in DEFAULT_MOBILITY_VOCAB.fields
                            },
                            "target_contract_direction_match": bool(
                                row_target_contract_direction == true_direction
                            ),
                            "true_reference_adherence": true_reference,
                            "route_scale_m": float(route_scale[row_idx].detach().cpu().item()),
                            "target_geometry_focus": _geometry_focus(target_geom_rows[row_idx]),
                            "candidate_labels": row_candidate_labels,
                            "candidate_direction_labels": row_direction_labels,
                            "candidate_reference_labels": row_reference_labels,
                            "candidate_geometry_focus": [
                                _geometry_focus(geom_row) for geom_row in row_geom_rows
                            ],
                            "candidate_direction_contract_gaps": [
                                _direction_contract_gaps(true_direction, _geometry_focus(geom_row))
                                for geom_row in row_geom_rows
                            ],
                            "candidate_field_mismatch_counts": mismatch_counts,
                            "candidate_ade_m": [
                                float(x) for x in ade[row_idx].detach().cpu().float().tolist()
                            ],
                            "candidate_fde_m": [
                                float(x) for x in fde[row_idx].detach().cpu().float().tolist()
                            ],
                            "candidate_logits": candidate_logits[row_idx],
                            "selected_idx": row_selected_idx,
                            "selected_direction": row_selected_direction,
                            "selected_reference_adherence": row_selected_reference,
                            "direction_oracle_idx": row_direction_oracle_idx,
                            "direction_oracle_direction": row_direction_oracle_label,
                            "reference_oracle_idx": int(row_reference_oracle_idx),
                            "reference_oracle_reference_adherence": row_reference_labels[row_reference_oracle_idx],
                            "ade_oracle_idx": row_ade_oracle_idx,
                            "control_oracle_idx": row_control_oracle_idx,
                            "has_true_direction_candidate": has_true,
                            "has_true_reference_candidate": reference_has_true,
                            "selected_matches_direction": bool(row_selected_direction == true_direction),
                            "selected_matches_reference_adherence": bool(
                                row_selected_reference == true_reference
                            ),
                            "selected_equals_direction_oracle": bool(row_selected_idx == row_direction_oracle_idx),
                            "selected_equals_reference_oracle": bool(
                                row_selected_idx == row_reference_oracle_idx
                            ),
                            "selected_equals_ade_oracle": bool(row_selected_idx == row_ade_oracle_idx),
                            "selected_equals_control_oracle": bool(row_selected_idx == row_control_oracle_idx),
                            "qwen": qwen_case,
                        }
                    )
                global_sample_idx += 1

    selected_metrics = macro_f1(selected_true, selected_pred, direction_names, include_empty=False)
    direction_oracle_metrics = macro_f1(
        selected_true,
        direction_oracle_pred,
        direction_names,
        include_empty=False,
    )
    reference_selected_metrics = macro_f1(
        reference_true,
        reference_selected_pred,
        reference_names,
        include_empty=False,
    )
    reference_oracle_metrics = macro_f1(
        reference_true,
        reference_oracle_pred,
        reference_names,
        include_empty=False,
    )
    target_contract_metrics = macro_f1(
        selected_true,
        target_contract_direction,
        direction_names,
        include_empty=False,
    )
    qwen_metrics = None
    if qwen_direction_pred:
        qwen_metrics = macro_f1(selected_true, qwen_direction_pred, direction_names, include_empty=False)

    has_true_rate = float(np.mean(candidate_has_true_values)) if candidate_has_true_values else float("nan")
    has_true_count = int(sum(candidate_has_true_values))
    missed_available_rate = (
        float(selected_missed_available_true / has_true_count) if has_true_count > 0 else float("nan")
    )
    reference_has_true_rate = (
        float(np.mean(reference_has_true_values)) if reference_has_true_values else float("nan")
    )
    reference_has_true_count = int(sum(reference_has_true_values))
    reference_missed_available_rate = (
        float(reference_selected_missed_available_true / reference_has_true_count)
        if reference_has_true_count > 0
        else float("nan")
    )
    target_contract_match_rate = (
        float(np.mean(target_contract_direction_match)) if target_contract_direction_match else float("nan")
    )
    qwen_direction_accuracy = (
        float(np.mean(qwen_direction_correct)) if qwen_direction_correct else None
    )
    metrics: dict[str, Any] = {
        "num_samples": int(len(selected_true)),
        "num_candidates": int(getattr(model.decoder, "num_candidates", 1)),
        "direction": {
            "selected_f1": float(selected_metrics["macro_f1"]),
            "selected_accuracy": float(selected_metrics["accuracy"]),
            "candidate_label_oracle_f1": float(direction_oracle_metrics["macro_f1"]),
            "candidate_label_oracle_accuracy": float(direction_oracle_metrics["accuracy"]),
            "candidate_has_true_rate": has_true_rate,
            "candidate_has_true_rate_by_label": {
                label: float(np.mean(values)) if values else float("nan")
                for label, values in sorted(candidate_has_true_by_label.items())
            },
            "failure_count_no_true_candidate": int(no_true_candidate),
            "selected_missed_available_true_count": int(selected_missed_available_true),
            "selected_missed_available_true_rate": missed_available_rate,
            "selected_metrics": selected_metrics,
            "candidate_label_oracle_metrics": direction_oracle_metrics,
        },
        "reference_adherence": {
            "selected_f1": float(reference_selected_metrics["macro_f1"]),
            "selected_accuracy": float(reference_selected_metrics["accuracy"]),
            "candidate_label_oracle_f1": float(reference_oracle_metrics["macro_f1"]),
            "candidate_label_oracle_accuracy": float(reference_oracle_metrics["accuracy"]),
            "candidate_has_true_rate": reference_has_true_rate,
            "candidate_has_true_rate_by_label": {
                label: float(np.mean(values)) if values else float("nan")
                for label, values in sorted(reference_has_true_by_label.items())
            },
            "failure_count_no_true_candidate": int(reference_no_true_candidate),
            "selected_missed_available_true_count": int(reference_selected_missed_available_true),
            "selected_missed_available_true_rate": reference_missed_available_rate,
            "selected_metrics": reference_selected_metrics,
            "candidate_label_oracle_metrics": reference_oracle_metrics,
        },
        "label_contract_alignment": {
            "target_contract_direction_match_rate": target_contract_match_rate,
            "target_contract_direction_f1_against_json_label": float(target_contract_metrics["macro_f1"]),
            "target_contract_direction_accuracy_against_json_label": float(target_contract_metrics["accuracy"]),
            "target_contract_direction_metrics": target_contract_metrics,
        },
        "qwen": {
            "semantic_plan_available": bool(
                semantic_plan_available_values and float(np.mean(semantic_plan_available_values)) > 0.0
            ),
            "semantic_plan_available_rate": (
                float(np.mean(semantic_plan_available_values)) if semantic_plan_available_values else 0.0
            ),
            "token_prediction_available": bool(qwen_metrics is not None),
            "direction_f1_against_json_label": (
                float(qwen_metrics["macro_f1"]) if qwen_metrics is not None else None
            ),
            "direction_accuracy_against_json_label": qwen_direction_accuracy,
            "all_token_exact_rate": (
                float(np.mean(qwen_all_token_exact)) if qwen_all_token_exact else None
            ),
            "direction_metrics": qwen_metrics,
        },
        "counts": {
            "true_direction": _counter_to_dict(true_counts),
            "selected_direction": _counter_to_dict(selected_counts),
            "direction_oracle": _counter_to_dict(direction_oracle_counts),
            "target_contract_direction": _counter_to_dict(target_contract_counts),
            "candidate_direction_total": _counter_to_dict(candidate_direction_counts),
            "true_reference_adherence": _counter_to_dict(reference_true_counts),
            "selected_reference_adherence": _counter_to_dict(reference_selected_counts),
            "reference_adherence_oracle": _counter_to_dict(reference_oracle_counts),
            "candidate_reference_adherence_total": _counter_to_dict(candidate_reference_counts),
            "selected_index": _counter_to_dict(selected_index_counts),
            "direction_oracle_index": _counter_to_dict(direction_oracle_index_counts),
            "ade_oracle_index": _counter_to_dict(ade_oracle_index_counts),
            "control_oracle_index": _counter_to_dict(control_oracle_index_counts),
        },
        "diagnosis": _diagnosis_summary(
            target_contract_match_rate,
            qwen_direction_accuracy,
            has_true_rate,
            missed_available_rate if np.isfinite(missed_available_rate) else 0.0,
        ),
        "cases": cases,
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the Table 3 instruction-to-candidate-to-reranker control chain."
    )
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "table3_control_chain_diagnostics")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--qwen-plan-cache",
        type=Path,
        default=None,
        help="Optional cached semantic_plan_tokens .npy aligned with --input; avoids rerunning Qwen.",
    )
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--qwen-plan-preview-values", type=int, default=0)
    parser.add_argument(
        "--allow-qwen-plan-cache-mismatch",
        action="store_true",
        help="Bypass cache manifest split checks. Use only for deliberate debugging.",
    )
    parser.add_argument("--token-calibration-path", type=Path, default=None)
    parser.add_argument("--direction-control-strength-override", type=float, default=None)
    parser.add_argument(
        "--direction-control-type-override",
        choices=["none", "token_basis", "hybrid_rule_basis", "direction_role_basis", "direction_structural_residual"],
        default=None,
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
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint, device, token_calibration_path=args.token_calibration_path)
    decoder = getattr(model, "decoder", None)
    if args.direction_control_type_override is not None:
        if decoder is None or not hasattr(decoder, "direction_control_type"):
            raise ValueError("--direction-control-type-override requires a compatible decoder.")
        decoder.direction_control_type = str(args.direction_control_type_override)
    if args.direction_control_strength_override is not None:
        if decoder is None or not hasattr(decoder, "set_direction_control_strength"):
            raise ValueError("--direction-control-strength-override requires a compatible decoder.")
        decoder.set_direction_control_strength(float(args.direction_control_strength_override))
    if args.reference_positive_candidate_strength_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_positive_candidate_strength"):
            raise ValueError("--reference-positive-candidate-strength-override requires a compatible decoder.")
        decoder.set_reference_positive_candidate_strength(float(args.reference_positive_candidate_strength_override))
    if args.reference_positive_candidate_scope_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_positive_candidate_scope"):
            raise ValueError("--reference-positive-candidate-scope-override requires a compatible decoder.")
        decoder.set_reference_positive_candidate_scope(str(args.reference_positive_candidate_scope_override))
    if args.reference_distance_candidate_strength_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_distance_candidate_strength"):
            raise ValueError("--reference-distance-candidate-strength-override requires a compatible decoder.")
        decoder.set_reference_distance_candidate_strength(float(args.reference_distance_candidate_strength_override))
    if args.reference_distance_candidate_scope_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_distance_candidate_scope"):
            raise ValueError("--reference-distance-candidate-scope-override requires a compatible decoder.")
        decoder.set_reference_distance_candidate_scope(str(args.reference_distance_candidate_scope_override))
    if args.reference_distance_candidate_mode_override is not None:
        if decoder is None or not hasattr(decoder, "set_reference_distance_candidate_mode"):
            raise ValueError("--reference-distance-candidate-mode-override requires a compatible decoder.")
        decoder.set_reference_distance_candidate_mode(str(args.reference_distance_candidate_mode_override))
    saved_args = checkpoint.get("args", {})
    attr_dim = int(saved_args.get("attr_dim", 6))
    qwen_plan_cache_manifest = None
    if not args.allow_qwen_plan_cache_mismatch:
        qwen_plan_cache_manifest = validate_qwen_plan_cache_alignment(args.input, args.qwen_plan_cache)
    dataset = InstructionTrajectoryDataset(
        args.input,
        max_samples=args.max_samples,
        target_key=args.target_key,
        qwen_plan_cache_path=args.qwen_plan_cache,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_instruction_batch,
    )
    qwen_provider = (
        build_qwen_provider(args.qwen_planner_checkpoint, args, device)
        if args.qwen_planner_checkpoint
        else None
    )
    metrics = evaluate(
        model,
        loader,
        device,
        attr_dim,
        dataset=args.dataset,
        qwen_provider=qwen_provider,
        max_cases=args.max_cases,
        qwen_plan_preview_values=args.qwen_plan_preview_values,
    )
    cases = metrics.pop("cases")
    metrics.update(
        {
            "dataset": args.dataset,
            "input": str(args.input),
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "target_key": args.target_key,
            "qwen_planner_checkpoint": str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None,
            "qwen_plan_cache": str(args.qwen_plan_cache) if args.qwen_plan_cache else None,
            "qwen_plan_cache_manifest": qwen_plan_cache_manifest,
            "qwen_model_name": args.qwen_model_name,
            "direction_control_type_override": args.direction_control_type_override,
            "direction_control_strength_override": args.direction_control_strength_override,
            "reference_positive_candidate_strength_override": args.reference_positive_candidate_strength_override,
            "reference_positive_candidate_scope_override": args.reference_positive_candidate_scope_override,
            "reference_distance_candidate_strength_override": args.reference_distance_candidate_strength_override,
            "reference_distance_candidate_scope_override": args.reference_distance_candidate_scope_override,
            "reference_distance_candidate_mode_override": args.reference_distance_candidate_mode_override,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    cases_path = args.output_dir / "cases.jsonl"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    cases_path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + ("\n" if cases else ""),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {**metrics, "metrics_path": str(metrics_path), "cases_path": str(cases_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
