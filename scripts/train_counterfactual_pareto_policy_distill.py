from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from langdirecttraj.data.instruction_dataset import collate_instruction_batch
from langdirecttraj.eval.eval_anchor_decoder import build_model, load_checkpoint, prepare_eval_batch
from langdirecttraj.eval.eval_counterfactual_controllability import (
    group_counterfactual_pairs,
    read_records,
    tensor_record,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.train.control_losses import load_counterfactual_transition_table
from langdirecttraj.train.train_anchor_decoder import (
    extract_qwen_semantic_bundle,
    make_attr_cond,
    road_condition_kwargs,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Pareto-policy distillation for the Stage-1 counterfactual "
            "selection head. The generator and ordinary reranker are frozen."
        )
    )
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidate-labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--pair-sampling", choices=["balanced_field", "sequential"], default="balanced_field")
    parser.add_argument("--fields", default="")
    parser.add_argument("--target-key", default="trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--transition-diagnostics-path", type=Path, default=None)
    parser.add_argument("--transition-min-count", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-qwen", action="store_true")
    parser.add_argument("--token-calibration-path", type=Path, default=None)
    parser.add_argument("--counterfactual-adapter-scale", type=float, default=1.0)
    parser.add_argument(
        "--counterfactual-direction-residual-type",
        default=None,
        choices=[
            None,
            "",
            "shared",
            "transition_contract_pulse",
            "transition_arc_skeleton",
            "transition_arc_multiturn_burst",
            "transition_hybrid_skeleton_reflect",
            "transition_pivot_skeleton",
            "transition_reflect_skeleton",
            "transition_skeleton_reversal",
            "transition_skeleton_reversal_scurve",
            "transition_skeleton",
            "transition_template",
            "transition_template_balanced",
            "transition_template_contract",
            "transition_template_mean_projected",
            "transition_template_projected",
        ],
        help="Optional checkpoint-level direction residual type for counterfactual policy distillation.",
    )
    parser.add_argument("--counterfactual-policy-blend-mode", choices=["override", "blend"], default="override")
    parser.add_argument("--counterfactual-policy-gate-init", type=float, default=0.99)
    parser.add_argument("--policy-head", choices=["metric", "generic"], default="metric")
    parser.add_argument("--counterfactual-metric-policy-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-metric-policy-feasible-first", action="store_true")
    parser.add_argument("--counterfactual-metric-policy-success-margin", type=float, default=0.0)
    parser.add_argument("--counterfactual-metric-policy-collateral-margin", type=float, default=0.08)
    parser.add_argument("--counterfactual-metric-policy-raw-collateral-margin", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--pairwise-margin-weight", type=float, default=0.5)
    parser.add_argument("--pairwise-margin", type=float, default=0.1)
    parser.add_argument("--soft-temperature", type=float, default=0.5)
    parser.add_argument("--high-collateral-margin", type=float, default=0.08)
    parser.add_argument("--raw-collateral-margin", type=float, default=0.15)
    parser.add_argument("--target-effect-negative-floor", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def _saved_args_dict(checkpoint: Dict) -> Dict:
    raw_args = checkpoint.get("args", {})
    return dict(raw_args) if isinstance(raw_args, dict) else vars(raw_args)


def _as_float(value, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    return result if np.isfinite(result) else float(default)


LabelKey = Tuple[int, int, str, str, str]


def _row_label_key(row: Dict) -> LabelKey:
    return (
        int(row.get("sample_id", -1)),
        int(row.get("counterfactual_id", row.get("sample_id", -1))),
        str(row.get("changed_field", "")),
        str(row.get("source_token", "")),
        str(row.get("target_token", "")),
    )


def _record_label_key(original: Dict, counterfactual: Dict) -> LabelKey:
    cf_meta = counterfactual.get("counterfactual", {})
    field = str(cf_meta.get("field", ""))
    source = str(cf_meta.get("from", original.get("labels", {}).get(field, "")))
    target = str(cf_meta.get("to", counterfactual.get("labels", {}).get(field, "")))
    return (
        int(original.get("sample_id", -1)),
        int(counterfactual.get("counterfactual_id", counterfactual.get("sample_id", -1))),
        field,
        source,
        target,
    )


def load_candidate_labels(path: Path) -> Dict[LabelKey, List[Dict]]:
    grouped: Dict[LabelKey, List[Dict]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("selection_policy", "")) != "hybrid_counterfactual":
                continue
            grouped[_row_label_key(row)].append(row)
    cleaned: Dict[LabelKey, List[Dict]] = {}
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int(item.get("mode", 0)))
        selected = [row for row in rows if bool(row.get("selected", False))]
        if not selected:
            continue
        cleaned[key] = rows
    if not cleaned:
        raise ValueError(
            f"No usable hybrid_counterfactual candidate labels found in {path}. "
            "Export labels with --selection-policy hybrid_counterfactual."
        )
    return cleaned


def _move_batch(batch: Dict, device: torch.device) -> Dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _trip_duration_tensor(batch: Dict, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            float(attrs.get("trip_duration_min", 0.0) or 0.0)
            if isinstance(attrs, dict)
            else 0.0
            for attrs in batch.get("trip_attributes", [{}])
        ],
        device=device,
        dtype=torch.float32,
    )


def _forward_record(
    model,
    record: Dict,
    attr_dim: int,
    target_key: str,
    device: torch.device,
    *,
    counterfactual_mode: bool = False,
    counterfactual_adapter_scale: float = 0.0,
    counterfactual_field_ids: torch.Tensor | None = None,
    base_selected_trajectory: torch.Tensor | None = None,
    base_mobility_tokens: torch.Tensor | None = None,
    base_selected_index: torch.Tensor | None = None,
):
    item = tensor_record(record, target_key=target_key)
    batch = _move_batch(collate_instruction_batch([item]), device)
    _, reference, route_scales = prepare_eval_batch(batch, model.decoder.target_len)
    attr_cond = make_attr_cond(batch, attr_dim, device)
    semantic_plan_tokens = None
    pred_out = model(
        reference,
        batch["mobility_tokens"],
        attr_cond,
        semantic_plan_tokens=semantic_plan_tokens,
        **road_condition_kwargs(batch),
        counterfactual_mode=bool(counterfactual_mode),
        counterfactual_adapter_scale=float(counterfactual_adapter_scale),
        counterfactual_field_ids=counterfactual_field_ids,
        base_selected_trajectory=base_selected_trajectory,
        base_mobility_tokens=base_mobility_tokens,
        base_selected_index=base_selected_index,
        route_scales=route_scales[:, 0],
        trip_duration_min=_trip_duration_tensor(batch, device),
        return_aux=True,
    )
    if not isinstance(pred_out, tuple):
        pred = pred_out[:, None] if pred_out.ndim == 3 else pred_out
        aux = {}
    else:
        pred, aux = pred_out
        if pred.ndim == 3:
            pred = pred[:, None]
    logits = aux.get("candidate_logits") if isinstance(aux, dict) else None
    if logits is None:
        logits = pred.new_zeros(pred.shape[:2])
    else:
        logits = logits[:, : pred.shape[1]]
    return {
        "batch": batch,
        "pred": pred,
        "aux": aux if isinstance(aux, dict) else {},
        "logits": logits,
    }


def _base_selection(model, original: Dict, attr_dim: int, target_key: str, device: torch.device):
    with torch.no_grad():
        base = _forward_record(model, original, attr_dim, target_key, device)
        logits = base["logits"]
        selected_index = logits.argmax(dim=1) if logits.ndim == 2 and logits.shape[1] > 0 else torch.zeros(
            1, device=device, dtype=torch.long
        )
        pred = base["pred"]
        gather_idx = selected_index.to(device=pred.device, dtype=torch.long).view(-1, 1, 1, 1)
        gather_idx = gather_idx.expand(-1, 1, pred.shape[-2], pred.shape[-1])
        selected_traj = pred.gather(1, gather_idx).squeeze(1)
    return selected_traj.detach(), base["batch"]["mobility_tokens"].detach(), selected_index.detach()


def _field_id_tensor(counterfactual: Dict, device: torch.device) -> torch.Tensor | None:
    field = str(counterfactual.get("counterfactual", {}).get("field", ""))
    if field not in DEFAULT_MOBILITY_VOCAB.fields:
        return None
    return torch.tensor([DEFAULT_MOBILITY_VOCAB.fields.index(field)], device=device, dtype=torch.long)


def _targets_from_rows(rows: List[Dict], num_modes: int, args: argparse.Namespace, device: torch.device):
    modes = [int(row.get("mode", idx)) for idx, row in enumerate(rows)]
    mode_to_row = {mode: row for mode, row in zip(modes, rows) if 0 <= mode < num_modes}
    if not mode_to_row:
        return None
    selected_modes = [
        int(row.get("mode", -1))
        for row in rows
        if bool(row.get("selected", False)) and 0 <= int(row.get("mode", -1)) < num_modes
    ]
    if not selected_modes:
        return None
    positive = int(selected_modes[0])
    valid_mask = torch.zeros(num_modes, device=device, dtype=torch.bool)
    valid_mask[list(sorted(mode_to_row))] = True

    costs = []
    target_effects = []
    independent = []
    raw = []
    for mode_idx in range(num_modes):
        row = mode_to_row.get(mode_idx, {})
        if mode_idx not in mode_to_row:
            costs.append(float("inf"))
            target_effects.append(float("-inf"))
            independent.append(float("inf"))
            raw.append(float("inf"))
            continue
        target = _as_float(row.get("target_effect_z"), 0.0)
        ind = _as_float(row.get("independent_collateral_z"), 0.0)
        raw_col = _as_float(row.get("raw_collateral_z"), 0.0)
        cost = _as_float(row.get("selector_cost"), float("nan"))
        if not np.isfinite(cost):
            cost = -1.2 * target + ind + 0.25 * raw_col
        costs.append(cost)
        target_effects.append(target)
        independent.append(ind)
        raw.append(raw_col)

    cost_tensor = torch.tensor(costs, device=device, dtype=torch.float32)
    cost_tensor = torch.where(valid_mask, cost_tensor, cost_tensor.new_tensor(float("inf")))
    temperature = max(float(args.soft_temperature), 1e-6)
    soft_target = torch.softmax(-cost_tensor / temperature, dim=0)
    soft_target = torch.where(valid_mask, soft_target, soft_target.new_zeros(()))
    soft_target = soft_target / soft_target.sum().clamp_min(1e-6)
    positive_tensor = torch.tensor([positive], device=device, dtype=torch.long)

    target_np = np.asarray(target_effects, dtype=np.float64)
    independent_np = np.asarray(independent, dtype=np.float64)
    raw_np = np.asarray(raw, dtype=np.float64)
    neg_mask = (
        (target_np >= float(args.target_effect_negative_floor))
        & (
            (independent_np > float(args.high_collateral_margin))
            | (raw_np > float(args.raw_collateral_margin))
        )
    )
    neg_mask[positive] = False
    negatives = torch.tensor(np.where(neg_mask)[0], device=device, dtype=torch.long)
    return positive_tensor, soft_target, negatives, valid_mask


def _policy_module(model, policy_head: str):
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise ValueError("Checkpoint has no decoder.")
    if str(policy_head) == "metric":
        policy = getattr(decoder, "counterfactual_metric_policy_head", None)
        if policy is None:
            raise ValueError("Checkpoint decoder has no counterfactual_metric_policy_head.")
        return policy
    policy = getattr(decoder, "counterfactual_policy_head", None)
    if policy is None:
        raise ValueError("Checkpoint decoder has no counterfactual_policy_head.")
    return policy


def _policy_parameters(model, policy_head: str) -> List[torch.nn.Parameter]:
    return list(_policy_module(model, policy_head).parameters())


def train_one_epoch(
    model,
    pairs: List[Tuple[Dict, Dict]],
    labels: Dict[LabelKey, List[Dict]],
    attr_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    totals = defaultdict(float)
    seen = 0
    iterator: Iterable = pairs
    if tqdm is not None:
        iterator = tqdm(pairs, desc=f"pareto policy epoch {epoch}", dynamic_ncols=True, mininterval=5.0)
    for original, counterfactual in iterator:
        key = _record_label_key(original, counterfactual)
        rows = labels.get(key)
        if not rows:
            continue
        cf_field_ids = _field_id_tensor(counterfactual, device)
        if cf_field_ids is None:
            continue
        base_traj, base_tokens, base_idx = _base_selection(model, original, attr_dim, args.target_key, device)
        cf = _forward_record(
            model,
            counterfactual,
            attr_dim,
            args.target_key,
            device,
            counterfactual_mode=True,
            counterfactual_adapter_scale=float(args.counterfactual_adapter_scale),
            counterfactual_field_ids=cf_field_ids,
            base_selected_trajectory=base_traj,
            base_mobility_tokens=base_tokens,
            base_selected_index=base_idx,
        )
        aux = cf["aux"]
        if str(args.policy_head) == "metric":
            logits = aux.get("counterfactual_metric_policy_logits")
        else:
            logits = aux.get("counterfactual_policy_logits")
        if logits is None:
            logits = cf["logits"]
        logits = logits[:, : cf["pred"].shape[1]]
        targets = _targets_from_rows(rows, int(logits.shape[1]), args, device)
        if targets is None:
            continue
        positive, soft_target, negatives, valid_mask = targets
        masked_logits = torch.where(
            valid_mask.view(1, -1),
            logits,
            logits.new_tensor(-1e9),
        )
        ce_loss = F.cross_entropy(masked_logits, positive)
        log_probs = torch.log_softmax(masked_logits[0] / max(float(args.soft_temperature), 1e-6), dim=0)
        kl_loss = F.kl_div(log_probs, soft_target, reduction="batchmean")
        pair_loss = logits.new_tensor(0.0)
        if negatives.numel() > 0:
            pos_logit = logits[0, positive[0]]
            pair_loss = F.softplus(-(pos_logit - logits[0, negatives] - float(args.pairwise_margin))).mean()
        loss = (
            float(args.ce_weight) * ce_loss
            + float(args.kl_weight) * kl_loss
            + float(args.pairwise_margin_weight) * pair_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_policy_parameters(model, str(args.policy_head)), 5.0)
        optimizer.step()

        seen += 1
        totals["loss"] += float(loss.detach().cpu().item())
        totals["ce"] += float(ce_loss.detach().cpu().item())
        totals["kl"] += float(kl_loss.detach().cpu().item())
        totals["pairwise"] += float(pair_loss.detach().cpu().item())
        totals["acc"] += float((masked_logits.detach().argmax(dim=1) == positive).float().mean().cpu().item())
        if args.log_every > 0 and seen % int(args.log_every) == 0:
            print(
                json.dumps(
                    {
                        "epoch": int(epoch),
                        "seen": int(seen),
                        "loss": totals["loss"] / seen,
                        "acc": totals["acc"] / seen,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if seen == 0:
        raise ValueError("No training pairs matched the candidate-label cache.")
    return {key: value / seen for key, value in totals.items()} | {"pairs": float(seen)}


def save_checkpoint(path: Path, checkpoint: Dict, model, args: argparse.Namespace, history: List[Dict]) -> None:
    saved_args = _saved_args_dict(checkpoint)
    if args.counterfactual_direction_residual_type:
        saved_args["counterfactual_direction_residual_type"] = str(args.counterfactual_direction_residual_type)
    saved_args["counterfactual_policy_blend_mode"] = str(args.counterfactual_policy_blend_mode)
    saved_args["counterfactual_metric_policy_weight"] = (
        float(args.counterfactual_metric_policy_weight)
        if str(args.policy_head) == "metric"
        else 0.0
    )
    saved_args["counterfactual_metric_policy_feasible_first"] = bool(
        args.counterfactual_metric_policy_feasible_first
    )
    saved_args["counterfactual_metric_policy_success_margin"] = float(
        args.counterfactual_metric_policy_success_margin
    )
    saved_args["counterfactual_metric_policy_collateral_margin"] = float(
        args.counterfactual_metric_policy_collateral_margin
    )
    saved_args["counterfactual_metric_policy_raw_collateral_margin"] = float(
        args.counterfactual_metric_policy_raw_collateral_margin
    )
    saved_args["counterfactual_policy_distill"] = {
        "source_checkpoint": str(args.checkpoint),
        "candidate_labels": str(args.candidate_labels),
        "policy_head": str(args.policy_head),
        "ce_weight": float(args.ce_weight),
        "kl_weight": float(args.kl_weight),
        "pairwise_margin_weight": float(args.pairwise_margin_weight),
        "pairwise_margin": float(args.pairwise_margin),
        "note": "Frozen-generator offline Pareto-policy distillation for Table IV selection.",
    }
    payload = {
        "epoch": int(history[-1]["epoch"]) if history else 0,
        "model_type": checkpoint.get("model_type", saved_args.get("decoder", "dit_anchor")),
        "model_state": model.state_dict(),
        "args": saved_args,
        "vocab_fields": checkpoint.get("vocab_fields", DEFAULT_MOBILITY_VOCAB.fields),
        "vocab_sizes": checkpoint.get(
            "vocab_sizes",
            {field: len(DEFAULT_MOBILITY_VOCAB.token_to_id.get(field, {})) for field in DEFAULT_MOBILITY_VOCAB.fields},
        ),
        "history": history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[pareto-policy] device={device}", flush=True)
    labels = load_candidate_labels(args.candidate_labels)
    print(f"[pareto-policy] loaded label pairs={len(labels)}", flush=True)
    transition_table = {}
    if args.transition_diagnostics_path is not None:
        transition_table = load_counterfactual_transition_table(
            args.transition_diagnostics_path,
            min_pairs=int(args.transition_min_count),
        )
    fields = {piece.strip() for piece in str(args.fields).split(",") if piece.strip()}
    records = read_records(args.input, max_records=int(args.max_records))
    pairs, pair_stats = group_counterfactual_pairs(
        records,
        fields=fields,
        max_pairs=0,
        transition_table=transition_table,
        pair_sampling=str(args.pair_sampling),
    )
    pairs = [
        (original, counterfactual)
        for original, counterfactual in pairs
        if _record_label_key(original, counterfactual) in labels
    ]
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]
    if not pairs:
        raise ValueError("No counterfactual pairs overlap with candidate labels.")
    print(
        json.dumps(
            {
                "train_pairs": len(pairs),
                "pair_stats": pair_stats,
                "input": str(args.input),
                "candidate_labels": str(args.candidate_labels),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    checkpoint = load_checkpoint(args.checkpoint, device)
    saved_args = _saved_args_dict(checkpoint)
    attr_dim = int(saved_args.get("attr_dim", 6))
    model = build_model(
        checkpoint,
        device,
        token_calibration_path=args.token_calibration_path,
        arg_overrides={
            **(
                {"counterfactual_direction_residual_type": str(args.counterfactual_direction_residual_type)}
                if args.counterfactual_direction_residual_type
                else {}
            ),
            "counterfactual_policy_blend_mode": str(args.counterfactual_policy_blend_mode),
            "counterfactual_metric_policy_weight": (
                float(args.counterfactual_metric_policy_weight)
                if str(args.policy_head) == "metric"
                else 0.0
            ),
            "counterfactual_metric_policy_success_margin": float(
                args.counterfactual_metric_policy_success_margin
            ),
            "counterfactual_metric_policy_feasible_first": bool(
                args.counterfactual_metric_policy_feasible_first
            ),
            "counterfactual_metric_policy_collateral_margin": float(
                args.counterfactual_metric_policy_collateral_margin
            ),
            "counterfactual_metric_policy_raw_collateral_margin": float(
                args.counterfactual_metric_policy_raw_collateral_margin
            ),
        },
    )
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    for param in _policy_parameters(model, str(args.policy_head)):
        param.requires_grad = True
    policy_head = getattr(model.decoder, "counterfactual_policy_head")
    if float(args.counterfactual_policy_gate_init) >= 0.0:
        gate = min(max(float(args.counterfactual_policy_gate_init), 1e-6), 1.0 - 1e-6)
        with torch.no_grad():
            policy_head.gate_logit.fill_(math.log(gate / (1.0 - gate)))
    optimizer = torch.optim.AdamW(
        _policy_parameters(model, str(args.policy_head)),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    history: List[Dict] = []
    for epoch in range(1, int(args.epochs) + 1):
        metrics = train_one_epoch(model, pairs, labels, attr_dim, args, device, optimizer, epoch)
        row = {"epoch": epoch, **metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    save_checkpoint(args.output_dir / "latest.pt", checkpoint, model, args, history)
    save_checkpoint(args.output_dir / "best.pt", checkpoint, model, args, history)
    with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "saved": str(args.output_dir / "latest.pt"),
                "epochs": int(args.epochs),
                "train_pairs": len(pairs),
                "last": history[-1] if history else {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
