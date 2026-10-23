from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from langdirecttraj.data.instruction_dataset import InstructionTrajectoryDataset, collate_instruction_batch
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.models.direction_geometry_classifier import DirectionGeometryClassifier
from langdirecttraj.train.train_anchor_decoder import prepare_training_batch_with_scale, resolve_device, set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local-frame trajectory -> direction token classifier.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--target-key", choices=["trajectory", "pseudo_trajectory"], default="pseudo_trajectory")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "direction_geometry_classifier")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--target-len", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--loss-type", choices=["ce", "focal", "balanced_ce", "balanced_focal"], default="balanced_focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--sampler",
        choices=["none", "balanced"],
        default="none",
        help="Use a class-balanced training sampler. This is safer than very large CE weights on skewed direction labels.",
    )
    return parser.parse_args()


def direction_targets(batch: Dict, device: torch.device) -> torch.Tensor:
    field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("direction")
    return batch["mobility_tokens"][:, field_idx].to(device=device, dtype=torch.long)


def class_balanced_weights(labels: Iterable[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    for item in labels:
        counts[int(item)] += 1.0
    valid = counts > 0
    weights = torch.zeros_like(counts)
    weights[valid] = 1.0 / counts[valid].clamp_min(1.0)
    if bool(valid.any()):
        weights[valid] = weights[valid] / weights[valid].mean().clamp_min(1e-8)
    return weights


def loss_fn(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    class_weights: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    weight = class_weights.to(device=logits.device, dtype=logits.dtype) if class_weights is not None else None
    if loss_type in {"ce", "balanced_ce"}:
        return F.cross_entropy(logits, target, weight=weight)
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt).pow(float(gamma)) * ce).mean()


def label_counts(labels: Iterable[int], num_classes: int) -> List[int]:
    counts = [0 for _ in range(num_classes)]
    for item in labels:
        counts[int(item)] += 1
    return counts


def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth)][int(pred)] += 1
    return matrix


def macro_f1(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    f1_values = []
    out: Dict[str, float] = {}
    for idx in range(num_classes):
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == idx and pred == idx)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != idx and pred == idx)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == idx and pred != idx)
        support = sum(1 for truth in y_true if truth == idx)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support > 0:
            f1_values.append(f1)
        out[f"f1_{idx}"] = float(f1)
        out[f"support_{idx}"] = float(support)
    out["macro_f1"] = float(np.mean(f1_values)) if f1_values else 0.0
    out["accuracy"] = float(np.mean([truth == pred for truth, pred in zip(y_true, y_pred)])) if y_true else 0.0
    out["true_counts"] = label_counts(y_true, num_classes)
    out["pred_counts"] = label_counts(y_pred, num_classes)
    out["confusion"] = confusion_matrix(y_true, y_pred, num_classes)
    return out


def run_epoch(
    model: DirectionGeometryClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    target_len: int,
    loss_type: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    all_true: List[int] = []
    all_pred: List[int] = []
    for raw_batch in loader:
        batch = dict(raw_batch)
        for key in ["traj", "traj_mask", "reference", "reference_mask", "mobility_tokens"]:
            if key in batch:
                batch[key] = batch[key].to(device)
        points, _, _, _ = prepare_training_batch_with_scale(batch, target_len)
        target = direction_targets(batch, device)
        with torch.set_grad_enabled(training):
            logits = model(points)
            loss = loss_fn(logits, target, loss_type, class_weights, focal_gamma)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        batch_size = int(target.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_count += batch_size
        all_true.extend(int(item) for item in target.detach().cpu().tolist())
        all_pred.extend(int(item) for item in logits.detach().argmax(dim=1).cpu().tolist())
    metrics = macro_f1(all_true, all_pred, len(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]))
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def save_checkpoint(
    path: Path,
    model: DirectionGeometryClassifier,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": vars(args),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "direction_vocab": DEFAULT_MOBILITY_VOCAB.token_to_id["direction"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    jsonl_path = args.input or (REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_counterfactual_instruction_samples.filtered.jsonl")
    dataset = InstructionTrajectoryDataset(jsonl_path, max_samples=args.max_samples, target_key=args.target_key)
    val_size = max(1, int(round(len(dataset) * args.val_ratio))) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Training split is empty.")
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator) if val_size else (dataset, None)
    num_classes = len(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"])
    train_labels = []
    for record_idx in train_set.indices if hasattr(train_set, "indices") else range(len(train_set)):
        record = dataset.records[int(record_idx)]
        train_labels.append(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"][record["labels"]["direction"]])
    weights = class_balanced_weights(train_labels, num_classes, device) if args.loss_type.startswith("balanced") else None
    sampler = None
    shuffle_train = True
    if args.sampler == "balanced":
        counts = torch.tensor(label_counts(train_labels, num_classes), dtype=torch.float32)
        sample_weights = torch.tensor(
            [1.0 / max(float(counts[int(label)].item()), 1.0) for label in train_labels],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        shuffle_train = False

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_instruction_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_instruction_batch,
    ) if val_set else None

    model = DirectionGeometryClassifier(num_classes=num_classes, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset": args.dataset,
        "input": str(jsonl_path),
        "num_samples": len(dataset),
        "train_size": train_size,
        "val_size": val_size,
        "device": str(device),
        "target_key": args.target_key,
        "direction_vocab": DEFAULT_MOBILITY_VOCAB.token_to_id["direction"],
        "sampler": args.sampler,
        "train_label_counts": label_counts(train_labels, num_classes),
        "class_weights": weights.detach().cpu().tolist() if weights is not None else None,
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    best_score = -math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.target_len, args.loss_type, weights, args.focal_gamma)
        val_metrics = run_epoch(model, val_loader, None, device, args.target_len, args.loss_type, weights, args.focal_gamma) if val_loader else dict(train_metrics)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        save_checkpoint(output_dir / "latest.pt", model, optimizer, args, epoch, train_metrics, val_metrics)
        score = float(val_metrics["macro_f1"])
        if score > best_score:
            best_score = score
            save_checkpoint(output_dir / "best.pt", model, optimizer, args, epoch, train_metrics, val_metrics)
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
