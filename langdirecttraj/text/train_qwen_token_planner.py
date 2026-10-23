from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.text.qwen_instruction_encoder import (
    QwenMobilityTokenPlanner,
    QwenPlannerConfig,
    batch_encode_qwen_texts,
    load_qwen_tokenizer,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class QwenTextTokenDataset(Dataset):
    def __init__(self, jsonl_path: Path, max_samples: int = 0):
        self.jsonl_path = Path(jsonl_path)
        self.records = self.read_records(self.jsonl_path, max_samples=max_samples)

    @staticmethod
    def read_records(jsonl_path: Path, max_samples: int = 0) -> List[Dict]:
        records: List[Dict] = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "instruction" not in record or "mobility_tokens" not in record:
                    continue
                records.append(record)
                if max_samples > 0 and len(records) >= max_samples:
                    break
        if not records:
            raise ValueError(f"No text records loaded from {jsonl_path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        return {
            "instruction": str(record["instruction"]),
            "mobility_tokens": torch.tensor(record["mobility_tokens"], dtype=torch.long),
        }


def collate_qwen_text_batch(batch: List[Dict]) -> Dict:
    return {
        "instruction": [item["instruction"] for item in batch],
        "mobility_tokens": torch.stack([item["mobility_tokens"] for item in batch], dim=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen-Qwen mobility-token planner.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "qwen_token_planner")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--plan-dim", type=int, default=256)
    parser.add_argument("--num-plan-tokens", type=int, default=8)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--pooling", default="last", choices=["last", "mean"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-plan-preview", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def field_losses(logits: Dict[str, torch.Tensor], targets: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
    losses = []
    acc = {}
    for i, field in enumerate(DEFAULT_MOBILITY_VOCAB.fields):
        field_target = targets[:, i]
        field_loss = F.cross_entropy(logits[field], field_target)
        losses.append(field_loss)
        pred = logits[field].argmax(dim=-1)
        acc[f"{field}_acc"] = float((pred == field_target).float().mean().item())
    return torch.stack(losses).mean(), acc


def run_epoch(model, tokenizer, loader, optimizer, device, max_len: int, save_plan_preview: bool = False) -> Dict:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    acc_sums = {f"{field}_acc": 0.0 for field in DEFAULT_MOBILITY_VOCAB.fields}
    plan_preview = None

    for raw_batch in loader:
        encoded = batch_encode_qwen_texts(tokenizer, raw_batch["instruction"], max_len=max_len, device=device)
        targets = raw_batch["mobility_tokens"].to(device)
        with torch.set_grad_enabled(training):
            outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            loss, acc = field_losses(outputs["logits"], targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()

        if save_plan_preview and plan_preview is None:
            plan_preview = {
                "instruction": raw_batch["instruction"][0],
                "pooled_shape": list(outputs["pooled"].shape),
                "semantic_plan_tokens_shape": list(outputs["semantic_plan_tokens"].shape),
                "semantic_plan_tokens_mean": float(outputs["semantic_plan_tokens"].detach().float().mean().cpu().item()),
                "semantic_plan_tokens_std": float(outputs["semantic_plan_tokens"].detach().float().std().cpu().item()),
            }

        batch_size = int(targets.shape[0])
        total += batch_size
        total_loss += float(loss.item()) * batch_size
        for key, value in acc.items():
            acc_sums[key] += value * batch_size

    out = {"loss": total_loss / max(total, 1)}
    out.update({key: value / max(total, 1) for key, value in acc_sums.items()})
    out["mean_acc"] = float(np.mean([out[f"{field}_acc"] for field in DEFAULT_MOBILITY_VOCAB.fields]))
    if plan_preview is not None:
        out["plan_preview"] = plan_preview
    return out


def make_config(args: argparse.Namespace) -> QwenPlannerConfig:
    return QwenPlannerConfig(
        model_name=args.model_name,
        plan_dim=int(args.plan_dim),
        num_plan_tokens=int(args.num_plan_tokens),
        num_attention_heads=int(args.num_attention_heads),
        dropout=float(args.dropout),
        pooling=args.pooling,
        freeze_backbone=not args.unfreeze_backbone,
        torch_dtype=args.torch_dtype,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=not args.no_trust_remote_code,
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )


def save_checkpoint(path: Path, model, optimizer, args, config, epoch: int, train_metrics, val_metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_names = set(model.trainable_parameter_names())
    trainable_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names or not name.startswith("encoder.backbone.")
    }
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": trainable_state,
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "qwen_config": config.to_dict(),
            "mobility_fields": DEFAULT_MOBILITY_VOCAB.fields,
            "mobility_vocab_sizes": DEFAULT_MOBILITY_VOCAB.sizes,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "trainable_parameter_names": sorted(trainable_names),
            "checkpoint_format": "trainable_qwen_planner_state",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    input_path = args.input or (REPO_ROOT / "outputs" / "text_data" / f"{args.dataset}_text_to_token.jsonl")
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = QwenTextTokenDataset(input_path, max_samples=args.max_samples)
    val_size = max(1, int(round(len(dataset) * args.val_ratio))) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator) if val_size else (dataset, None)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_qwen_text_batch)
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_qwen_text_batch)
        if val_set is not None
        else None
    )

    config = make_config(args)
    tokenizer = load_qwen_tokenizer(
        config.model_name,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    model = QwenMobilityTokenPlanner(config).to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metadata = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "num_samples": len(dataset),
        "train_size": train_size,
        "val_size": val_size,
        "device": str(device),
        "model_name": config.model_name,
        "freeze_backbone": config.freeze_backbone,
        "trainable_parameters": sum(param.numel() for param in model.parameters() if param.requires_grad),
        "total_parameters": sum(param.numel() for param in model.parameters()),
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            tokenizer,
            train_loader,
            optimizer,
            device,
            max_len=args.max_len,
            save_plan_preview=args.save_plan_preview,
        )
        val_metrics = (
            run_epoch(model, tokenizer, val_loader, None, device, max_len=args.max_len, save_plan_preview=args.save_plan_preview)
            if val_loader is not None
            else dict(train_metrics)
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        save_checkpoint(output_dir / "latest.pt", model, optimizer, args, config, epoch, train_metrics, val_metrics)
        if val_metrics["mean_acc"] > best:
            best = val_metrics["mean_acc"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, args, config, epoch, train_metrics, val_metrics)

    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
