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
from langdirecttraj.text.text_instruction_encoder import (
    TextToMobilityTokenPlanner,
    TextVocab,
    batch_encode_texts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class TextTokenDataset(Dataset):
    def __init__(self, jsonl_path: Path, text_vocab: TextVocab, max_len: int, max_samples: int = 0):
        self.jsonl_path = Path(jsonl_path)
        self.text_vocab = text_vocab
        self.max_len = int(max_len)
        self.records = self._load(max_samples=max_samples)

    @staticmethod
    def read_records(jsonl_path: Path, max_samples: int = 0) -> List[Dict]:
        records: List[Dict] = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
                if max_samples > 0 and len(records) >= max_samples:
                    break
        if not records:
            raise ValueError(f"No text records loaded from {jsonl_path}")
        return records

    def _load(self, max_samples: int) -> List[Dict]:
        return self.read_records(self.jsonl_path, max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        return {
            "input_ids": torch.tensor(self.text_vocab.encode(record["instruction"], self.max_len), dtype=torch.long),
            "mobility_tokens": torch.tensor(record["mobility_tokens"], dtype=torch.long),
            "instruction": record["instruction"],
        }


def collate_text_batch(batch: List[Dict]) -> Dict:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "mobility_tokens": torch.stack([item["mobility_tokens"] for item in batch], dim=0),
        "instruction": [item["instruction"] for item in batch],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local text-to-mobility-token planner.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "text_to_token")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=20000)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
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


def build_text_vocab(jsonl_path: Path, max_samples: int, min_freq: int, max_vocab_size: int) -> TextVocab:
    records = TextTokenDataset.read_records(jsonl_path, max_samples=max_samples)
    return TextVocab.build(
        (record["instruction"] for record in records),
        min_freq=min_freq,
        max_size=max_vocab_size,
    )


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


def run_epoch(model, loader, optimizer, device) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    acc_sums = {f"{field}_acc": 0.0 for field in DEFAULT_MOBILITY_VOCAB.fields}

    for raw_batch in loader:
        input_ids = raw_batch["input_ids"].to(device)
        targets = raw_batch["mobility_tokens"].to(device)
        with torch.set_grad_enabled(training):
            logits = model(input_ids)
            loss, acc = field_losses(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = int(input_ids.shape[0])
        total += batch_size
        total_loss += float(loss.item()) * batch_size
        for key, value in acc.items():
            acc_sums[key] += value * batch_size

    out = {"loss": total_loss / max(total, 1)}
    out.update({key: value / max(total, 1) for key, value in acc_sums.items()})
    out["mean_acc"] = float(np.mean([out[f"{field}_acc"] for field in DEFAULT_MOBILITY_VOCAB.fields]))
    return out


def save_checkpoint(path: Path, model, optimizer, args, vocab: TextVocab, epoch: int, train_metrics, val_metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "text_vocab": vocab.to_dict(),
            "mobility_fields": DEFAULT_MOBILITY_VOCAB.fields,
            "mobility_vocab_sizes": DEFAULT_MOBILITY_VOCAB.sizes,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
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

    text_vocab = build_text_vocab(input_path, args.max_samples, args.min_freq, args.max_vocab_size)
    dataset = TextTokenDataset(input_path, text_vocab=text_vocab, max_len=args.max_len, max_samples=args.max_samples)
    val_size = max(1, int(round(len(dataset) * args.val_ratio))) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator) if val_size else (dataset, None)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_text_batch)
    val_loader = (
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_text_batch)
        if val_set is not None
        else None
    )

    model = TextToMobilityTokenPlanner(
        vocab_size=len(text_vocab.token_to_id),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        text_dim=args.text_dim,
        dropout=args.dropout,
        pad_id=text_vocab.pad_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metadata = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "num_samples": len(dataset),
        "vocab_size": len(text_vocab.token_to_id),
        "device": str(device),
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_metrics = run_epoch(model, val_loader, None, device) if val_loader is not None else dict(train_metrics)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        save_checkpoint(output_dir / "latest.pt", model, optimizer, args, text_vocab, epoch, train_metrics, val_metrics)
        if val_metrics["mean_acc"] > best:
            best = val_metrics["mean_acc"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, args, text_vocab, epoch, train_metrics, val_metrics)

    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

