from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from langdirecttraj.data.instruction_dataset import (
    InstructionTrajectoryDataset,
    collate_instruction_batch,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.models import MobilityTokenEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test instruction trajectory dataset.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--encoder-output-dim", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = args.input or (REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_instruction_samples.jsonl")
    dataset = InstructionTrajectoryDataset(jsonl_path, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_instruction_batch,
    )
    batch = next(iter(loader))
    encoder = MobilityTokenEncoder(output_dim=args.encoder_output_dim)
    with torch.no_grad():
        token_cond = encoder(batch["mobility_tokens"])

    first = dataset[0]
    output = {
        "jsonl_path": str(jsonl_path),
        "num_loaded": len(dataset),
        "vocab_fields": DEFAULT_MOBILITY_VOCAB.fields,
        "vocab_sizes": DEFAULT_MOBILITY_VOCAB.sizes,
        "traj_shape": list(batch["traj"].shape),
        "traj_mask_shape": list(batch["traj_mask"].shape),
        "reference_shape": list(batch["reference"].shape) if "reference" in batch else None,
        "mobility_tokens_shape": list(batch["mobility_tokens"].shape),
        "mobility_token_cond_shape": list(token_cond.shape),
        "first_sample": {
            "sample_id": first["sample_id"],
            "mobility_token_ids": first["mobility_tokens"].tolist(),
            "decoded_tokens": DEFAULT_MOBILITY_VOCAB.decode(first["mobility_tokens"].tolist()),
            "instruction": first["instruction"],
            "traj_shape": list(first["traj"].shape),
            "reference_shape": list(first["reference"].shape) if "reference" in first else None,
        },
        "label_distribution": dataset.label_distribution(),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
