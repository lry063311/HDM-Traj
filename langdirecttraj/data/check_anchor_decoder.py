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
from langdirecttraj.models import AnchorResidualDecoder, ConditionFusion, MobilityTokenEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test minimal anchor residual decoder.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--attr-dim", type=int, default=6)
    parser.add_argument("--cond-dim", type=int, default=128)
    parser.add_argument("--target-len", type=int, default=200)
    parser.add_argument("--fusion-mode", choices=["add", "concat_mlp", "gated"], default="gated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = args.input or (REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_instruction_samples.jsonl")
    dataset = InstructionTrajectoryDataset(jsonl_path, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_instruction_batch)
    batch = next(iter(loader))

    token_encoder = MobilityTokenEncoder(output_dim=args.cond_dim)
    fusion = ConditionFusion(
        attr_dim=args.attr_dim,
        mobility_dim=args.cond_dim,
        output_dim=args.cond_dim,
        mode=args.fusion_mode,
    )
    decoder = AnchorResidualDecoder(cond_dim=args.cond_dim, target_len=args.target_len)

    attr_cond = torch.zeros(batch["mobility_tokens"].shape[0], args.attr_dim, dtype=torch.float32)
    with torch.no_grad():
        mobility_cond = token_encoder(batch["mobility_tokens"])
        fused_cond = fusion(attr_cond, mobility_cond)
        trajectory, aux = decoder(batch["reference"], fused_cond, return_aux=True)

    output = {
        "jsonl_path": str(jsonl_path),
        "reference_shape": list(batch["reference"].shape),
        "fused_cond_shape": list(fused_cond.shape),
        "trajectory_shape": list(trajectory.shape),
        "anchor_shape": list(aux["anchor"].shape),
        "gate_mean": float(aux["gate"].mean().item()),
        "residual_abs_mean": float(aux["residual"].abs().mean().item()),
        "trajectory_min": float(trajectory.min().item()),
        "trajectory_max": float(trajectory.max().item()),
        "sample_tokens": batch["mobility_tokens"][0].tolist(),
        "sample_instruction": batch["instruction"][0],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
