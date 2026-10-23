from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.text.text_instruction_encoder import (
    TextToMobilityTokenPlanner,
    TextVocab,
    batch_encode_texts,
    rule_override_tokens,
)
from langdirecttraj.text.train_text_to_token_planner import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check text-to-token planner predictions.")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "outputs" / "text_to_token" / "porto" / "best.pt")
    parser.add_argument(
        "--text",
        default="Generate a short city trajectory that turns right, stays close to the reference, and moves with normal progress.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    saved_args = checkpoint["args"]
    text_vocab = TextVocab.from_dict(checkpoint["text_vocab"])
    model = TextToMobilityTokenPlanner(
        vocab_size=len(text_vocab.token_to_id),
        embed_dim=int(saved_args["embed_dim"]),
        hidden_dim=int(saved_args["hidden_dim"]),
        text_dim=int(saved_args["text_dim"]),
        dropout=0.0,
        pad_id=text_vocab.pad_id,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_ids = batch_encode_texts([args.text], text_vocab, max_len=int(saved_args["max_len"])).to(device)
    with torch.no_grad():
        raw_token_ids = model.predict_tokens(input_ids)[0].cpu().tolist()
        token_ids = rule_override_tokens(args.text, raw_token_ids)
        logits = model(input_ids)
        confidences = {
            field: float(torch.softmax(logits[field], dim=-1).max().item())
            for field in DEFAULT_MOBILITY_VOCAB.fields
        }
    output = {
        "text": args.text,
        "raw_token_ids": raw_token_ids,
        "raw_decoded_tokens": DEFAULT_MOBILITY_VOCAB.decode(raw_token_ids),
        "token_ids": token_ids,
        "decoded_tokens": DEFAULT_MOBILITY_VOCAB.decode(token_ids),
        "confidences": confidences,
        "checkpoint": str(args.checkpoint),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
