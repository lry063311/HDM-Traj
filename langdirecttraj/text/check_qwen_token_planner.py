from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.text.qwen_instruction_encoder import (
    QwenMobilityTokenPlanner,
    QwenPlannerConfig,
    batch_encode_qwen_texts,
    load_qwen_tokenizer,
)
from langdirecttraj.text.text_instruction_encoder import rule_override_tokens
from langdirecttraj.text.train_qwen_token_planner import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check frozen-Qwen token planner predictions.")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "outputs" / "qwen_token_planner" / "porto" / "best.pt")
    parser.add_argument(
        "--text",
        default="Generate a short city trajectory that turns right, stays close to the reference, and moves with normal progress.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
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
    config_payload = dict(checkpoint["qwen_config"])
    if args.local_files_only:
        config_payload["local_files_only"] = True
    config = QwenPlannerConfig(**config_payload)
    tokenizer = load_qwen_tokenizer(
        config.model_name,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    model = QwenMobilityTokenPlanner(config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    encoded = batch_encode_qwen_texts(tokenizer, [args.text], max_len=int(saved_args["max_len"]), device=device)
    with torch.no_grad():
        outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        raw_token_ids = torch.stack(
            [outputs["logits"][field].argmax(dim=-1) for field in DEFAULT_MOBILITY_VOCAB.fields],
            dim=1,
        )[0].cpu().tolist()
        token_ids = rule_override_tokens(args.text, raw_token_ids)
        confidences = {
            field: float(torch.softmax(outputs["logits"][field], dim=-1).max().item())
            for field in DEFAULT_MOBILITY_VOCAB.fields
        }
    output = {
        "text": args.text,
        "raw_token_ids": raw_token_ids,
        "raw_decoded_tokens": DEFAULT_MOBILITY_VOCAB.decode(raw_token_ids),
        "token_ids": token_ids,
        "decoded_tokens": DEFAULT_MOBILITY_VOCAB.decode(token_ids),
        "confidences": confidences,
        "pooled_shape": list(outputs["pooled"].shape),
        "semantic_plan_tokens_shape": list(outputs["semantic_plan_tokens"].shape),
        "checkpoint": str(args.checkpoint),
        "missing_backbone_keys": len([key for key in missing if key.startswith("encoder.backbone.")]),
        "unexpected_keys": list(unexpected),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
