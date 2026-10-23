from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from langdirecttraj.data.instruction_dataset import InstructionTrajectoryDataset, collate_instruction_batch
from langdirecttraj.eval.eval_anchor_decoder import build_model, load_checkpoint, select_prediction
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.text.text_instruction_encoder import (
    TextToMobilityTokenPlanner,
    TextVocab,
    batch_encode_texts,
    rule_override_tokens,
)
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    prepare_training_batch,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def arg_value(saved_args: Dict, key: str, default):
    return saved_args.get(key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one trajectory from natural-language instruction text.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--text-checkpoint", type=Path, default=REPO_ROOT / "outputs" / "text_to_token" / "porto" / "best.pt")
    parser.add_argument("--traj-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=None)
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "text_generation")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_text_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_text_planner(checkpoint: Dict, device: torch.device):
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
    return model, text_vocab, int(saved_args["max_len"])


def predict_mobility_tokens(text: str, model, text_vocab: TextVocab, max_len: int, device: torch.device):
    input_ids = batch_encode_texts([text], text_vocab, max_len=max_len).to(device)
    with torch.no_grad():
        logits = model(input_ids)
        raw_token_ids = model.predict_tokens(input_ids)
        confidences = {
            field: float(torch.softmax(logits[field], dim=-1).max().item())
            for field in DEFAULT_MOBILITY_VOCAB.fields
        }
    token_ids = torch.tensor(
        [rule_override_tokens(text, raw_token_ids[0].detach().cpu().tolist())],
        dtype=torch.long,
        device=device,
    )
    return token_ids, raw_token_ids, confidences


def to_batch(sample: Dict) -> Dict:
    return collate_instruction_batch([sample])


def plot_generation(record: Dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    scale = float(record["route_scale"])
    target = np.asarray(record["target"], dtype=np.float32) * scale
    reference = np.asarray(record["reference"], dtype=np.float32) * scale
    prediction = np.asarray(record["prediction"], dtype=np.float32) * scale

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(reference[:, 0], reference[:, 1], color="#8a8a8a", linewidth=1.7, label="reference")
    ax.plot(target[:, 0], target[:, 1], color="#2474b5", linewidth=1.8, label="target")
    ax.plot(prediction[:, 0], prediction[:, 1], color="#d64b3c", linewidth=1.9, label="prediction")
    ax.scatter(target[0, 0], target[0, 1], color="#1b9e77", s=24, label="start")
    ax.scatter(target[-1, 0], target[-1, 1], color="#222222", s=24, label="end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.set_title("Text-conditioned trajectory generation", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    jsonl_path = args.input or (REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_instruction_samples.jsonl")

    text_checkpoint = load_text_checkpoint(args.text_checkpoint, device)
    text_model, text_vocab, max_len = build_text_planner(text_checkpoint, device)
    pred_tokens, raw_pred_tokens, confidences = predict_mobility_tokens(args.text, text_model, text_vocab, max_len, device)

    traj_checkpoint = load_checkpoint(args.traj_checkpoint, device)
    traj_model = build_model(traj_checkpoint, device)
    attr_dim = int(arg_value(traj_checkpoint.get("args", {}), "attr_dim", 6))
    saved_traj_args = traj_checkpoint.get("args", {})
    if args.qwen_planner_checkpoint is None:
        saved_qwen_checkpoint = arg_value(saved_traj_args, "qwen_planner_checkpoint", None)
        if saved_qwen_checkpoint:
            args.qwen_planner_checkpoint = Path(saved_qwen_checkpoint)
    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device)

    dataset = InstructionTrajectoryDataset(jsonl_path)
    sample = dataset[int(args.sample_index)]
    batch = to_batch(sample)
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    target, reference, target_mask = prepare_training_batch(batch, traj_model.decoder.target_len)
    attr_cond = make_attr_cond(batch, attr_dim, device)
    semantic_plan_tokens = extract_qwen_semantic_tokens(qwen_provider, [args.text], device)

    with torch.no_grad():
        prediction = select_prediction(
            traj_model(
                reference,
                pred_tokens.to(device),
                attr_cond,
                semantic_plan_tokens=semantic_plan_tokens,
                return_aux=True,
            )
        )

    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_id = int(sample["sample_id"])
    json_path = output_dir / f"text_generation_sample_{sample_id}.json"
    plot_path = output_dir / f"text_generation_sample_{sample_id}.png"

    result = {
        "dataset": args.dataset,
        "sample_id": sample_id,
        "sample_index": int(args.sample_index),
        "input_text": args.text,
        "raw_predicted_token_ids": raw_pred_tokens[0].detach().cpu().tolist(),
        "raw_predicted_tokens": DEFAULT_MOBILITY_VOCAB.decode(raw_pred_tokens[0].detach().cpu().tolist()),
        "predicted_token_ids": pred_tokens[0].detach().cpu().tolist(),
        "predicted_tokens": DEFAULT_MOBILITY_VOCAB.decode(pred_tokens[0].detach().cpu().tolist()),
        "planner_confidences": confidences,
        "dataset_instruction": sample["instruction"],
        "dataset_token_ids": sample["mobility_tokens"].tolist(),
        "dataset_tokens": DEFAULT_MOBILITY_VOCAB.decode(sample["mobility_tokens"].tolist()),
        "target": target[0].detach().cpu().tolist(),
        "reference": reference[0].detach().cpu().tolist(),
        "prediction": prediction[0].detach().cpu().tolist(),
        "route_scale": float(torch.linalg.norm((target[0, 1:] - target[0, :-1]), dim=-1).sum().item()),
        "text_checkpoint": str(args.text_checkpoint),
        "traj_checkpoint": str(args.traj_checkpoint),
        "qwen_planner_checkpoint": str(args.qwen_planner_checkpoint) if args.qwen_planner_checkpoint else None,
        "qwen_semantic_tokens_shape": list(semantic_plan_tokens.shape) if semantic_plan_tokens is not None else None,
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_generation(result, plot_path)

    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "plot_path": str(plot_path),
                "predicted_tokens": result["predicted_tokens"],
                "planner_confidences": confidences,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
