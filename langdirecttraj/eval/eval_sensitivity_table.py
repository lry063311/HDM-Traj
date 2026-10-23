from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from langdirecttraj.data.instruction_dataset import (
    InstructionTrajectoryDataset,
    collate_instruction_batch,
)
from langdirecttraj.eval.eval_anchor_decoder import (
    _arg_value,
    build_model,
    evaluate_model,
    load_checkpoint,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.train.train_anchor_decoder import resolve_device


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate token sensitivity fields in one table.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "anchor_decoder_sensitivity")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["direction", "curvature", "progress", "reference_adherence"],
        help="Mobility-token fields to evaluate.",
    )
    return parser.parse_args()


def flatten_rows(metrics_by_field: Dict[str, Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for field, metrics in metrics_by_field.items():
        sensitivity = metrics["counterfactual_sensitivity_m"]
        changed_only = metrics["counterfactual_sensitivity_changed_only_m"]
        for token_name, stats in sensitivity.items():
            changed_stats = changed_only.get(token_name, {})
            rows.append(
                {
                    "field": field,
                    "token": token_name,
                    "all_mean_m": stats.get("mean"),
                    "all_p50_m": stats.get("p50"),
                    "all_p90_m": stats.get("p90"),
                    "changed_mean_m": changed_stats.get("mean"),
                    "changed_p50_m": changed_stats.get("p50"),
                    "changed_p90_m": changed_stats.get("p90"),
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    for field in args.fields:
        if field not in DEFAULT_MOBILITY_VOCAB.fields:
            raise ValueError(f"Unknown mobility-token field: {field}")

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint, device)
    saved_args = checkpoint.get("args", {})
    attr_dim = int(_arg_value(saved_args, "attr_dim", 6))

    jsonl_path = args.input or (REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_instruction_samples.jsonl")
    dataset = InstructionTrajectoryDataset(jsonl_path, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_instruction_batch,
    )

    metrics_by_field = {}
    for field in args.fields:
        metrics, _, _ = evaluate_model(
            model=model,
            loader=loader,
            device=device,
            attr_dim=attr_dim,
            sensitivity_field=field,
        )
        metrics_by_field[field] = metrics

    rows = flatten_rows(metrics_by_field)
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sensitivity_table.json"
    csv_path = output_dir / "sensitivity_table.csv"
    summary = {
        "dataset": args.dataset,
        "jsonl_path": str(jsonl_path),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "model_type": str(checkpoint.get("model_type", "unknown")),
        "num_evaluated": len(dataset),
        "fields": args.fields,
        "metrics_by_field": metrics_by_field,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    print(json.dumps({"json_path": str(json_path), "csv_path": str(csv_path), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
