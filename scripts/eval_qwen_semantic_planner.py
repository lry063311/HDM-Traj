#!/usr/bin/env python
"""Evaluate Qwen semantic planning for the paper Table II."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langdirecttraj.eval.instruction_geometry import macro_f1  # noqa: E402
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB  # noqa: E402
from langdirecttraj.text.qwen_instruction_encoder import (  # noqa: E402
    QwenMobilityTokenPlanner,
    QwenPlannerConfig,
    batch_encode_qwen_texts,
    load_qwen_tokenizer,
)
from langdirecttraj.text.text_instruction_encoder import rule_override_tokens  # noqa: E402


def _install_pathlib_pickle_compat() -> None:
    module = sys.modules.get("pathlib._local")
    if module is None:
        module = types.ModuleType("pathlib._local")
        sys.modules["pathlib._local"] = module
    module.Path = pathlib.Path
    if sys.platform == "win32":
        module.PosixPath = pathlib.WindowsPath
        module.WindowsPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath
        module.PosixPath = pathlib.PosixPath
        module.WindowsPath = pathlib.PosixPath


_install_pathlib_pickle_compat()


class PlannerEvalDataset(Dataset):
    def __init__(self, path: Path, max_samples: int = 0):
        self.records = self._read(path, int(max_samples))
        if not self.records:
            raise ValueError(f"No planner eval records loaded from {path}")

    @staticmethod
    def _target_tokens(record: Dict) -> List[int]:
        if "mobility_tokens" in record:
            return [int(x) for x in record["mobility_tokens"]]
        labels = record.get("labels")
        if isinstance(labels, dict):
            return DEFAULT_MOBILITY_VOCAB.encode(labels)
        raise KeyError("Record must contain mobility_tokens or labels.")

    @classmethod
    def _read(cls, path: Path, max_samples: int) -> List[Dict]:
        records: List[Dict] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "instruction" not in record:
                    continue
                records.append(
                    {
                        "sample_id": int(record.get("sample_id", len(records))),
                        "instruction": str(record["instruction"]),
                        "mobility_tokens": cls._target_tokens(record),
                    }
                )
                if max_samples > 0 and len(records) >= max_samples:
                    break
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]


def collate(batch: List[Dict]) -> Dict:
    return {
        "sample_id": [int(item["sample_id"]) for item in batch],
        "instruction": [str(item["instruction"]) for item in batch],
        "mobility_tokens": torch.tensor([item["mobility_tokens"] for item in batch], dtype=torch.long),
    }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model-name", default="")
    parser.add_argument("--max-len", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-rule-override", action="store_true")
    parser.add_argument("--include-empty-labels", action="store_true")
    return parser.parse_args()


def normalize_model_name(saved_name: str, override: str = "") -> str:
    if override:
        return str(override)
    name = str(saved_name)
    looks_like_windows_path = ":\\" in name or "\\" in name
    if not looks_like_windows_path:
        return name
    basename = name.replace("\\", "/").rstrip("/").split("/")[-1]
    candidates = [
        Path("models") / basename,
        REPO_ROOT / "models" / basename,
        Path("/root/LangDirectTraj/models") / basename,
        Path("/root/models") / basename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if basename:
        return f"Qwen/{basename}"
    return name


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    saved_args = checkpoint.get("args", {})
    config_payload = dict(checkpoint["qwen_config"])
    config_payload["model_name"] = normalize_model_name(config_payload.get("model_name", ""), args.model_name)
    if args.local_files_only:
        config_payload["local_files_only"] = True
    config = QwenPlannerConfig(**config_payload)
    max_len = int(args.max_len or saved_args.get("max_len", 128))

    tokenizer = load_qwen_tokenizer(
        config.model_name,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )
    model = QwenMobilityTokenPlanner(config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    dataset = PlannerEvalDataset(args.input, max_samples=int(args.max_samples))
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate)
    y_true: Dict[str, List[str]] = {field: [] for field in DEFAULT_MOBILITY_VOCAB.fields}
    y_pred: Dict[str, List[str]] = {field: [] for field in DEFAULT_MOBILITY_VOCAB.fields}
    raw_y_pred: Dict[str, List[str]] = {field: [] for field in DEFAULT_MOBILITY_VOCAB.fields}
    confidences: Dict[str, List[float]] = {field: [] for field in DEFAULT_MOBILITY_VOCAB.fields}
    exact = []
    raw_exact = []
    preview = []

    with torch.no_grad():
        for batch in loader:
            encoded = batch_encode_qwen_texts(tokenizer, batch["instruction"], max_len=max_len, device=device)
            outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            raw_ids = torch.stack(
                [outputs["logits"][field].argmax(dim=-1) for field in DEFAULT_MOBILITY_VOCAB.fields],
                dim=1,
            ).cpu().numpy()
            targets = batch["mobility_tokens"].numpy()
            probs = {
                field: torch.softmax(outputs["logits"][field], dim=-1).max(dim=-1).values.detach().cpu().numpy()
                for field in DEFAULT_MOBILITY_VOCAB.fields
            }
            pred_ids = []
            for instruction, row in zip(batch["instruction"], raw_ids.tolist()):
                if bool(args.no_rule_override):
                    pred_ids.append(row)
                else:
                    pred_ids.append(rule_override_tokens(instruction, row))
            pred_ids = np.asarray(pred_ids, dtype=np.int64)
            exact.extend((pred_ids == targets).all(axis=1).astype(bool).tolist())
            raw_exact.extend((raw_ids == targets).all(axis=1).astype(bool).tolist())

            for row_idx in range(targets.shape[0]):
                true_labels = DEFAULT_MOBILITY_VOCAB.decode(targets[row_idx].tolist())
                pred_labels = DEFAULT_MOBILITY_VOCAB.decode(pred_ids[row_idx].tolist())
                raw_pred_labels = DEFAULT_MOBILITY_VOCAB.decode(raw_ids[row_idx].tolist())
                if len(preview) < 8:
                    preview.append(
                        {
                            "sample_id": int(batch["sample_id"][row_idx]),
                            "instruction": batch["instruction"][row_idx],
                            "target": true_labels,
                            "prediction": pred_labels,
                            "raw_prediction": raw_pred_labels,
                        }
                    )
                for field in DEFAULT_MOBILITY_VOCAB.fields:
                    y_true[field].append(true_labels[field])
                    y_pred[field].append(pred_labels[field])
                    raw_y_pred[field].append(raw_pred_labels[field])
                    confidences[field].append(float(probs[field][row_idx]))

    field_metrics = {}
    raw_field_metrics = {}
    table2 = {}
    for field in DEFAULT_MOBILITY_VOCAB.fields:
        labels = DEFAULT_MOBILITY_VOCAB.token_to_id[field].keys()
        metric = macro_f1(y_true[field], y_pred[field], labels, include_empty=bool(args.include_empty_labels))
        raw_metric = macro_f1(y_true[field], raw_y_pred[field], labels, include_empty=bool(args.include_empty_labels))
        metric["mean_confidence"] = float(np.mean(confidences[field])) if confidences[field] else float("nan")
        field_metrics[field] = metric
        raw_field_metrics[field] = raw_metric
        table2[f"{field}_accuracy"] = float(metric["accuracy"])
        table2[f"{field}_f1"] = float(metric["macro_f1"])

    table2["mean_accuracy"] = float(np.mean([field_metrics[field]["accuracy"] for field in DEFAULT_MOBILITY_VOCAB.fields]))
    table2["mean_macro_f1"] = float(np.mean([field_metrics[field]["macro_f1"] for field in DEFAULT_MOBILITY_VOCAB.fields]))
    table2["exact_match"] = float(np.mean(exact)) if exact else float("nan")
    table2["raw_exact_match"] = float(np.mean(raw_exact)) if raw_exact else float("nan")
    table2["num_samples"] = int(len(dataset))

    out = {
        "dataset": str(args.dataset),
        "num_samples": int(len(dataset)),
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "model_name": str(config.model_name),
        "max_len": int(max_len),
        "rule_override": not bool(args.no_rule_override),
        "table2_semantic_planning": table2,
        "field_metrics": field_metrics,
        "raw_field_metrics": raw_field_metrics,
        "preview": preview,
        "missing_backbone_keys": len([key for key in missing if key.startswith("encoder.backbone.")]),
        "unexpected_keys": list(unexpected),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
