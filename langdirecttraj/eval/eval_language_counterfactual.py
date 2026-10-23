from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from langdirecttraj.data.instruction_dataset import (
    collate_instruction_batch,
    validate_official_record_protocol,
)
from langdirecttraj.eval.eval_anchor_decoder import (
    build_model,
    load_checkpoint,
    prepare_eval_batch,
    select_prediction,
    summarize,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.text.text_instruction_encoder import rule_override_tokens
from langdirecttraj.train.train_anchor_decoder import (
    build_qwen_provider,
    extract_qwen_semantic_tokens,
    make_attr_cond,
    resolve_device,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_records(path: Path, max_records: int = 0) -> List[Dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_records > 0 and len(records) >= max_records:
                break
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def tensor_record(record: Dict, target_key: str) -> Dict:
    traj = np.asarray(record.get(target_key, record.get("trajectory", [])), dtype=np.float32)
    validate_official_record_protocol(record, row_index=int(record.get("sample_id", 0)))
    if "reference" not in record:
        raise ValueError("Official OD-chord language counterfactual eval requires an explicit reference field.")
    reference = np.asarray(record["reference"], dtype=np.float32)
    labels = dict(record.get("labels", {}))
    return {
        "sample_id": int(record.get("sample_id", record.get("counterfactual_id", 0))),
        "dataset": record.get("dataset", ""),
        "traj": torch.from_numpy(traj).float(),
        "reference": torch.from_numpy(reference).float(),
        "mobility_tokens": torch.tensor(DEFAULT_MOBILITY_VOCAB.encode(labels), dtype=torch.long),
        "labels": labels,
        "instruction": str(record.get("instruction", "")),
        "metrics": dict(record.get("metrics", {})),
    }


def qwen_predict_tokens(qwen_provider, instructions: List[str], device: torch.device) -> torch.Tensor:
    tokenizer = qwen_provider["tokenizer"]
    planner = qwen_provider["planner"]
    from langdirecttraj.text.qwen_instruction_encoder import batch_encode_qwen_texts

    encoded = batch_encode_qwen_texts(tokenizer, instructions, max_len=qwen_provider["max_len"], device=device)
    with torch.no_grad():
        outputs = planner(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        raw = torch.stack(
            [outputs["logits"][field].argmax(dim=-1) for field in DEFAULT_MOBILITY_VOCAB.fields],
            dim=1,
        )
    fixed = [
        rule_override_tokens(instruction, token_ids)
        for instruction, token_ids in zip(instructions, raw.detach().cpu().tolist())
    ]
    return torch.tensor(fixed, dtype=torch.long, device=device)


def generate_for_record(model, qwen_provider, record: Dict, attr_dim: int, target_key: str, device: torch.device):
    item = tensor_record(record, target_key=target_key)
    batch = collate_instruction_batch([item])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    target, reference, route_scales = prepare_eval_batch(batch, model.decoder.target_len)
    attr_cond = make_attr_cond(batch, attr_dim, device)
    semantic = extract_qwen_semantic_tokens(qwen_provider, batch["instruction"], device)
    token_ids = qwen_predict_tokens(qwen_provider, batch["instruction"], device)
    with torch.no_grad():
        pred = select_prediction(
            model(
                reference,
                token_ids,
                attr_cond,
                semantic_plan_tokens=semantic,
                return_aux=True,
            )
        )
    return {
        "prediction": pred,
        "target": target,
        "reference": reference,
        "route_scale": route_scales,
        "token_ids": token_ids,
    }


def group_pairs(records: List[Dict], max_pairs: int, fields: List[str]) -> List[tuple[Dict, Dict]]:
    originals = {}
    counterfactuals = defaultdict(list)
    for record in records:
        cf = record.get("counterfactual", {})
        source_id = int(cf.get("source_sample_id", record.get("sample_id", -1)))
        if not cf.get("enabled"):
            originals[source_id] = record
        elif not fields or cf.get("field") in fields:
            counterfactuals[source_id].append(record)
    pairs = []
    for source_id, cfs in counterfactuals.items():
        original = originals.get(source_id)
        if original is None:
            continue
        for cf_record in cfs:
            pairs.append((original, cf_record))
            if max_pairs > 0 and len(pairs) >= max_pairs:
                return pairs
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate language-level counterfactual control.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "local_porto_od_chord_256_verify"
        / "controllability_data"
        / "porto_counterfactual_instruction_samples.filtered.jsonl",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-planner-checkpoint", type=Path, default=REPO_ROOT / "outputs" / "qwen_token_planner" / "porto" / "best.pt")
    parser.add_argument("--qwen-model-name", default=None)
    parser.add_argument("--qwen-max-len", type=int, default=96)
    parser.add_argument("--qwen-local-files-only", action="store_true")
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--fields", nargs="+", default=["direction", "curvature", "progress", "reference_adherence"])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "language_counterfactual_eval")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(checkpoint, device)
    saved_args = checkpoint.get("args", {})
    attr_dim = int(saved_args.get("attr_dim", 6))
    qwen_provider = build_qwen_provider(args.qwen_planner_checkpoint, args, device)
    records = read_records(args.input, max_records=args.max_records)
    pairs = group_pairs(records, max_pairs=args.max_pairs, fields=args.fields)
    if not pairs:
        raise ValueError("No original/counterfactual pairs found.")

    deltas = []
    cf_ade_values = []
    original_ade_values = []
    token_match = []
    by_field = defaultdict(lambda: {"delta_m": [], "cf_ade_m": [], "token_match": []})

    for original, counterfactual in pairs:
        base = generate_for_record(model, qwen_provider, original, attr_dim, args.target_key, device)
        cf = generate_for_record(model, qwen_provider, counterfactual, attr_dim, args.target_key, device)
        scale = cf["route_scale"][:, 0]
        delta = torch.linalg.norm(cf["prediction"] - base["prediction"], dim=-1).mean(dim=1) * scale
        cf_ade = torch.linalg.norm(cf["prediction"] - cf["target"], dim=-1).mean(dim=1) * scale
        base_ade = torch.linalg.norm(base["prediction"] - base["target"], dim=-1).mean(dim=1) * base["route_scale"][:, 0]

        expected = torch.tensor(
            [DEFAULT_MOBILITY_VOCAB.encode(counterfactual["labels"])],
            dtype=torch.long,
            device=device,
        )
        match = (cf["token_ids"] == expected).float().mean(dim=1)
        field = counterfactual.get("counterfactual", {}).get("field", "unknown")
        deltas.extend(delta.detach().cpu().tolist())
        cf_ade_values.extend(cf_ade.detach().cpu().tolist())
        original_ade_values.extend(base_ade.detach().cpu().tolist())
        token_match.extend(match.detach().cpu().tolist())
        by_field[field]["delta_m"].extend(delta.detach().cpu().tolist())
        by_field[field]["cf_ade_m"].extend(cf_ade.detach().cpu().tolist())
        by_field[field]["token_match"].extend(match.detach().cpu().tolist())

    metrics = {
        "num_pairs": len(pairs),
        "prediction_delta_m": summarize(deltas),
        "counterfactual_target_ade_m": summarize(cf_ade_values),
        "original_target_ade_m": summarize(original_ade_values),
        "qwen_token_match_mean": float(np.mean(token_match)),
        "by_field": {
            field: {
                "prediction_delta_m": summarize(values["delta_m"]),
                "counterfactual_target_ade_m": summarize(values["cf_ade_m"]),
                "qwen_token_match_mean": float(np.mean(values["token_match"])),
                "num_pairs": len(values["delta_m"]),
            }
            for field, values in sorted(by_field.items())
        },
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "target_key": args.target_key,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "metrics.json"
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**metrics, "metrics_path": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
