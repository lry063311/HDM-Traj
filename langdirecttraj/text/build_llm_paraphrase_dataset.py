from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.text.build_gpt_paraphrase_dataset import template_paraphrases


REPO_ROOT = Path(__file__).resolve().parents[2]


SYSTEM_PROMPT = """You are a precise data augmentation assistant for trajectory-generation research.
Your job is to rewrite trajectory instructions while preserving the exact structured labels.
Do not add new requirements. Do not change direction, distance, curvature, progress, or reference adherence.
Return only a JSON array of strings."""


def build_prompt(labels: Dict[str, str], source_instruction: str, num_paraphrases: int) -> str:
    return f"""Create {num_paraphrases} concise English paraphrases for this urban trajectory instruction.

Structured labels that must be preserved:
- distance: {labels.get("distance")}
- direction: {labels.get("direction")}
- curvature: {labels.get("curvature")}
- progress: {labels.get("progress")}
- reference_adherence: {labels.get("reference_adherence")}

Source instruction:
{source_instruction}

Rules:
- Each paraphrase must explicitly preserve the same labels.
- Use natural wording, including short command-style instructions.
- Do not mention labels as JSON keys in the sentence.
- Return only a JSON array of strings."""


def extract_json_array(text: str) -> List[str]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", stripped)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item).strip() for item in payload if str(item).strip()]


class LocalHFGenerator:
    def __init__(
        self,
        model_name_or_path: str,
        device_map: str,
        torch_dtype: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[torch_dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=self.top_p if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


class OpenAICompatibleGenerator:
    def __init__(
        self,
        model: str,
        base_url: str | None,
        api_key_env: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {api_key_env}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message.content or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM paraphrase data for text-to-token planning.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--backend", choices=["local_hf", "openai_compatible"], default="local_hf")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--llm-paraphrases-per-sample", type=int, default=3)
    parser.add_argument("--template-paraphrases-per-sample", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fallback-to-template", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_generator(args: argparse.Namespace):
    if args.backend == "local_hf":
        return LocalHFGenerator(
            model_name_or_path=args.model,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    return OpenAICompatibleGenerator(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


def main() -> None:
    args = parse_args()
    input_path = args.input or (REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_instruction_samples.jsonl")
    output_path = args.output or (REPO_ROOT / "outputs" / "text_data" / f"{args.dataset}_text_to_token_llm.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = None if args.dry_run else build_generator(args)
    written = 0
    failures = 0
    import random

    rng = random.Random(args.seed)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_idx, line in enumerate(src):
            if args.max_samples > 0 and line_idx >= args.max_samples:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            labels = dict(record.get("labels", {}))
            source_instruction = str(record.get("instruction", ""))
            texts = [source_instruction]

            prompt = build_prompt(labels, source_instruction, args.llm_paraphrases_per_sample)
            if args.dry_run:
                llm_texts = []
                print(json.dumps({"prompt": prompt}, ensure_ascii=False, indent=2))
            else:
                raw = generator.generate(prompt)
                llm_texts = extract_json_array(raw)
                if not llm_texts:
                    failures += 1
                    if args.fallback_to_template:
                        llm_texts = template_paraphrases(labels, rng, args.llm_paraphrases_per_sample)
                if args.sleep > 0:
                    time.sleep(args.sleep)

            texts.extend(llm_texts[: args.llm_paraphrases_per_sample])
            if args.template_paraphrases_per_sample > 0:
                texts.extend(template_paraphrases(labels, rng, args.template_paraphrases_per_sample))

            seen = set()
            for variant_idx, text in enumerate(texts):
                text = str(text).strip()
                if not text or text.lower() in seen:
                    continue
                seen.add(text.lower())
                out = {
                    "sample_id": int(record["sample_id"]),
                    "variant_id": int(variant_idx),
                    "dataset": record.get("dataset", args.dataset),
                    "instruction": text,
                    "source_instruction": source_instruction,
                    "labels": labels,
                    "mobility_tokens": DEFAULT_MOBILITY_VOCAB.encode(labels),
                    "source": "rule" if variant_idx == 0 else ("llm" if variant_idx <= args.llm_paraphrases_per_sample else "template"),
                    "llm_model": args.model if variant_idx > 0 and variant_idx <= args.llm_paraphrases_per_sample else None,
                }
                dst.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "backend": args.backend,
        "model": args.model,
        "num_records": written,
        "failures": failures,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

