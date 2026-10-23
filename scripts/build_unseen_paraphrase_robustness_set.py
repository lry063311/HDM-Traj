#!/usr/bin/env python
"""Build a controlled, rule-uncovered paraphrase set for planner evaluation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB  # noqa: E402
from langdirecttraj.text.text_instruction_encoder import rule_token_overrides  # noqa: E402


PHRASES: Dict[str, Dict[str, List[str]]] = {
    "distance": {
        "very_short": [
            "minimize the overall distance",
            "cover only a tiny distance",
            "use the smallest available travel extent",
            "choose the minimum-distance option",
        ],
        "short": [
            "favor fewer kilometers",
            "reduce the overall travel distance",
            "use a distance-efficient path",
            "keep the travel extent below the usual option",
        ],
        "medium": [
            "use an intermediate travel extent",
            "avoid both minimal and excessive distance",
            "keep the travel distance near the middle of the available options",
            "use an ordinary amount of distance",
        ],
        "long": [
            "allow substantially more travel distance",
            "cover many more kilometers",
            "use a large overall travel extent",
            "choose a distance-intensive option",
        ],
    },
    "direction": {
        "straight": [
            "maintain nearly the same heading",
            "avoid pronounced changes in heading",
            "keep the net directional change minimal",
            "use little lateral change in orientation",
        ],
        "left_turn": [
            "favor a counterclockwise change in heading",
            "arc toward the left-hand side",
            "shift the heading counterclockwise along the way",
            "place the main directional change on the left-hand side",
        ],
        "right_turn": [
            "favor a clockwise change in heading",
            "arc toward the right-hand side",
            "shift the heading clockwise along the way",
            "place the main directional change on the right-hand side",
        ],
        "u_turn": [
            "fold back toward the origin",
            "include a pronounced reversal in orientation",
            "use a hairpin-like change toward the origin",
            "swing the heading around by roughly half a circle",
        ],
        "multi_turn": [
            "change heading repeatedly",
            "include distinct changes in travel orientation",
            "alternate the heading more than once",
            "use recurring directional changes",
        ],
    },
    "curvature": {
        "smooth": [
            "keep changes in heading gradual and low-frequency",
            "use gentle and slowly varying geometry",
            "avoid abrupt oscillations in the path shape",
            "maintain a regular arc with limited bending",
        ],
        "moderate": [
            "use an intermediate level of geometric bending",
            "allow noticeable but not excessive changes in heading",
            "keep the shape between gentle and highly irregular",
            "use a balanced amount of geometric bending",
        ],
        "winding": [
            "include frequent and pronounced changes of heading",
            "use geometry with repeated alternating bends",
            "meander through many changes in orientation",
            "create a highly sinuous shape with recurring bends",
        ],
    },
    "progress": {
        "slow": [
            "advance conservatively along the endpoint displacement",
            "make only gradual advancement toward the destination",
            "reduce the rate at which the path closes the OD gap",
            "use restrained advancement toward the destination",
        ],
        "normal": [
            "maintain an ordinary rate of advancement",
            "use a middle level of advancement along the OD displacement",
            "keep advancement neither restrained nor accelerated",
            "proceed at a conventional rate toward the endpoint",
        ],
        "fast": [
            "advance assertively toward the destination",
            "close the OD gap at an accelerated rate",
            "favor rapid advancement along the endpoint displacement",
            "move decisively toward the destination",
        ],
    },
    "reference_adherence": {
        "tight": [
            "keep the path near the direct endpoint alignment",
            "remain within a narrow corridor around the endpoint alignment",
            "limit separation from the straight endpoint-to-endpoint alignment",
            "track the direct OD alignment with little lateral offset",
        ],
        "moderate": [
            "permit limited departure from the direct endpoint alignment",
            "allow some bounded separation from the OD alignment",
            "use a modest lateral offset from the endpoint connection",
            "use an intermediate amount of separation from the endpoint alignment",
        ],
        "loose": [
            "permit substantial departure from the direct endpoint alignment",
            "allow a wide lateral offset from the OD alignment",
            "leave broad freedom away from the endpoint connection",
            "use a large separation from the direct OD alignment",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--candidate-pool", type=int, default=4096)
    return parser.parse_args()


def iter_records(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            labels = record.get("labels")
            if not isinstance(labels, dict):
                raise KeyError(f"Missing labels at {path}:{line_number}")
            normalized = {field: str(labels[field]) for field in DEFAULT_MOBILITY_VOCAB.fields}
            if normalized["reference_adherence"] == "unknown":
                continue
            if any(normalized[field] not in PHRASES[field] for field in DEFAULT_MOBILITY_VOCAB.fields):
                continue
            yield {
                "sample_id": int(record.get("sample_id", line_number - 1)),
                "instruction": str(record["instruction"]),
                "labels": normalized,
            }


def quotas(num_samples: int) -> Dict[str, Dict[str, int]]:
    output: Dict[str, Dict[str, int]] = {}
    for field in DEFAULT_MOBILITY_VOCAB.fields:
        values = list(PHRASES[field])
        base, remainder = divmod(num_samples, len(values))
        output[field] = {
            value: base + int(index < remainder)
            for index, value in enumerate(values)
        }
    return output


def select_records(records: List[Dict], num_samples: int, seed: int, candidate_pool: int) -> List[Dict]:
    if len(records) < num_samples:
        raise ValueError(f"Requested {num_samples} samples but only {len(records)} are available")
    rng = random.Random(seed)
    targets = quotas(num_samples)
    counts = {field: Counter() for field in DEFAULT_MOBILITY_VOCAB.fields}
    available = list(range(len(records)))
    selected: List[Dict] = []

    for _ in range(num_samples):
        pool_size = min(int(candidate_pool), len(available))
        candidates = rng.sample(available, pool_size)

        def score(index: int) -> float:
            labels = records[index]["labels"]
            value = 0.0
            for field in DEFAULT_MOBILITY_VOCAB.fields:
                label = labels[field]
                deficit = targets[field][label] - counts[field][label]
                if deficit > 0:
                    value += deficit / max(1, targets[field][label])
            return value

        best_score = max(score(index) for index in candidates)
        best = [index for index in candidates if score(index) == best_score]
        chosen = rng.choice(best)
        available.remove(chosen)
        record = records[chosen]
        selected.append(record)
        for field, label in record["labels"].items():
            counts[field][label] += 1
    return selected


def paraphrase(record: Dict, index: int, seed: int) -> str:
    rng = random.Random(seed * 1_000_003 + int(record["sample_id"]) * 97 + index)
    clauses = []
    for field in DEFAULT_MOBILITY_VOCAB.fields:
        variants = PHRASES[field][record["labels"][field]]
        clauses.append(variants[rng.randrange(len(variants))])
    if index % 2:
        clauses = [clauses[1], clauses[0], clauses[4], clauses[2], clauses[3]]
    openings = [
        "Plan this trajectory to",
        "Construct a route that will",
        "For this request, create a trajectory that will",
        "Generate the journey so that it will",
    ]
    return f"{openings[index % len(openings)]} {', '.join(clauses[:-1])}, and {clauses[-1]}."


def main() -> None:
    args = parse_args()
    records = list(iter_records(args.input))
    selected = select_records(
        records,
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        candidate_pool=int(args.candidate_pool),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    label_counts = {field: Counter() for field in DEFAULT_MOBILITY_VOCAB.fields}
    zero_rule_count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(selected):
            text = paraphrase(record, index=index, seed=int(args.seed))
            overrides = rule_token_overrides(text)
            if not overrides:
                zero_rule_count += 1
            for field, label in record["labels"].items():
                label_counts[field][label] += 1
            output = {
                "sample_id": int(record["sample_id"]),
                "dataset": "porto",
                "instruction": text,
                "source_instruction": record["instruction"],
                "labels": record["labels"],
                "mobility_tokens": DEFAULT_MOBILITY_VOCAB.encode(record["labels"]),
                "source": "controlled_unseen_paraphrase",
                "rule_overrides": {
                    field: DEFAULT_MOBILITY_VOCAB.id_to_token[field][token_id]
                    for field, token_id in overrides.items()
                },
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    summary = {
        "protocol": "controlled_rule_uncovered_paraphrases_v2_concise",
        "source_input": str(args.input),
        "output": str(args.output),
        "seed": int(args.seed),
        "num_samples": len(selected),
        "zero_rule_trigger_examples": int(zero_rule_count),
        "zero_rule_trigger_rate": float(zero_rule_count / len(selected)),
        "all_examples_rule_uncovered": zero_rule_count == len(selected),
        "label_counts": {field: dict(counts) for field, counts in label_counts.items()},
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if zero_rule_count != len(selected):
        raise RuntimeError("At least one paraphrase matched a deterministic rule trigger")


if __name__ == "__main__":
    main()
