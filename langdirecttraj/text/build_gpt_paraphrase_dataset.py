from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from langdirecttraj.instructions.templates import (
    CURVATURE_TEXT,
    DIRECTION_TEXT,
    DISTANCE_TEXT,
    PROGRESS_TEXT,
    REFERENCE_TEXT,
)
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB


REPO_ROOT = Path(__file__).resolve().parents[2]


DISTANCE_ALT = {
    "very_short": ["very short", "brief", "nearby"],
    "short": ["short", "compact", "local"],
    "medium": ["medium-length", "moderate-distance", "mid-range"],
    "long": ["long", "extended", "long-distance"],
}

DIRECTION_ALT = {
    "straight": [
        "mostly straight",
        "go straight",
        "continue straight",
        "with little turning",
        "without a major turn",
        "following a straight trend",
    ],
    "left_turn": [
        "turn left",
        "turns left",
        "make a left turn",
        "makes a left turn",
        "with a left-turn pattern",
        "turning left overall",
        "bending to the left",
        "curving left rather than right",
    ],
    "right_turn": [
        "turn right",
        "turns right",
        "make a right turn",
        "makes a right turn",
        "with a right-turn pattern",
        "turning right overall",
        "bending to the right",
        "curving right rather than left",
    ],
    "u_turn": [
        "make a U-turn",
        "makes a U-turn",
        "make a U-turn-like movement",
        "with a U-turn-like shape",
        "double back",
        "doubles back",
        "reverse direction",
        "reverses direction",
        "return toward the starting direction",
    ],
    "multi_turn": [
        "make multiple turns",
        "take several turns",
        "through several turns",
        "with a multi-turn route shape",
        "follow a zigzag-like multi-turn path",
        "include more than one major turn",
    ],
}

DIRECTION_COMMANDS = {
    "straight": [
        "Go straight.",
        "Generate a mostly straight route.",
        "Create a route with no major turn.",
        "Keep the trajectory almost straight.",
    ],
    "left_turn": [
        "Turn left.",
        "Generate a left-turn route.",
        "Create a trajectory that makes a left turn.",
        "The route should bend left.",
    ],
    "right_turn": [
        "Turn right.",
        "Turns right.",
        "Generate a right-turn route.",
        "Create a trajectory that makes a right turn.",
        "The route should bend right.",
        "Generate a route that turns right.",
        "Create a route turning right.",
    ],
    "u_turn": [
        "Make a U-turn.",
        "Generate a U-turn-like route.",
        "Create a trajectory that doubles back.",
        "The route should reverse direction.",
    ],
    "multi_turn": [
        "Make multiple turns.",
        "Generate a multi-turn route.",
        "Create a trajectory with several turns.",
        "The route should include more than one major turn.",
    ],
}

CURVATURE_ALT = {
    "smooth": ["smooth", "gently curved", "low-curvature"],
    "moderate": ["moderately curved", "with moderate curvature", "not too winding"],
    "winding": ["winding", "curvy", "high-curvature"],
}

PROGRESS_ALT = {
    "slow": ["conservative progress", "slow progress", "less direct progress"],
    "normal": ["normal progress", "typical progress", "balanced progress"],
    "fast": ["aggressive progress", "fast progress", "more direct progress"],
}

REFERENCE_ALT = {
    "tight": ["stay close to the reference", "closely follow the reference route", "keep tight reference adherence"],
    "moderate": ["generally follow the reference", "allow small deviations from the reference", "roughly track the reference route"],
    "loose": ["allow noticeable deviation", "take a looser route around the reference", "deviate from the reference when needed"],
    "unknown": ["use the reference as weak context", "treat the reference route as guidance", "consider the reference route loosely"],
}

REFERENCE_COMMANDS = {
    "tight": [
        "stay close to the reference",
        "stays close to the reference",
        "tight reference adherence",
        "closely follow the reference",
    ],
    "moderate": [
        "moderate reference adherence",
        "roughly follow the reference",
        "allow small deviations from the reference",
    ],
    "loose": [
        "loose reference adherence",
        "loosely follow the reference",
        "allow loose reference adherence",
        "deviate from the reference",
        "allow noticeable deviation from the reference",
    ],
    "unknown": [
        "use the reference as context",
        "weak reference guidance",
        "reference as weak context",
    ],
}

FIELD_FOCUSED_PATTERNS = [
    "Generate a {distance} route that should {direction}. It should {reference}, with {progress} and {curvature} geometry.",
    "Create a {distance} city trip: {direction}. Keep {reference}; use {progress}; make the path {curvature}.",
    "I need a {distance} trajectory that will {direction}. The route should {reference} and remain {curvature}.",
    "Plan a {distance} path: {direction}, {reference}, {progress}, {curvature}.",
]

COMPACT_PATTERNS = [
    "{distance} route, {direction}, {reference}, {progress}, {curvature}.",
    "{direction}; {distance} trip; {reference}; {curvature}; {progress}.",
    "Generate: {distance}, {direction}, {reference}, {curvature}, {progress}.",
    "Generate a {distance} route that {direction} and {reference}.",
    "Create a {distance} trajectory that {direction} with {reference}.",
    "Generate a {distance} route with {curvature} shape, {progress}, and {reference}.",
]


def template_paraphrases(labels: Dict[str, str], rng: random.Random, num: int) -> List[str]:
    distance = labels.get("distance", "medium")
    direction = labels.get("direction", "straight")
    curvature = labels.get("curvature", "smooth")
    progress = labels.get("progress", "normal")
    adherence = labels.get("reference_adherence", "unknown")

    variants = []
    direction_commands = DIRECTION_COMMANDS.get(direction, [])
    for command in direction_commands[: min(len(direction_commands), max(1, num // 4))]:
        variants.append(
            f"{command} Make it {DISTANCE_ALT.get(distance, [distance])[0]}, "
            f"{REFERENCE_ALT.get(adherence, [adherence])[0]}, with {PROGRESS_ALT.get(progress, [progress])[0]} "
            f"and {CURVATURE_ALT.get(curvature, [curvature])[0]} geometry."
        )

    while len(variants) < num:
        d = rng.choice(DISTANCE_ALT.get(distance, [DISTANCE_TEXT.get(distance, "urban")]))
        turn = rng.choice(DIRECTION_ALT.get(direction, [DIRECTION_TEXT.get(direction, "follow the route")]))
        curve = rng.choice(CURVATURE_ALT.get(curvature, [CURVATURE_TEXT.get(curvature, "realistic")]))
        prog = rng.choice(PROGRESS_ALT.get(progress, [PROGRESS_TEXT.get(progress, "normal progress")]))
        ref = rng.choice(
            REFERENCE_ALT.get(adherence, [REFERENCE_TEXT.get(adherence, "use the reference")])
            + REFERENCE_COMMANDS.get(adherence, [])
        )
        pattern = rng.choice(FIELD_FOCUSED_PATTERNS + COMPACT_PATTERNS)
        variants.append(
            pattern.format(
                distance=d,
                direction=turn,
                reference=ref,
                progress=prog,
                curvature=curve,
            )
        )
    return variants[:num]


def focused_field_examples(labels: Dict[str, str]) -> List[str]:
    distance = labels.get("distance", "medium")
    direction = labels.get("direction", "straight")
    curvature = labels.get("curvature", "smooth")
    progress = labels.get("progress", "normal")
    adherence = labels.get("reference_adherence", "unknown")
    return [
        f"Distance: {distance}. Direction: {direction}. Curvature: {curvature}. Progress: {progress}. Reference adherence: {adherence}.",
        f"Use {distance} distance, {direction} direction, {curvature} curvature, {progress} progress, and {adherence} reference adherence.",
        f"Generate a trajectory with {curvature} curvature and {adherence} reference adherence.",
        f"Create a {distance} trajectory with {progress} progress.",
        f"Make the route {direction}.",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local paraphrase data for text-to-token planning.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--paraphrases-per-sample", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_path = args.input or (REPO_ROOT / "outputs" / "instruction_data" / f"{args.dataset}_instruction_samples.jsonl")
    output_path = args.output or (REPO_ROOT / "outputs" / "text_data" / f"{args.dataset}_text_to_token.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_idx, line in enumerate(src):
            if args.max_samples > 0 and line_idx >= args.max_samples:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            labels = dict(record.get("labels", {}))
            texts = [str(record.get("instruction", ""))]
            texts.extend(template_paraphrases(labels, rng, args.paraphrases_per_sample))
            texts.extend(focused_field_examples(labels))
            for variant_idx, text in enumerate(texts):
                out = {
                    "sample_id": int(record["sample_id"]),
                    "variant_id": int(variant_idx),
                    "dataset": record.get("dataset", args.dataset),
                    "instruction": text,
                    "source_instruction": record.get("instruction", ""),
                    "labels": labels,
                    "mobility_tokens": DEFAULT_MOBILITY_VOCAB.encode(labels),
                    "source": "template" if variant_idx > 0 else "rule",
                }
                dst.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "num_records": written,
        "paraphrases_per_sample": args.paraphrases_per_sample,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
