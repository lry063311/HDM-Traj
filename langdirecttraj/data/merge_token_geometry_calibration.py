from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict

from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_field_sources(values: list[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected FIELD=PATH override, got: {item}")
        field, raw_path = item.split("=", 1)
        field = field.strip()
        if field not in DEFAULT_MOBILITY_VOCAB.fields:
            raise ValueError(f"Unknown calibration field override: {field}")
        out[field] = Path(raw_path)
    return out


def load_json(path: Path) -> dict:
    if not path.is_absolute():
        path = REPO_ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def merge_calibrations(base_path: Path, overrides: Dict[str, Path]) -> dict:
    merged = deepcopy(load_json(base_path))
    merged.setdefault("fields", {})
    merge_sources = {"base": str(base_path), "overrides": {}}
    for field, path in overrides.items():
        payload = load_json(path)
        if field not in payload.get("fields", {}):
            raise ValueError(f"Field '{field}' was not found in override calibration: {path}")
        merged["fields"][field] = deepcopy(payload["fields"][field])
        merge_sources["overrides"][field] = str(path)
    merged["merged_from"] = merge_sources
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge token-geometry calibration fields from multiple JSON files.")
    parser.add_argument("--base", type=Path, required=True, help="Base calibration JSON used for fields without overrides.")
    parser.add_argument("--override", nargs="+", default=[], help="Field overrides like direction=path/to/calibration.json")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_calibrations(args.base, parse_field_sources(args.override))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    preview = {
        "output": str(args.output),
        "base": str(args.base),
        "overrides": sorted(parse_field_sources(args.override)),
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
