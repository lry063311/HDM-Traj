from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np

from langdirecttraj.data.build_token_geometry_calibration import (
    geometry_metrics,
    local_normalized_pair,
    read_jsonl,
    resample_polyline,
)
from langdirecttraj.data.instruction_dataset import validate_official_record_protocol
from langdirecttraj.eval.instruction_geometry import contract_direction_rule_scores
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "outputs"
    / "local_porto_od_chord_256_verify"
    / "controllability_data"
    / "porto_counterfactual_instruction_samples.filtered.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "calibration" / "direction_calibration_diagnostic.json"
CALIBRATION_METRIC_ALIASES = {
    "direction_token_metric": "signed_net_heading_rad",
}


class JsonDirectionCalibrator:
    def __init__(self, stats: Dict[str, Dict[str, float]]) -> None:
        self.stats = stats

    @classmethod
    def from_json(cls, path: Path) -> "JsonDirectionCalibrator":
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokens = payload.get("fields", {}).get("direction", {}).get("tokens", {})
        default_metric = payload.get("fields", {}).get("direction", {}).get("metric", "signed_net_heading_rad")
        default_metric = CALIBRATION_METRIC_ALIASES.get(str(default_metric), str(default_metric))
        stats = {}
        for token in DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]:
            token_payload = tokens.get(token, {})
            metric = str(token_payload.get("direction_metric", default_metric))
            metric = CALIBRATION_METRIC_ALIASES.get(metric, metric)
            center = float(token_payload.get("p50", token_payload.get("mean", 0.0)))
            scale = float(token_payload.get("scale", token_payload.get("iqr", token_payload.get("std", 1.0))))
            if not math.isfinite(center):
                center = 0.0
            if not math.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            stats[token] = {
                "metric": metric,
                "center": center,
                "scale": max(scale, 1e-6),
            }
        return cls(stats)

    def score(self, token: str, geom: Mapping[str, float]) -> float:
        stat = self.stats[token]
        value = float(geom.get(str(stat["metric"]), 0.0))
        return abs(value - float(stat["center"])) / max(float(stat["scale"]), 1e-6)

    def predict(self, geom: Mapping[str, float]) -> str:
        scores = contract_direction_rule_scores(geom)
        return min(self.stats.keys(), key=lambda token: scores.get(token, self.score(token, geom)))


def macro_f1(y_true: List[str], y_pred: List[str], labels: Iterable[str]) -> Dict:
    per_label = {}
    values = []
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        support = sum(1 for truth in y_true if truth == label)
        predicted = sum(1 for pred in y_pred if pred == label)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        if support > 0:
            values.append(f1)
        per_label[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
            "predicted": int(predicted),
        }
    return {
        "macro_f1": float(np.mean(values)) if values else float("nan"),
        "accuracy": float(np.mean([truth == pred for truth, pred in zip(y_true, y_pred)])) if y_true else float("nan"),
        "per_label": per_label,
    }


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _moving_diffs(points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    diffs = points[1:] - points[:-1]
    keep = np.linalg.norm(diffs, axis=1) > eps
    return diffs[keep]


def _heading_stats_from_diffs(diffs: np.ndarray) -> Dict[str, float]:
    if len(diffs) < 2:
        return {
            "heading_diff": 0.0,
            "signed_total_turn": 0.0,
            "total_abs_turn": 0.0,
            "max_abs_turn": 0.0,
            "turn_count": 0.0,
            "left_turn_ratio": 0.0,
            "right_turn_ratio": 0.0,
        }
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    deltas = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
    threshold = math.radians(15.0)
    strong = np.abs(deltas) > threshold
    strong_count = max(float(strong.sum()), 1.0)
    return {
        "heading_diff": _wrap_angle(float(headings[-1] - headings[0])),
        "signed_total_turn": float(deltas.sum()),
        "total_abs_turn": float(np.abs(deltas).sum()),
        "max_abs_turn": float(np.abs(deltas).max()) if deltas.size else 0.0,
        "turn_count": float(strong.sum()),
        "left_turn_ratio": float((deltas > threshold).sum() / strong_count),
        "right_turn_ratio": float((deltas < -threshold).sum() / strong_count),
    }


def imotion_style_direction(
    record: Mapping,
    target_key: str,
    target_len: int,
    straight_heading_rad: float,
    straight_lateral: float,
    uturn_longitudinal: float,
    uturn_heading_rad: float,
    multiturn_total_abs_rad: float,
    multiturn_count: int,
) -> str:
    """Classify macro direction with iMotion-LLM-like local-frame rules."""
    target = np.asarray(record.get(target_key, record.get("trajectory", [])), dtype=np.float64)
    validate_official_record_protocol(dict(record), row_index=int(record.get("sample_id", 0)))
    if "reference" not in record:
        raise ValueError("Official OD-chord direction calibration requires an explicit reference field.")
    reference = np.asarray(record["reference"], dtype=np.float64)
    if target.ndim != 2 or target.shape[-1] != 2 or len(target) < 2:
        return "straight"
    if reference.ndim != 2 or reference.shape[-1] != 2 or len(reference) < 2:
        raise ValueError("Official OD-chord direction calibration received an invalid reference polyline.")
    points, _, _ = local_normalized_pair(target, reference)
    points = resample_polyline(points, target_len)
    return imotion_style_direction_from_points(
        points,
        straight_heading_rad=straight_heading_rad,
        straight_lateral=straight_lateral,
        uturn_longitudinal=uturn_longitudinal,
        uturn_heading_rad=uturn_heading_rad,
        multiturn_total_abs_rad=multiturn_total_abs_rad,
        multiturn_count=multiturn_count,
    )


def imotion_style_direction_from_points(
    points: np.ndarray,
    straight_heading_rad: float,
    straight_lateral: float,
    uturn_longitudinal: float,
    uturn_heading_rad: float,
    multiturn_total_abs_rad: float,
    multiturn_count: int,
) -> str:
    """Classify already-normalized local points with iMotion-LLM-like rules."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[-1] != 2 or len(points) < 3:
        return "straight"
    diffs = _moving_diffs(points)
    if len(diffs) < 2:
        return "straight"

    start_heading = math.atan2(float(diffs[0, 1]), float(diffs[0, 0]))
    cos_h = math.cos(-start_heading)
    sin_h = math.sin(-start_heading)
    rot = np.asarray([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
    local = (points - points[0:1]) @ rot.T
    local_diffs = _moving_diffs(local)
    stats = _heading_stats_from_diffs(local_diffs)
    dx = float(local[-1, 0] - local[0, 0])
    dy = float(local[-1, 1] - local[0, 1])

    if dx < uturn_longitudinal and abs(stats["heading_diff"]) >= uturn_heading_rad:
        return "u_turn"
    if abs(stats["heading_diff"]) < straight_heading_rad and abs(dy) < straight_lateral:
        return "straight"
    left_strength = max(
        dy / max(straight_lateral, 1e-6),
        stats["signed_total_turn"] / max(straight_heading_rad, 1e-6),
        stats["left_turn_ratio"],
    )
    right_strength = max(
        -dy / max(straight_lateral, 1e-6),
        -stats["signed_total_turn"] / max(straight_heading_rad, 1e-6),
        stats["right_turn_ratio"],
    )
    if left_strength >= 1.0 and left_strength >= right_strength:
        return "left_turn"
    if right_strength >= 1.0 and right_strength > left_strength:
        return "right_turn"
    if stats["turn_count"] >= multiturn_count or stats["total_abs_turn"] >= multiturn_total_abs_rad:
        return "multi_turn"
    if dy > 0 or stats["signed_total_turn"] > 0:
        return "left_turn"
    return "right_turn"


def confusion_matrix(y_true: List[str], y_pred: List[str], labels: Iterable[str]) -> Dict[str, Dict[str, int]]:
    matrix = {label: {other: 0 for other in labels} for label in labels}
    for truth, pred in zip(y_true, y_pred):
        if truth in matrix and pred in matrix[truth]:
            matrix[truth][pred] += 1
    return matrix


def summarize_predictions(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    metrics = macro_f1(y_true, y_pred, labels)
    return {
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "label_distribution": dict(sorted(Counter(y_true).items())),
        "prediction_distribution": dict(sorted(Counter(y_pred).items())),
        "per_label": metrics["per_label"],
        "confusion": confusion_matrix(y_true, y_pred, labels),
    }


def hybrid_predictions(
    auto_pred: List[str],
    imotion_pred: List[str],
    policy: str,
) -> List[str]:
    if policy == "auto_lr_else_imotion":
        return [
            auto if auto in {"left_turn", "right_turn"} else imotion
            for auto, imotion in zip(auto_pred, imotion_pred)
        ]
    if policy == "auto_lr_when_imotion_multi_else_imotion":
        return [
            auto if auto in {"left_turn", "right_turn"} and imotion == "multi_turn" else imotion
            for auto, imotion in zip(auto_pred, imotion_pred)
        ]
    if policy == "auto_straight_lr_else_imotion":
        return [
            auto if auto in {"straight", "left_turn", "right_turn"} else imotion
            for auto, imotion in zip(auto_pred, imotion_pred)
        ]
    raise ValueError(f"Unknown hybrid policy: {policy}")


def parse_calibration_arg(values: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in values:
        if "=" in item:
            name, raw_path = item.split("=", 1)
        else:
            path = Path(item)
            name, raw_path = path.stem, item
        out[name.strip()] = Path(raw_path)
    return out


def build_diagnostic(args: argparse.Namespace) -> Dict:
    labels = list(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"].keys())
    rows: List[tuple[str, Dict[str, float], Mapping]] = []
    skipped = 0
    for record in read_jsonl(args.input, args.max_samples):
        direction = str(record.get("labels", {}).get("direction", "unknown"))
        geom = geometry_metrics(record, args.target_key, args.target_len)
        if direction not in labels or geom is None:
            skipped += 1
            continue
        rows.append((direction, geom, record))

    y_true = [truth for truth, _, _ in rows]
    results = {}
    calibration_predictions = {}
    for name, path in parse_calibration_arg(args.calibration).items():
        if not path.is_absolute():
            path = REPO_ROOT / path
        calibrator = JsonDirectionCalibrator.from_json(path)
        y_pred = [calibrator.predict(geom) for _, geom, _ in rows]
        calibration_predictions[name] = y_pred
        results[name] = {
            "type": "geometry_token_calibration",
            "path": str(path),
            **summarize_predictions(y_true, y_pred, labels),
        }

    imotion_predictions = None
    if args.include_imotion_style:
        imotion_predictions = [
            imotion_style_direction(
                record,
                target_key=args.target_key,
                target_len=args.target_len,
                straight_heading_rad=args.imotion_straight_heading_rad,
                straight_lateral=args.imotion_straight_lateral,
                uturn_longitudinal=args.imotion_uturn_longitudinal,
                uturn_heading_rad=args.imotion_uturn_heading_rad,
                multiturn_total_abs_rad=args.imotion_multiturn_total_abs_rad,
                multiturn_count=args.imotion_multiturn_count,
            )
            for _, _, record in rows
        ]
        results["imotion_style"] = {
            "type": "local_frame_rule_classifier",
            "params": {
                "straight_heading_rad": args.imotion_straight_heading_rad,
                "straight_lateral": args.imotion_straight_lateral,
                "uturn_longitudinal": args.imotion_uturn_longitudinal,
                "uturn_heading_rad": args.imotion_uturn_heading_rad,
                "multiturn_total_abs_rad": args.imotion_multiturn_total_abs_rad,
                "multiturn_count": args.imotion_multiturn_count,
            },
            **summarize_predictions(y_true, imotion_predictions, labels),
        }

    if args.include_hybrid and imotion_predictions is not None:
        auto_name = args.hybrid_auto_source
        if auto_name not in calibration_predictions:
            raise ValueError(f"--hybrid-auto-source '{auto_name}' was not found in --calibration names.")
        for policy in args.hybrid_policy:
            y_pred = hybrid_predictions(calibration_predictions[auto_name], imotion_predictions, policy)
            results[f"hybrid_{policy}"] = {
                "type": "hybrid_calibration",
                "auto_source": auto_name,
                "policy": policy,
                **summarize_predictions(y_true, y_pred, labels),
            }

    ranking = sorted(
        (
            {
                "name": name,
                "macro_f1": payload["macro_f1"],
                "accuracy": payload["accuracy"],
            }
            for name, payload in results.items()
        ),
        key=lambda item: (item["macro_f1"], item["accuracy"]),
        reverse=True,
    )
    return {
        "input": str(args.input),
        "target_key": args.target_key,
        "target_len": args.target_len,
        "num_samples": len(rows),
        "num_skipped": skipped,
        "labels": labels,
        "ranking": ranking,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direction calibration choices on target trajectories.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--target-key", default="pseudo_trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--target-len", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--calibration",
        nargs="+",
        default=[
            "old=outputs/calibration/porto_token_geometry_calibration.json",
            "v2_fixed=outputs/calibration/porto_token_geometry_calibration_direction_v2.json",
            "v2_auto=outputs/calibration/porto_token_geometry_calibration_direction_v2_auto.json",
        ],
        help="Calibration files as name=path.",
    )
    parser.add_argument("--include-imotion-style", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-hybrid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hybrid-auto-source", default="v2_auto")
    parser.add_argument(
        "--hybrid-policy",
        nargs="+",
        default=[
            "auto_lr_else_imotion",
            "auto_lr_when_imotion_multi_else_imotion",
            "auto_straight_lr_else_imotion",
        ],
    )
    parser.add_argument("--imotion-straight-heading-rad", type=float, default=math.pi / 6.0)
    parser.add_argument("--imotion-straight-lateral", type=float, default=0.08)
    parser.add_argument("--imotion-uturn-longitudinal", type=float, default=-0.05)
    parser.add_argument("--imotion-uturn-heading-rad", type=float, default=2.0)
    parser.add_argument("--imotion-multiturn-total-abs-rad", type=float, default=4.5)
    parser.add_argument("--imotion-multiturn-count", type=int, default=3)
    args = parser.parse_args()

    diagnostic = build_diagnostic(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
    preview = {
        "output": str(args.output),
        "num_samples": diagnostic["num_samples"],
        "ranking": diagnostic["ranking"],
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
