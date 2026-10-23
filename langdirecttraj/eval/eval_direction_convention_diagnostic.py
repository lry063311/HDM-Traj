from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch

from langdirecttraj.data.build_token_geometry_calibration import geometry_metrics, read_jsonl
from langdirecttraj.eval.eval_direction_calibration import (
    JsonDirectionCalibrator,
    imotion_style_direction_from_points,
    macro_f1,
)
from langdirecttraj.eval.instruction_geometry import GeometryTokenCalibrator, geometry_values, tensor_dict_to_rows
from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "outputs"
    / "local_porto_od_chord_256_coverage_8k"
    / "instruction_data"
    / "porto_instruction_samples.jsonl"
)
DEFAULT_CALIBRATION = (
    REPO_ROOT
    / "outputs"
    / "local_goal_cf_5field_balanced128_v1"
    / "porto_token_geometry_calibration.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "direction_convention_diagnostic"


CURRENT_DECODER_DIRECTION_SIGN = {
    "straight": 0.0,
    "left_turn": -1.0,
    "right_turn": 1.0,
    "u_turn": 0.0,
    "multi_turn": 0.35,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether direction labels, calibration centers, evaluator "
            "rules, and decoder direction signs share one left/right convention."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--target-key", default="trajectory", choices=["trajectory", "pseudo_trajectory"])
    parser.add_argument("--target-len", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-cases", type=int, default=128)
    parser.add_argument("--probe-len", type=int, default=96)
    parser.add_argument("--probe-amp", type=float, default=0.08)
    return parser.parse_args()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": float("nan"), "p50": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _sign(value: float, eps: float = 1e-6) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _read_calibration_direction(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokens = payload.get("fields", {}).get("direction", {}).get("tokens", {})
    out = {}
    for token in DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]:
        item = tokens.get(token, {})
        center = _safe_float(item.get("p50", item.get("mean", 0.0)))
        out[token] = {
            "metric": str(item.get("direction_metric", item.get("metric", "direction_token_metric"))),
            "center": center,
            "center_sign": _sign(center),
            "count": int(item.get("count", 0) or 0),
            "scale": _safe_float(item.get("scale", item.get("iqr", item.get("std", 1.0))), 1.0),
        }
    return out


def _imotion_from_geom(geom: Mapping[str, float]) -> str:
    dx = _safe_float(geom.get("local_dx"))
    dy = _safe_float(geom.get("local_dy"))
    heading = _safe_float(geom.get("local_heading_diff_rad"))
    signed_turn = _safe_float(geom.get("local_signed_total_turn_rad"))
    total_abs = _safe_float(geom.get("local_total_abs_turn_rad"))
    turn_count = _safe_float(geom.get("local_turn_count"))
    left_ratio = _safe_float(geom.get("local_left_turn_ratio"))
    right_ratio = _safe_float(geom.get("local_right_turn_ratio"))
    if dx < -0.05 and abs(heading) >= 2.0:
        return "u_turn"
    if abs(heading) < math.pi / 6.0 and abs(dy) < 0.08:
        return "straight"
    left_strength = max(dy / 0.08, signed_turn / (math.pi / 6.0), left_ratio)
    right_strength = max(-dy / 0.08, -signed_turn / (math.pi / 6.0), right_ratio)
    if left_strength >= 1.0 and left_strength >= right_strength:
        return "left_turn"
    if right_strength >= 1.0 and right_strength > left_strength:
        return "right_turn"
    if turn_count >= 3.0 or total_abs >= 4.5:
        return "multi_turn"
    if dy > 0 or signed_turn > 0:
        return "left_turn"
    return "right_turn"


def _decoder_probe(calibrator: GeometryTokenCalibrator, target_len: int, amp: float) -> List[Dict]:
    t = torch.linspace(0.0, 1.0, int(target_len))
    reference = torch.stack([t, torch.zeros_like(t)], dim=-1).unsqueeze(0)
    route_scales = torch.ones(1)
    rows = []
    for token in ["left_turn", "right_turn"]:
        decoder_sign = float(CURRENT_DECODER_DIRECTION_SIGN[token])
        points = reference.clone()
        points[:, :, 1] = decoder_sign * float(amp) * torch.sin(math.pi * t)
        geom = tensor_dict_to_rows(geometry_values(points, reference, route_scales))[0]
        norm_points = points[0].detach().cpu().numpy()
        imotion = imotion_style_direction_from_points(
            norm_points,
            straight_heading_rad=math.pi / 6.0,
            straight_lateral=0.08,
            uturn_longitudinal=-0.05,
            uturn_heading_rad=2.0,
            multiturn_total_abs_rad=4.5,
            multiturn_count=3,
        )
        rows.append(
            {
                "token": token,
                "decoder_direction_sign": decoder_sign,
                "midpoint_normal_sign": _sign(float(points[0, target_len // 2, 1].item())),
                "local_signed_total_turn_rad": _safe_float(geom.get("local_signed_total_turn_rad")),
                "local_heading_diff_rad": _safe_float(geom.get("local_heading_diff_rad")),
                "local_dy": _safe_float(geom.get("local_dy")),
                "imotion_prediction": imotion,
                "calibration_prediction": calibrator.predict_field("direction", geom),
            }
        )
    return rows


def _build_rows(args: argparse.Namespace, calibrator: JsonDirectionCalibrator) -> List[Dict]:
    rows = []
    for idx, record in enumerate(read_jsonl(args.input, args.max_samples)):
        labels = record.get("labels", {}) if isinstance(record.get("labels"), dict) else {}
        label = str(labels.get("direction", "unknown"))
        if label not in DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]:
            continue
        geom = geometry_metrics(record, args.target_key, args.target_len)
        if geom is None:
            continue
        auto = calibrator.predict(geom)
        imotion = _imotion_from_geom(geom)
        row = {
            "row_index": idx,
            "sample_id": record.get("sample_id", idx),
            "label": label,
            "calibration_prediction": auto,
            "imotion_prediction": imotion,
            "label_eq_calibration": bool(label == auto),
            "label_eq_imotion": bool(label == imotion),
            "left_right_label": bool(label in {"left_turn", "right_turn"}),
            "left_right_calibration": bool(auto in {"left_turn", "right_turn"}),
            "left_right_imotion": bool(imotion in {"left_turn", "right_turn"}),
            "local_signed_total_turn_rad": _safe_float(geom.get("local_signed_total_turn_rad")),
            "local_dy": _safe_float(geom.get("local_dy")),
            "local_heading_diff_rad": _safe_float(geom.get("local_heading_diff_rad")),
            "local_total_abs_turn_rad": _safe_float(geom.get("local_total_abs_turn_rad")),
            "local_turn_count": _safe_float(geom.get("local_turn_count")),
            "local_left_turn_ratio": _safe_float(geom.get("local_left_turn_ratio")),
            "local_right_turn_ratio": _safe_float(geom.get("local_right_turn_ratio")),
            "instruction": str(record.get("instruction", "")),
        }
        rows.append(row)
    return rows


def _by_label_summary(rows: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for label in DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]:
        items = [row for row in rows if row["label"] == label]
        out[label] = {
            "count": len(items),
            "local_signed_total_turn_rad": _summarize(row["local_signed_total_turn_rad"] for row in items),
            "local_dy": _summarize(row["local_dy"] for row in items),
            "calibration_prediction_counts": dict(Counter(row["calibration_prediction"] for row in items)),
            "imotion_prediction_counts": dict(Counter(row["imotion_prediction"] for row in items)),
            "label_eq_calibration_rate": float(np.mean([row["label_eq_calibration"] for row in items])) if items else float("nan"),
            "label_eq_imotion_rate": float(np.mean([row["label_eq_imotion"] for row in items])) if items else float("nan"),
        }
    return out


def _agreement_rate(rows: List[Dict], left_positive: bool) -> float:
    lr = [row for row in rows if row["label"] in {"left_turn", "right_turn"}]
    if not lr:
        return float("nan")
    expected = {"left_turn": 1 if left_positive else -1, "right_turn": -1 if left_positive else 1}
    return float(np.mean([_sign(row["local_signed_total_turn_rad"]) == expected[row["label"]] for row in lr]))


def build_summary(args: argparse.Namespace) -> Dict:
    json_calibrator = JsonDirectionCalibrator.from_json(args.calibration_path)
    geometry_calibrator = GeometryTokenCalibrator.from_json(str(args.calibration_path))
    rows = _build_rows(args, json_calibrator)
    if not rows:
        raise ValueError(f"No valid direction rows read from {args.input}")

    labels = list(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"].keys())
    y_true = [str(row["label"]) for row in rows]
    y_calib = [str(row["calibration_prediction"]) for row in rows]
    y_imotion = [str(row["imotion_prediction"]) for row in rows]
    by_label = _by_label_summary(rows)
    calibration = _read_calibration_direction(args.calibration_path)
    decoder_probe = _decoder_probe(geometry_calibrator, args.probe_len, args.probe_amp)

    left_calib_sign = calibration.get("left_turn", {}).get("center_sign")
    right_calib_sign = calibration.get("right_turn", {}).get("center_sign")
    left_label_median = by_label["left_turn"]["local_signed_total_turn_rad"]["p50"]
    right_label_median = by_label["right_turn"]["local_signed_total_turn_rad"]["p50"]
    expected_left_positive = {
        "left_turn_expected_sign": 1,
        "right_turn_expected_sign": -1,
        "source": "eval_direction_calibration.imotion_style_direction_from_points and geometry_contract.direction_label",
    }
    flags = {
        "calibration_left_right_inverted_vs_evaluator": bool(left_calib_sign == -1 and right_calib_sign == 1),
        "label_left_right_inverted_vs_evaluator": bool(_sign(left_label_median) == -1 and _sign(right_label_median) == 1),
        "decoder_probe_inverted_vs_evaluator": any(
            row["token"] != row["imotion_prediction"] for row in decoder_probe if row["token"] in {"left_turn", "right_turn"}
        ),
        "right_turn_calibration_support_low": by_label["right_turn"]["count"] > 0
        and by_label["right_turn"]["label_eq_calibration_rate"] < 0.5,
        "right_turn_imotion_support_low": by_label["right_turn"]["count"] > 0
        and by_label["right_turn"]["label_eq_imotion_rate"] < 0.5,
    }
    cases = [
        row
        for row in rows
        if (
            row["label"] in {"left_turn", "right_turn"}
            and (not row["label_eq_calibration"] or not row["label_eq_imotion"])
        )
    ][: int(args.max_cases)]
    return {
        "current_best_baseline": {
            "checkpoint": "outputs/local_goal_reference_control_all_s035_only_e1_64/chunk_001_epoch_0001/train/dit_anchor/porto/latest.pt",
            "table1_route_transition_jsd": 0.226981,
            "table2_bleu_like": 0.229203,
            "table2_c_jsd": 0.317937,
            "table3_direction_f1": 0.524286,
            "table4_success": 0.781250,
            "table4_independent_collateral": 0.144208,
        },
        "metric_priority": {
            "primary": "Table3 Direction F1 and right_turn candidate coverage",
            "secondary": "Table4 target effect/success",
            "guardrails": "Route Transition JSD, Table2 C-JSD/BLEU-like, collateral drift",
        },
        "reference_evidence_table": [
            {
                "reference": "iMotion-LLM",
                "reusable_principle": "Use local-frame direction rules and filter inconsistent direction semantics.",
                "why_it_matters_here": "Left/right must be defined in one local frame before losses or calibration can help.",
                "confidence": "high",
            },
            {
                "reference": "GTG",
                "reusable_principle": "Guidance/control needs guardrails so one objective does not collapse realism.",
                "why_it_matters_here": "Direction fixes must not worsen Route Transition JSD or Table2 C-JSD.",
                "confidence": "medium",
            },
            {
                "reference": "TrajGPT-R",
                "reusable_principle": "Token metrics are useful mobility-pattern guardrails, not continuous-geometry contracts.",
                "why_it_matters_here": "Table2 cannot decide whether left/right continuous trajectories are geometrically correct.",
                "confidence": "medium",
            },
        ],
        "input": str(args.input),
        "target_key": args.target_key,
        "num_rows": len(rows),
        "calibration_path": str(args.calibration_path),
        "evaluator_expected_sign": expected_left_positive,
        "calibration_direction_centers": calibration,
        "dataset_label_sign_agreement": {
            "left_positive_contract_agreement_rate": _agreement_rate(rows, left_positive=True),
            "left_negative_contract_agreement_rate": _agreement_rate(rows, left_positive=False),
            "left_turn_label_median_signed_turn": left_label_median,
            "right_turn_label_median_signed_turn": right_label_median,
        },
        "prediction_metrics": {
            "calibration": macro_f1(y_true, y_calib, labels),
            "imotion": macro_f1(y_true, y_imotion, labels),
        },
        "by_label": by_label,
        "decoder_probe": decoder_probe,
        "abnormal_flags": flags,
        "root_cause_hypothesis": (
            "Direction is not yet one coherent contract if any abnormal flag is true: "
            "labels/calibration/decoder/evaluator are then optimizing opposite left-right signs."
        ),
        "decision_threshold": {
            "continue_to_training_if": [
                "left/right labels, calibration centers, evaluator rules, and decoder probe agree on sign",
                "right_turn candidate coverage becomes non-zero on Table3 32-sample diagnostic",
                "Route Transition JSD and Table2 C-JSD do not degrade beyond current baseline guardrails",
            ],
            "fix_calibration_or_labels_if": [
                "calibration centers or dataset label medians are inverted against evaluator expected sign",
            ],
            "fix_decoder_if": [
                "calibration/labels agree with evaluator but decoder probe predicts the opposite turn",
            ],
        },
        "cases": cases,
    }


def write_outputs(summary: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = list(summary.pop("cases"))
    metrics_path = output_dir / "metrics.json"
    cases_path = output_dir / "cases.jsonl"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with cases_path.open("w", encoding="utf-8") as f:
        for row in cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"metrics_path": str(metrics_path), "cases_path": str(cases_path), **summary}, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    write_outputs(summary, args.output_dir)


if __name__ == "__main__":
    main()
