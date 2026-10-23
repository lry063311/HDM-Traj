from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langdirecttraj.eval.distribution_metrics import haversine_km, infer_bounds, to_grid_ids  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def compact_route_hash(traj: np.ndarray, bounds: dict, grid_size: int) -> tuple[int, ...]:
    ids = to_grid_ids(np.asarray(traj, dtype=np.float32), bounds, grid_size)
    if len(ids) == 0:
        return tuple()
    keep = np.ones(len(ids), dtype=bool)
    keep[1:] = ids[1:] != ids[:-1]
    return tuple(int(x) for x in ids[keep])


def load_npz_key(path: Path, key: str) -> np.ndarray:
    payload = np.load(Path(path))
    if key not in payload.files:
        raise KeyError(f"{key!r} is not in {path}; available keys: {payload.files}")
    return np.asarray(payload[key])


def summarize_jsonl_guard(path: Path, max_records: int) -> dict:
    counts = Counter()
    chord_error_max = 0.0
    chord_error_mean_max = 0.0
    non_endpoint_source_max = 0.0
    reference_to_target_mean_sum = 0.0
    reference_to_target_p90_sum = 0.0

    for i, record in enumerate(iter_jsonl(path)):
        if max_records and i >= max_records:
            break
        counts["records"] += 1
        guard = record.get("leakage_guard", {})
        audit = record.get("reference_audit", {})
        if guard.get("reference_uses_gt_middle_geometry") is False:
            counts["guard_no_gt_middle"] += 1
        if guard.get("reference_uses_only_origin_destination") is True:
            counts["guard_od_only"] += 1
        if record.get("reference_protocol") == "od_chord_only":
            counts["od_chord_protocol"] += 1
        if record.get("condition_source") == "origin_destination_chord":
            counts["od_chord_condition"] += 1

        chord_error_max = max(chord_error_max, float(audit.get("reference_chord_error_max_m", 0.0)))
        chord_error_mean_max = max(chord_error_mean_max, float(audit.get("reference_chord_error_mean_m", 0.0)))
        non_endpoint_source_max = max(non_endpoint_source_max, float(audit.get("reference_non_endpoint_source_points", 0.0)))
        reference_to_target_mean_sum += float(audit.get("reference_to_target_distance_mean_m", 0.0))
        reference_to_target_p90_sum += float(audit.get("reference_to_target_distance_p90_m", 0.0))

    n = int(counts["records"])
    denom = max(n, 1)
    return {
        "records_checked": n,
        "reference_uses_gt_middle_geometry_false_rate": float(counts["guard_no_gt_middle"] / denom),
        "reference_uses_only_origin_destination_true_rate": float(counts["guard_od_only"] / denom),
        "od_chord_protocol_rate": float(counts["od_chord_protocol"] / denom),
        "origin_destination_chord_condition_rate": float(counts["od_chord_condition"] / denom),
        "reference_chord_error_max_m": float(chord_error_max),
        "reference_chord_error_mean_max_m": float(chord_error_mean_max),
        "reference_non_endpoint_source_points_max": float(non_endpoint_source_max),
        "reference_to_target_distance_mean_m": float(reference_to_target_mean_sum / denom),
        "reference_to_target_distance_p90_m": float(reference_to_target_p90_sum / denom),
    }


def copy_distance_summary(generated: np.ndarray, target: np.ndarray, thresholds_m: list[float]) -> dict:
    if generated.shape != target.shape:
        raise ValueError(f"generated and target shapes differ: {generated.shape} vs {target.shape}")
    point_dist_m = haversine_km(generated, target) * 1000.0
    mean_m = point_dist_m.mean(axis=1)
    max_m = point_dist_m.max(axis=1)
    out = {
        "same_sample_mean_point_distance_m": {
            "mean": float(mean_m.mean()),
            "p50": float(np.percentile(mean_m, 50)),
            "p90": float(np.percentile(mean_m, 90)),
            "p99": float(np.percentile(mean_m, 99)),
        },
        "same_sample_max_point_distance_m": {
            "mean": float(max_m.mean()),
            "p50": float(np.percentile(max_m, 50)),
            "p90": float(np.percentile(max_m, 90)),
            "p99": float(np.percentile(max_m, 99)),
        },
    }
    n = max(len(generated), 1)
    for threshold in thresholds_m:
        out[f"same_sample_all_points_within_{threshold:g}m_rate"] = float(np.mean(max_m <= threshold))
        out[f"same_sample_mean_distance_within_{threshold:g}m_rate"] = float(np.mean(mean_m <= threshold))
    out["num_pairs_checked"] = int(n)
    return out


def sequence_overlap_summary(generated: np.ndarray, target: np.ndarray, train_memory: np.ndarray | None, grid_size: int) -> dict:
    bounds = infer_bounds(target.astype(np.float32), generated.astype(np.float32), padding_ratio=0.0)
    gen_hashes = [compact_route_hash(traj, bounds, grid_size) for traj in generated]
    target_hashes = [compact_route_hash(traj, bounds, grid_size) for traj in target]
    same_sample_exact = sum(gh == th for gh, th in zip(gen_hashes, target_hashes))
    test_hash_set = set(target_hashes)

    out = {
        "grid_size": int(grid_size),
        "same_sample_compact_cell_sequence_exact_rate": float(same_sample_exact / max(len(gen_hashes), 1)),
        "same_sample_compact_cell_sequence_exact_count": int(same_sample_exact),
        "generated_sequence_matches_any_test_sequence_rate": float(
            sum(gh in test_hash_set for gh in gen_hashes) / max(len(gen_hashes), 1)
        ),
        "generated_sequence_matches_any_test_sequence_count": int(sum(gh in test_hash_set for gh in gen_hashes)),
        "unique_generated_compact_sequences": int(len(set(gen_hashes))),
        "unique_test_compact_sequences": int(len(test_hash_set)),
    }

    if train_memory is not None:
        train_hashes = {compact_route_hash(traj, bounds, grid_size) for traj in train_memory}
        train_matches = sum(gh in train_hashes for gh in gen_hashes)
        out.update(
            {
                "allowed_train_memory_sequences_checked": int(len(train_memory)),
                "generated_sequence_matches_allowed_train_memory_rate": float(train_matches / max(len(gen_hashes), 1)),
                "generated_sequence_matches_allowed_train_memory_count": int(train_matches),
                "unique_allowed_train_memory_compact_sequences": int(len(train_hashes)),
            }
        )
    return out


def build_audit(args: argparse.Namespace) -> dict:
    generated_payload = np.load(args.generated_npz)
    generated = np.asarray(generated_payload[args.generated_key], dtype=np.float32)
    if args.generated_real_key and args.generated_real_key in generated_payload.files:
        test_real = np.asarray(generated_payload[args.generated_real_key], dtype=np.float32)
    else:
        test_real = load_npz_key(args.test_real_npz, args.real_key).astype(np.float32)

    n = len(generated)
    if args.max_samples:
        n = min(n, int(args.max_samples))
        generated = generated[:n]
        test_real = test_real[:n]

    train_ids = load_npz_key(args.train_real_npz, args.id_key)
    test_ids = load_npz_key(args.test_real_npz, args.id_key)
    train_base = load_npz_key(args.train_real_npz, args.base_key) if args.base_key else np.array([], dtype=np.int64)
    test_base = load_npz_key(args.test_real_npz, args.base_key) if args.base_key else np.array([], dtype=np.int64)

    train_memory = None
    if args.train_memory_samples != 0:
        train_all = load_npz_key(args.train_real_npz, args.real_key).astype(np.float32)
        limit = len(train_all) if args.train_memory_samples < 0 else min(int(args.train_memory_samples), len(train_all))
        train_memory = train_all[:limit]

    split = {
        "train_samples": int(len(train_ids)),
        "test_samples": int(len(test_ids)),
        "sample_id_overlap_count": int(len(set(map(int, train_ids)).intersection(map(int, test_ids)))),
    }
    if len(train_base) and len(test_base):
        split["base_index_overlap_count"] = int(len(set(map(int, train_base)).intersection(map(int, test_base))))

    audit = {
        "dataset": args.dataset,
        "protocol": "OD-chord-only condition with train-permitted route memory",
        "input_jsonl": str(args.input_jsonl),
        "generated_npz": str(args.generated_npz),
        "generated_key": args.generated_key,
        "split_separation": split,
        "reference_condition_guard": summarize_jsonl_guard(args.input_jsonl, int(args.max_records)),
        "same_sample_copy_distance": copy_distance_summary(
            generated,
            test_real,
            thresholds_m=[float(x) for x in args.copy_thresholds_m],
        ),
        "compact_route_sequence_overlap": sequence_overlap_summary(
            generated,
            test_real,
            train_memory,
            int(args.grid_size),
        ),
        "interpretation": {
            "forbidden_source": "validation/test middle-route geometry",
            "allowed_source": "training-split route memory and endpoint-derived OD chord",
            "direct_copy_test": "same-sample point-level and compact-cell exact-copy checks",
            "train_memory_reuse_note": (
                "A generated sequence matching train memory is not leakage under this protocol; "
                "it is reported separately because Mode A explicitly permits train-split route evidence."
            ),
        },
    }
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit leakage guards and same-sample copy risk for OD-chord trajectory generation.")
    parser.add_argument("--dataset", default="porto")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--train-real-npz", type=Path, required=True)
    parser.add_argument("--test-real-npz", type=Path, required=True)
    parser.add_argument("--generated-npz", type=Path, required=True)
    parser.add_argument("--generated-key", default="generated_trajectories")
    parser.add_argument("--generated-real-key", default="real_trajectories")
    parser.add_argument("--real-key", default="real_trajectories")
    parser.add_argument("--id-key", default="sample_id")
    parser.add_argument("--base-key", default="base_index")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--train-memory-samples", type=int, default=50000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--copy-thresholds-m", type=float, nargs="+", default=[0.1, 1.0, 5.0])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit = build_audit(args)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
