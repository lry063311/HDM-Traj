#!/usr/bin/env bash
set -euo pipefail

CITY="${1:-}"
STAGE="${2:-check}"
SEED="${SEED:-2026}"
GRID_NUM="${GRID_NUM:-32}"
LENGTH_BINS="${LENGTH_BINS:-32}"
BOUNDS_JSON="${BOUNDS_JSON:-}"
NO_CALIBRATION="${NO_CALIBRATION:-0}"

if [[ "$CITY" != "beijing" && "$CITY" != "rome" ]]; then
  echo "Usage: $0 {beijing|rome} {check|train|test}" >&2
  exit 2
fi
if [[ "$STAGE" != "check" && "$STAGE" != "train" && "$STAGE" != "test" ]]; then
  echo "Usage: $0 {beijing|rome} {check|train|test}" >&2
  exit 2
fi
if [[ "$NO_CALIBRATION" != "0" && "$NO_CALIBRATION" != "1" ]]; then
  echo "NO_CALIBRATION must be 0 or 1" >&2
  exit 2
fi

SCRIPT_DIR="${BASH_SOURCE[0]%/*}"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="outputs/road_full/$CITY"
REAL_ROOT="$DATA_ROOT/real"
META_ROOT="$DATA_ROOT/instruction_data"
TRAIN_REAL="$REAL_ROOT/${CITY}_train_real_trajectories.npz"
VAL_REAL="$REAL_ROOT/${CITY}_val_real_trajectories.npz"
TEST_REAL="$REAL_ROOT/${CITY}_test_real_trajectories.npz"
TRAIN_META="$META_ROOT/${CITY}_instruction_meta.train.jsonl"
VAL_META="$META_ROOT/${CITY}_instruction_meta.val.jsonl"
TEST_META="$META_ROOT/${CITY}_instruction_meta.test.jsonl"
if [[ -n "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="$RUN_ROOT"
elif [[ "$NO_CALIBRATION" == "1" ]]; then
  RUN_ROOT="outputs/paper_full/${CITY}_k4_strict_od_no_cal_seed${SEED}"
else
  RUN_ROOT="outputs/paper_full/${CITY}_k4_strict_od_seed${SEED}"
fi
SELECTOR_ROOT="$RUN_ROOT/selector"
TEST_ROOT="$RUN_ROOT/final_test_seed${SEED}"
CACHE_ROOT="$RUN_ROOT/cache"

required=(
  "$TRAIN_REAL" "$VAL_REAL" "$TEST_REAL"
  "$TRAIN_META" "$VAL_META" "$TEST_META"
  "$META_ROOT/${CITY}_instruction_meta.train.compact_manifest.json"
  "$META_ROOT/${CITY}_instruction_meta.val.compact_manifest.json"
  "$META_ROOT/${CITY}_instruction_meta.test.compact_manifest.json"
  "$META_ROOT/${CITY}_base_lonlat.npy"
  "$META_ROOT/${CITY}_reference_lonlat.npy"
  "$META_ROOT/${CITY}_split_manifest.json"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
if [[ -n "$BOUNDS_JSON" ]]; then
  [[ -f "$BOUNDS_JSON" ]] || { echo "Missing required file: $BOUNDS_JSON" >&2; exit 1; }
fi

python - "$CITY" "$REAL_ROOT" "$META_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

city, real_root, meta_root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
split_ids = {}
for split in ("train", "val", "test"):
    real_path = real_root / f"{city}_{split}_real_trajectories.npz"
    with np.load(real_path, allow_pickle=False) as payload:
        required = {"real_trajectories", "sample_id"}
        missing = required.difference(payload.files)
        if missing:
            raise SystemExit(f"{real_path}: missing keys {sorted(missing)}")
        traj = payload["real_trajectories"]
        ids = np.asarray(payload["sample_id"], dtype=np.int64)
    if traj.ndim != 3 or traj.shape[1:] != (200, 2):
        raise SystemExit(f"{real_path}: expected (N, 200, 2), got {traj.shape}")
    if len(ids) != len(traj) or len(np.unique(ids)) != len(ids):
        raise SystemExit(f"{real_path}: invalid or duplicate sample_id")

    meta_path = meta_root / f"{city}_instruction_meta.{split}.jsonl"
    meta_ids = []
    with meta_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("reference_protocol") != "od_chord_only":
                raise SystemExit(f"{meta_path}:{line_no}: non-OD-chord protocol")
            if record.get("condition_source") != "origin_destination_chord":
                raise SystemExit(f"{meta_path}:{line_no}: invalid condition source")
            guard = record.get("leakage_guard", {})
            if guard.get("reference_uses_gt_middle_geometry") is not False:
                raise SystemExit(f"{meta_path}:{line_no}: middle-route leakage flag")
            if guard.get("reference_uses_only_origin_destination") is not True:
                raise SystemExit(f"{meta_path}:{line_no}: reference is not OD-only")
            meta_ids.append(int(record["sample_id"]))
    if len(meta_ids) != len(ids) or set(meta_ids) != set(ids.tolist()):
        raise SystemExit(f"{split}: metadata and real NPZ sample_id mismatch")
    split_ids[split] = set(ids.tolist())
    print(f"{city} {split}: n={len(ids):,}, shape={traj.shape}, IDs/protocol OK")

for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = split_ids[left].intersection(split_ids[right])
    if overlap:
        raise SystemExit(f"split overlap {left}/{right}: {len(overlap)}")
print(f"{city}: split disjointness and strict OD protocol PASSED")
PY

if [[ -n "$BOUNDS_JSON" ]]; then
  python scripts/precheck_fixed_grid_protocol.py \
    --city "$CITY" \
    --grid-num "$GRID_NUM" \
    --length-bins "$LENGTH_BINS" \
    --bounds-json "$BOUNDS_JSON" \
    --out-of-bounds-policy overflow
fi

if [[ "$STAGE" == "check" ]]; then
  exit 0
fi

mkdir -p "$CACHE_ROOT" "$RUN_ROOT/logs"

common_args=(
  --memory-real-npz "$TRAIN_REAL"
  --train-query-real-npz "$TRAIN_REAL"
  --val-real-npz "$VAL_REAL"
  --test-real-npz "$TEST_REAL"
  --train-input "$TRAIN_META"
  --val-input "$VAL_META"
  --test-input "$TEST_META"
  --val-counterfactual-input "$VAL_META"
  --test-counterfactual-input "$TEST_META"
  --memory-samples 50000
  --train-query-samples 0
  --prior-samples 10000
  --candidate-topm 4
  --feature-mode od
  --retrieval-method kdtree
  --target-length-scale 1.15
  --target-length-knn 16
  --target-length-weight 10.0
  --grid-num "$GRID_NUM"
  --od-grid 6
  --length-bins "$LENGTH_BINS"
  --min-group-count 3
  --global-mix 0.1
  --cell-weight 0.25
  --transition-weight 1.0
  --length-ratio-weight 2.0
  --prototype-weight 0.5
  --mode-index-weight 0.005
  --prior-score-feature
  --table1-cost-feature
  --disable-semantic-costs
  --selection-policy weighted_sum
  --selection-logit-weight 1.0
  --semantic-weight 0.0
  --semantic-fallback-policy none
  --selection-target-length-weight 0.0
  --refine-target-length-weight 0.0
  --batch-refine-cycles 0
  --post-fallback-refine-cycles 0
  --hidden-dim 192
  --batch-size 256
  --seed "$SEED"
  --device cuda
  --skip-table3
  --skip-table4
)
if [[ -n "$BOUNDS_JSON" ]]; then
  common_args+=(
    --bounds-json "$BOUNDS_JSON"
    --bounds-source "preprocessing_global_stats:${CITY}:${BOUNDS_JSON}"
    --out-of-bounds-policy overflow
  )
fi
if [[ "$NO_CALIBRATION" == "1" ]]; then
  common_args+=(--disable-length-transition-calibration)
fi

if [[ "$STAGE" == "train" ]]; then
  python -u scripts/train_semantic_route_memory_reranker.py \
    "${common_args[@]}" \
    --output-dir "$SELECTOR_ROOT" \
    --eval-split val \
    --val-output-name val_full \
    --eval-samples 0 \
    --teacher-ce-weight 1.0 \
    --teacher-soft-weight 1.0 \
    --teacher-score-weight 0.2 \
    --teacher-temperature 0.35 \
    --teacher-score-mode route_first_soft \
    --teacher-density-weight 1.0 \
    --teacher-density-marginal-weight 1.0 \
    --teacher-transition-weight 2.0 \
    --teacher-length-weight 2.5 \
    --teacher-raw-length-weight 0.0 \
    --teacher-length-marginal-weight 0.0 \
    --teacher-use-real-train-length \
    --teacher-semantic-cost-weight 0.0 \
    --student-density-cost-weight 0.5 \
    --student-transition-cost-weight 0.8 \
    --student-length-cost-weight 4.0 \
    --student-raw-length-cost-weight 1.5 \
    --stream-train-features \
    --train-route-memory-candidate-cache "$CACHE_ROOT/train_k4_candidates.npz" \
    --reuse-train-route-memory-candidate-cache \
    --train-feature-cache "$CACHE_ROOT/train_k4_features" \
    --reuse-train-feature-cache \
    --route-memory-candidate-cache "$CACHE_ROOT/val_full_k4_candidates.npz" \
    --reuse-route-memory-candidate-cache \
    --train-feature-chunk-size 1024 \
    --candidate-cache-chunk-size 2048 \
    --epochs 80 \
    --lr 0.001 \
    --weight-decay 0.0001
  exit 0
fi

CHECKPOINT="$SELECTOR_ROOT/model.pt"
[[ -f "$CHECKPOINT" ]] || { echo "Missing frozen checkpoint: $CHECKPOINT" >&2; exit 1; }
python -u scripts/train_semantic_route_memory_reranker.py \
  "${common_args[@]}" \
  --eval-only \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$TEST_ROOT" \
  --eval-split test \
  --test-output-name test_full \
  --eval-samples 0 \
  --route-memory-candidate-cache "$CACHE_ROOT/test_full_k4_candidates.npz" \
  --reuse-route-memory-candidate-cache \
  --candidate-cache-chunk-size 2048

python - "$CITY" "$TEST_REAL" "$TEST_ROOT/test_full" "$GRID_NUM" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

city, canonical_path, output_dir, expected_grid = (
    sys.argv[1],
    Path(sys.argv[2]),
    Path(sys.argv[3]),
    int(sys.argv[4]),
)
metrics = json.loads((output_dir / "table1_metrics.json").read_text(encoding="utf-8"))
with np.load(canonical_path, allow_pickle=False) as canonical, np.load(
    output_dir / "semantic_reranker_trajectories.npz", allow_pickle=False
) as prediction:
    if not np.array_equal(prediction["sample_id"], canonical["sample_id"]):
        raise SystemExit("final prediction sample_id is not canonically aligned")
    if not np.array_equal(prediction["real_trajectories"], canonical["real_trajectories"]):
        raise SystemExit("final prediction embeds a non-canonical real array")
    if prediction["generated_trajectories"].shape != canonical["real_trajectories"].shape:
        raise SystemExit("final generated/canonical shape mismatch")
assert metrics["candidate_topm"] == 4
assert metrics["grid_num"] == expected_grid
assert metrics["out_of_bounds_policy"] == "overflow"
assert metrics["bounds_source"].startswith("preprocessing_global_stats:")
assert metrics["eval_batch_refine"] is False
assert metrics["semantic_weight"] == 0.0
assert metrics["selection_target_length_weight"] == 0.0
assert metrics["refine_target_length_weight"] == 0.0
assert metrics["refined_stats"] == {}
print(
    f"{city} FINAL PASSED: n={metrics['num_compared']:,}, "
    f"Density={metrics['Density Error']:.9f}, "
    f"Route={metrics['Route Transition JSD']:.9f}, "
    f"Length={metrics['Length Error']:.9f}"
)
PY
