#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-check}"
RAW_PATH="${2:-${ROME_RAW_PATH:-/root/datasets/Rome/taxi_february.txt}}"
DEVICE="${DEVICE:-cuda}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ROME_ROOT="outputs/road_full/rome"
INSTRUCTION_ROOT="$ROME_ROOT/instruction_data"
REAL_ROOT="$ROME_ROOT/real"
TRAIN_META="$INSTRUCTION_ROOT/rome_instruction_meta.train.jsonl"
VAL_META="$INSTRUCTION_ROOT/rome_instruction_meta.val.jsonl"
TEST_META="$INSTRUCTION_ROOT/rome_instruction_meta.test.jsonl"
TRAIN_REAL="$REAL_ROOT/rome_train_real_trajectories.npz"
VAL_REAL="$REAL_ROOT/rome_val_real_trajectories.npz"
TEST_REAL="$REAL_ROOT/rome_test_real_trajectories.npz"
RUN_ROOT="outputs/paper_full/rome_table1"
TRAIN_OUTPUT="$RUN_ROOT/selector"
TEST_OUTPUT="$RUN_ROOT/eval_test"
CACHE_ROOT="$RUN_ROOT/cache"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing $2: $1" >&2
    exit 1
  fi
}

show_inputs() {
  echo "Rome raw data:       $RAW_PATH"
  echo "Processed data root: $ROME_ROOT"
  echo "Final metrics:       $TEST_OUTPUT/test_full/table1_metrics.json"
  echo "Matched protocol:    candidate_topm=16 memory=50000 prior=10000 seed=7"
  require_file "$RAW_PATH" "Rome raw dataset"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
  fi
}

require_processed() {
  require_file "$TRAIN_META" "Rome train metadata"
  require_file "$VAL_META" "Rome validation metadata"
  require_file "$TEST_META" "Rome test metadata"
  require_file "$TRAIN_REAL" "Rome train trajectories"
  require_file "$VAL_REAL" "Rome validation trajectories"
  require_file "$TEST_REAL" "Rome test trajectories"
}

run_preprocess() {
  if [[ -d "$ROME_ROOT" && "${OVERWRITE_PREPROCESS:-0}" != "1" ]]; then
    echo "$ROME_ROOT already exists. Verify it or set OVERWRITE_PREPROCESS=1." >&2
    exit 1
  fi
  args=(
    -u -m langdirecttraj.data.preprocess_public_taxi_raw
    --dataset rome
    --raw-path "$RAW_PATH"
    --output-dir outputs/road_full
    --max-samples 0
    --sample-strategy sequential
    --split-ratios 0.8,0.1,0.1
    --split-seed 20260617
    --target-len 200
    --min-traj-len 20
    --min-distance-km 0.5
    --max-duration-min 180
    --max-speed-kmh 140
    --max-gap-seconds 600
    --storage-format compact
  )
  if [[ "${OVERWRITE_PREPROCESS:-0}" == "1" ]]; then
    args+=(--overwrite)
  fi
  python "${args[@]}"
}

common_selector_args=(
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
  --candidate-topm 16
  --feature-mode od_len_angle
  --retrieval-method kdtree
  --target-length-scale 1.15
  --target-length-knn 16
  --target-length-weight 10.0
  --grid-num 32
  --od-grid 6
  --length-bins 32
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
  --batch-refine-cycles 3
  --batch-density-refine-penalty 0.1
  --batch-length-refine-penalty 4.0
  --batch-refine-topk 32
  --batch-density-hot-cells 64
  --batch-refine-length-target train
  --batch-refine-score-source selection
  --hidden-dim 192
  --batch-size 256
  --seed 7
  --device "$DEVICE"
  --skip-table3
  --skip-table4
)

run_train() {
  require_processed
  mkdir -p "$CACHE_ROOT" "$RUN_ROOT/logs"
  python -u scripts/train_semantic_route_memory_reranker.py \
    "${common_selector_args[@]}" \
    --output-dir "$TRAIN_OUTPUT" \
    --eval-split val \
    --val-output-name val_10000 \
    --eval-samples 10000 \
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
    --train-route-memory-candidate-cache "$CACHE_ROOT/train_top16_candidates.npz" \
    --train-feature-cache "$CACHE_ROOT/train_top16_features" \
    --train-feature-chunk-size 1024 \
    --candidate-cache-chunk-size 2048 \
    --epochs 80 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    2>&1 | tee "$RUN_ROOT/logs/train.log"
}

run_test() {
  require_processed
  require_file "$TRAIN_OUTPUT/model.pt" "Rome selector checkpoint"
  mkdir -p "$CACHE_ROOT" "$RUN_ROOT/logs"
  python -u scripts/train_semantic_route_memory_reranker.py \
    "${common_selector_args[@]}" \
    --eval-only \
    --checkpoint "$TRAIN_OUTPUT/model.pt" \
    --output-dir "$TEST_OUTPUT" \
    --eval-split test \
    --test-output-name test_full \
    --eval-samples 0 \
    --route-memory-candidate-cache "$CACHE_ROOT/test_full_top16_candidates.npz" \
    --candidate-cache-chunk-size 2048 \
    2>&1 | tee "$RUN_ROOT/logs/test.log"
}

run_od_chord() {
  require_processed
  output="$RUN_ROOT/od_chord"
  mkdir -p "$output"
  python -u -m langdirecttraj.data.export_baseline_dataset \
    --dataset rome \
    --input "$TEST_META" \
    --output-dir "$output" \
    --max-samples 0 \
    --target-len 200 \
    --grid-size 64 \
    --continuous-mode references \
    --no-compress
  python -u -m langdirecttraj.eval.distribution_metrics \
    --real "$TEST_REAL" \
    --generated "$output/rome/rome_continuous_eval.npz" \
    --real-key real_trajectories \
    --generated-key references \
    --grid-num 32 \
    --length-bins 32 \
    --output "$output/table1_metrics.json"
}

show_inputs
case "${STAGE,,}" in
  check) ;;
  preprocess) run_preprocess ;;
  train) run_train ;;
  test) run_test ;;
  odchord) run_od_chord ;;
  all)
    run_preprocess
    run_train
    run_test
    run_od_chord
    ;;
  *)
    echo "Unknown stage: $STAGE (use check, preprocess, train, test, odchord, or all)" >&2
    exit 2
    ;;
esac
