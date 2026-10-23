#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-help}"
REPO_ROOT="${REPO_ROOT:-/root/LangDirectTraj}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

cd "$REPO_ROOT"

GRID_NUM="${GRID_NUM:-16}"
LENGTH_BINS="${LENGTH_BINS:-32}"
SEED="${SEED:-2026}"
K_VALUES=(${K_VALUES:-4 8 16})

RUN_ROOT="${RUN_ROOT:-outputs/paper_full/porto_fixed${GRID_NUM}_k_sweep_seed${SEED}}"
CACHE_ROOT="$RUN_ROOT/cache"
LOG_ROOT="$RUN_ROOT/logs"
LOG_FILE="$LOG_ROOT/pipeline.log"
PID_FILE="$LOG_ROOT/pipeline.pid"

REAL_ROOT="outputs/road_full/real"
META_ROOT="data/proto/porto_od_chord_compact_v1/instruction_data"
TRAIN_REAL="$REAL_ROOT/porto_train_real_trajectories.npz"
VAL_REAL="$REAL_ROOT/porto_val_real_trajectories.npz"
TEST_REAL="$REAL_ROOT/porto_test_real_trajectories.npz"
TRAIN_META="$META_ROOT/porto_instruction_meta.train.jsonl"
VAL_META="$META_ROOT/porto_instruction_meta.val.jsonl"
TEST_META="$META_ROOT/porto_instruction_meta.test.jsonl"
BOUNDS_JSON="${BOUNDS_JSON:-DiffTraj/data/porto_od_only/porto_global_stats.json}"

usage() {
  cat <<EOF
Usage:
  bash scripts/run_porto_fixed16_k_sweep.sh check
  bash scripts/run_porto_fixed16_k_sweep.sh start
  bash scripts/run_porto_fixed16_k_sweep.sh status
  bash scripts/run_porto_fixed16_k_sweep.sh logs
  bash scripts/run_porto_fixed16_k_sweep.sh summary

Environment overrides:
  GRID_NUM=16 LENGTH_BINS=32 SEED=2026 K_VALUES="4 8 16"

This reruns Porto validation model selection under one fixed city grid. Raw
candidate coordinates may be regenerated into this run directory, but train
priors, route features, validation metrics, and metrics provenance all use the
same preprocessing global_stats bounds.
EOF
}

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $REPO_ROOT/$1" >&2; exit 1; }
}

precheck() {
  require_file "scripts/precheck_fixed_grid_protocol.py"
  require_file "scripts/train_semantic_route_memory_reranker.py"
  require_file "$TRAIN_REAL"
  require_file "$VAL_REAL"
  require_file "$TEST_REAL"
  require_file "$TRAIN_META"
  require_file "$VAL_META"
  require_file "$TEST_META"
  require_file "$BOUNDS_JSON"

  "$PYTHON_BIN" scripts/precheck_fixed_grid_protocol.py \
    --city porto \
    --grid-num "$GRID_NUM" \
    --length-bins "$LENGTH_BINS" \
    --bounds-json "$BOUNDS_JSON" \
    --out-of-bounds-policy overflow
}

run_one_k() {
  local k="$1"
  local out="$RUN_ROOT/k${k}"
  local cache="$CACHE_ROOT/k${k}"
  mkdir -p "$out" "$cache"

  echo "================================================================"
  echo "[$(date '+%F %T')] Porto fixed ${GRID_NUM}x${GRID_NUM}: train/evaluate K=$k"
  echo "================================================================"

  "$PYTHON_BIN" -u scripts/train_semantic_route_memory_reranker.py \
    --memory-real-npz "$TRAIN_REAL" \
    --train-query-real-npz "$TRAIN_REAL" \
    --val-real-npz "$VAL_REAL" \
    --test-real-npz "$TEST_REAL" \
    --train-input "$TRAIN_META" \
    --val-input "$VAL_META" \
    --test-input "$TEST_META" \
    --val-counterfactual-input "$VAL_META" \
    --test-counterfactual-input "$TEST_META" \
    --memory-samples 50000 \
    --train-query-samples 0 \
    --prior-samples 10000 \
    --candidate-topm "$k" \
    --feature-mode od \
    --retrieval-method kdtree \
    --target-length-scale 1.15 \
    --target-length-knn 16 \
    --target-length-weight 10.0 \
    --grid-num "$GRID_NUM" \
    --od-grid 6 \
    --length-bins "$LENGTH_BINS" \
    --bounds-json "$BOUNDS_JSON" \
    --bounds-source "preprocessing_global_stats:porto:$BOUNDS_JSON" \
    --out-of-bounds-policy overflow \
    --min-group-count 3 \
    --global-mix 0.1 \
    --cell-weight 0.25 \
    --transition-weight 1.0 \
    --length-ratio-weight 2.0 \
    --prototype-weight 0.5 \
    --mode-index-weight 0.005 \
    --prior-score-feature \
    --table1-cost-feature \
    --disable-semantic-costs \
    --selection-policy weighted_sum \
    --selection-logit-weight 1.0 \
    --semantic-weight 0.0 \
    --semantic-fallback-policy none \
    --selection-target-length-weight 0.0 \
    --refine-target-length-weight 0.0 \
    --batch-refine-cycles 0 \
    --post-fallback-refine-cycles 0 \
    --hidden-dim 192 \
    --batch-size 256 \
    --seed "$SEED" \
    --device cuda \
    --skip-table3 \
    --skip-table4 \
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
    --train-route-memory-candidate-cache "$cache/train_candidates.npz" \
    --reuse-train-route-memory-candidate-cache \
    --train-feature-cache "$cache/train_features" \
    --reuse-train-feature-cache \
    --route-memory-candidate-cache "$cache/val_full_candidates.npz" \
    --reuse-route-memory-candidate-cache \
    --train-feature-chunk-size 1024 \
    --candidate-cache-chunk-size 2048 \
    --epochs 80 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --output-dir "$out"
}

write_manifest() {
  mkdir -p "$RUN_ROOT"
  "$PYTHON_BIN" - "$RUN_ROOT/protocol_manifest.json" "$GRID_NUM" "$LENGTH_BINS" "$SEED" "${K_VALUES[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
grid_num = int(sys.argv[2])
length_bins = int(sys.argv[3])
seed = int(sys.argv[4])
k_values = [int(x) for x in sys.argv[5:]]
payload = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset": "porto",
    "split": "validation",
    "purpose": "HDM-Traj fixed-grid K model selection",
    "candidate_budgets": k_values,
    "grid_num": grid_num,
    "length_bins": length_bins,
    "bounds_json": "DiffTraj/data/porto_od_only/porto_global_stats.json",
    "bounds_source": "preprocessing_global_stats",
    "out_of_bounds_policy": "overflow",
    "seed": seed,
    "postprocessing": False,
    "selection_rule": "choose the validation configuration with minimum Density+Route+Length unless paper text states otherwise",
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {output}")
PY
}

write_summary() {
  "$PYTHON_BIN" - "$RUN_ROOT" "$GRID_NUM" "${K_VALUES[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
grid_num = int(sys.argv[2])
k_values = [int(x) for x in sys.argv[3:]]
rows = []
for k in k_values:
    metrics_path = root / f"k{k}" / "val_full" / "table1_metrics.json"
    checkpoint_path = root / f"k{k}" / "model.pt"
    if not metrics_path.exists():
        raise SystemExit(f"Missing K={k} metrics: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key, expected in {
        "grid_num": grid_num,
        "out_of_bounds_policy": "overflow",
        "candidate_topm": k,
        "eval_batch_refine": False,
    }.items():
        if payload.get(key) != expected:
            raise SystemExit(f"K={k}: {key} expected {expected!r}, got {payload.get(key)!r}")
    score = (
        float(payload["Density Error"])
        + float(payload["Route Transition JSD"])
        + float(payload["Length Error"])
    )
    rows.append(
        {
            "K": k,
            "score_density_route_length": score,
            "num_compared": int(payload["num_compared"]),
            "Density Error": float(payload["Density Error"]),
            "Route Transition JSD": float(payload["Route Transition JSD"]),
            "Length Error": float(payload["Length Error"]),
            "Endpoint Error (m)": float(payload["Endpoint Error (m)"]),
            "OD Adherence <=1m": float(payload["OD Adherence <=1m"]),
            "checkpoint_path": str(checkpoint_path),
            "metrics_path": str(metrics_path),
        }
    )
best = min(rows, key=lambda row: row["score_density_route_length"])
summary = {
    "dataset": "porto",
    "split": "validation",
    "selection_rule": "min Density Error + Route Transition JSD + Length Error",
    "selected": best,
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
(root / "selected_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nPORTO FIXED-GRID K SWEEP")
for row in rows:
    print(
        f"K={row['K']:>2}: Density={row['Density Error']:.9f}, "
        f"Route={row['Route Transition JSD']:.9f}, "
        f"Length={row['Length Error']:.9f}, "
        f"Endpoint={row['Endpoint Error (m)']:.3f} m, "
        f"score={row['score_density_route_length']:.9f}"
    )
print(
    f"SELECTED K={best['K']}: "
    f"Density={best['Density Error']:.9f}, "
    f"Route={best['Route Transition JSD']:.9f}, "
    f"Length={best['Length Error']:.9f}"
)
print(f"Summary: {root / 'summary.json'}")
PY
}

run_pipeline() {
  precheck
  mkdir -p "$RUN_ROOT" "$CACHE_ROOT" "$LOG_ROOT"
  write_manifest
  for k in "${K_VALUES[@]}"; do
    run_one_k "$k"
  done
  write_summary
  echo "[$(date '+%F %T')] PIPELINE COMPLETED"
}

case "$MODE" in
  check)
    precheck
    ;;
  start)
    precheck
    mkdir -p "$LOG_ROOT"
    nohup bash "$0" run > "$LOG_FILE" 2>&1 &
    echo "$!" > "$PID_FILE"
    echo "Pipeline started: pid=$(cat "$PID_FILE")"
    echo "Log: $REPO_ROOT/$LOG_FILE"
    echo "Follow: bash scripts/run_porto_fixed16_k_sweep.sh logs"
    ;;
  run)
    run_pipeline
    ;;
  status)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "RUNNING pid=$(cat "$PID_FILE")"
    else
      echo "NOT RUNNING"
    fi
    ;;
  logs)
    mkdir -p "$LOG_ROOT"
    touch "$LOG_FILE"
    tail -f "$LOG_FILE"
    ;;
  summary)
    write_summary
    ;;
  *)
    usage
    exit 2
    ;;
esac
