#!/usr/bin/env bash
set -euo pipefail

PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
case "$SCRIPT_SOURCE" in
  /*) SCRIPT_PATH="$SCRIPT_SOURCE" ;;
  *) SCRIPT_PATH="$(pwd -P)/$SCRIPT_SOURCE" ;;
esac
REPO_ROOT="${REPO_ROOT:-/root/LangDirectTraj}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_SAMPLES="${NUM_SAMPLES:-100000}"
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
DEVICE="${DEVICE:-cuda}"
MEMORY_SAMPLES="${MEMORY_SAMPLES:-50000}"
PRIOR_SAMPLES="${PRIOR_SAMPLES:-10000}"
GRID_NUM="${GRID_NUM:-16}"
LENGTH_BINS="${LENGTH_BINS:-32}"
OUT_OF_BOUNDS_POLICY="${OUT_OF_BOUNDS_POLICY:-overflow}"

CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/outputs/paper_full/porto_fixed16_k_sweep_seed2026/k4/model.pt}"
BOUNDS_JSON="${BOUNDS_JSON:-$REPO_ROOT/DiffTraj/data/porto_od_only/porto_global_stats.json}"
BOUNDS_SOURCE="${BOUNDS_SOURCE:-preprocessing_global_stats:porto:DiffTraj/data/porto_od_only/porto_global_stats.json}"
REAL_ROOT="$REPO_ROOT/outputs/road_full/real"
META_ROOT="$REPO_ROOT/data/proto/porto_od_chord_compact_v1/instruction_data"
TRAIN_REAL="$REAL_ROOT/porto_train_real_trajectories.npz"
VAL_REAL="$REAL_ROOT/porto_val_real_trajectories.npz"
TRAIN_META="$META_ROOT/porto_instruction_meta.train.jsonl"
VAL_META="$META_ROOT/porto_instruction_meta.val.jsonl"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/paper_full/porto_fixed16_k4_final/downstream}"
CHUNK_ROOT="$OUTPUT_ROOT/chunks"
RAW_ROOT="$OUTPUT_ROOT/raw"
LOG_ROOT="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_ROOT/pipeline.log"
PID_FILE="$LOG_ROOT/pipeline.pid"
FINAL_NPZ="$OUTPUT_ROOT/hdm_k4_train_100k.npz"
MANIFEST="$OUTPUT_ROOT/hdm_k4_train_100k_manifest.json"
PROTOCOL_MANIFEST="$OUTPUT_ROOT/protocol_manifest.json"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_porto_k4_followup_downstream.sh MODE

Modes:
  check    Verify all inputs, alignment, and the OD-chord leakage guards.
  start    Start the resumable 100k export in the background.
  run      Run the export in the foreground (used internally by start).
  logs     Follow the persistent log file.
  status   Show whether the background pipeline is running.
  summary  Print the final manifest and output file sizes.
  kill     Stop the background pipeline.

Optional environment variables:
  REPO_ROOT, PYTHON_BIN, NUM_SAMPLES, CHUNK_SIZE, DEVICE, CHECKPOINT,
  OUTPUT_ROOT, MEMORY_SAMPLES, PRIOR_SAMPLES, GRID_NUM, LENGTH_BINS,
  BOUNDS_JSON, BOUNDS_SOURCE, OUT_OF_BOUNDS_POLICY
EOF
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required file is missing: $path" >&2
    exit 1
  fi
}

precheck() {
  require_file "$CHECKPOINT"
  require_file "$BOUNDS_JSON"
  require_file "$TRAIN_REAL"
  require_file "$VAL_REAL"
  require_file "$TRAIN_META"
  require_file "$VAL_META"
  require_file "$REPO_ROOT/scripts/train_semantic_route_memory_reranker.py"

  "$PYTHON_BIN" - "$TRAIN_REAL" "$TRAIN_META" "$NUM_SAMPLES" "$CHECKPOINT" \
    "$GRID_NUM" "$LENGTH_BINS" "$OUT_OF_BOUNDS_POLICY" "$BOUNDS_JSON" "$BOUNDS_SOURCE" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import torch

real_path = Path(sys.argv[1])
meta_path = Path(sys.argv[2])
n = int(sys.argv[3])
checkpoint = Path(sys.argv[4])
grid_num = int(sys.argv[5])
length_bins = int(sys.argv[6])
out_of_bounds_policy = sys.argv[7]
bounds_json = Path(sys.argv[8])
bounds_source = sys.argv[9]

payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
saved_args = payload.get("args", {})
if not isinstance(saved_args, dict):
    saved_args = vars(saved_args)
expected = {
    "candidate_topm": 4,
    "grid_num": grid_num,
    "length_bins": length_bins,
    "out_of_bounds_policy": out_of_bounds_policy,
}
mismatches = {
    key: {"expected": value, "checkpoint": saved_args.get(key)}
    for key, value in expected.items()
    if saved_args.get(key) != value
}
if mismatches:
    raise SystemExit(f"Checkpoint protocol mismatch: {mismatches}")
saved_bounds_source = str(saved_args.get("bounds_source", ""))
if saved_bounds_source != bounds_source:
    raise SystemExit(
        "Checkpoint bounds source mismatch: "
        f"expected={bounds_source!r}, checkpoint={saved_bounds_source!r}"
    )
if not bounds_json.is_file():
    raise SystemExit(f"Missing fixed bounds JSON: {bounds_json}")

with np.load(real_path, allow_pickle=False) as payload:
    required = {"real_trajectories", "sample_id"}
    missing = required.difference(payload.files)
    if missing:
        raise SystemExit(f"{real_path}: missing keys {sorted(missing)}")
    real = np.asarray(payload["real_trajectories"][:n], dtype=np.float32)
    sample_ids = np.asarray(payload["sample_id"][:n], dtype=np.int64)

if len(real) != n or len(sample_ids) != n:
    raise SystemExit(f"Canonical train split has only {len(real):,} rows; expected {n:,}")
if real.ndim != 3 or real.shape[-1] != 2:
    raise SystemExit(f"Invalid canonical trajectory shape: {real.shape}")
if not np.isfinite(real).all():
    raise SystemExit("Canonical train trajectories contain NaN or Inf")
if len(np.unique(sample_ids)) != n:
    raise SystemExit("Canonical first-100k sample IDs are not unique")

meta_ids = []
guard_failures = []
with meta_path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        meta_ids.append(int(record["sample_id"]))
        guard = record.get("leakage_guard", {})
        if (
            record.get("reference_protocol") != "od_chord_only"
            or record.get("condition_source") != "origin_destination_chord"
            or guard.get("reference_uses_gt_middle_geometry") is not False
            or guard.get("reference_uses_only_origin_destination") is not True
        ):
            guard_failures.append((line_number, int(record["sample_id"])))
        if len(meta_ids) == n:
            break

meta_ids = np.asarray(meta_ids, dtype=np.int64)
if len(meta_ids) != n:
    raise SystemExit(f"Metadata has only {len(meta_ids):,} usable rows; expected {n:,}")
if not np.array_equal(meta_ids, sample_ids):
    mismatch = np.flatnonzero(meta_ids != sample_ids)
    first = int(mismatch[0]) if len(mismatch) else -1
    raise SystemExit(
        "Metadata/canonical sample order mismatch at row "
        f"{first}: meta={int(meta_ids[first])}, real={int(sample_ids[first])}"
    )
if guard_failures:
    raise SystemExit(f"OD-chord guard failed for {len(guard_failures):,} rows; first={guard_failures[:3]}")

print("PRECHECK PASSED")
print(f"Checkpoint: {checkpoint}")
print(f"Fixed grid: {grid_num}x{grid_num}, length_bins={length_bins}, policy={out_of_bounds_policy}")
print(f"Bounds source: {bounds_source}")
print(f"Canonical train rows: {len(sample_ids):,}")
print(f"Trajectory shape: {tuple(real.shape)}")
print(f"Sample IDs: {int(sample_ids[0])} ... {int(sample_ids[-1])}")
print("OD-chord leakage guards: 100%")
PY
}

prepare_protocol_manifest() {
  "$PYTHON_BIN" - "$PROTOCOL_MANIFEST" "$CHECKPOINT" "$TRAIN_REAL" "$NUM_SAMPLES" \
    "$CHUNK_SIZE" "$GRID_NUM" "$LENGTH_BINS" "$OUT_OF_BOUNDS_POLICY" \
    "$BOUNDS_JSON" "$BOUNDS_SOURCE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    manifest_path,
    checkpoint,
    train_real,
    num_samples,
    chunk_size,
    grid_num,
    length_bins,
    out_of_bounds_policy,
    bounds_json,
    bounds_source,
) = sys.argv[1:]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

checkpoint = Path(checkpoint).resolve()
bounds_json = Path(bounds_json).resolve()
desired = {
    "dataset": "porto",
    "split": "train",
    "num_samples": int(num_samples),
    "chunk_size": int(chunk_size),
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": sha256(checkpoint),
    "candidate_topm": 4,
    "seed": 2026,
    "grid_num": int(grid_num),
    "length_bins": int(length_bins),
    "out_of_bounds_policy": out_of_bounds_policy,
    "bounds_json": str(bounds_json),
    "bounds_source": bounds_source,
    "canonical_real_path": str(Path(train_real).resolve()),
    "selection_policy": "weighted_sum",
    "semantic_weight": 0.0,
    "refinement": False,
    "postprocessing": False,
}

manifest_path = Path(manifest_path)
if manifest_path.exists():
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing != desired:
        raise SystemExit(
            "Existing chunks belong to a different protocol. Use a new OUTPUT_ROOT "
            "or remove that output directory after verifying its absolute path."
        )
    print(f"PROTOCOL MATCHED: {manifest_path}")
else:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(desired, indent=2), encoding="utf-8")
    print(f"PROTOCOL LOCKED: {manifest_path}")
PY
}

compact_chunk() {
  local start="$1"
  local count="$2"
  local raw_npz="$3"
  local compact_npz="$4"

  "$PYTHON_BIN" - "$TRAIN_REAL" "$raw_npz" "$compact_npz" "$start" "$count" <<'PY'
import sys
from pathlib import Path

import numpy as np

real_path, raw_path, output_path = map(Path, sys.argv[1:4])
start, count = map(int, sys.argv[4:6])

with np.load(real_path, allow_pickle=False) as canonical:
    expected_ids = np.asarray(canonical["sample_id"][start : start + count], dtype=np.int64)
    expected_real = np.asarray(canonical["real_trajectories"][start : start + count], dtype=np.float32)
with np.load(raw_path, allow_pickle=False) as raw:
    ids = np.asarray(raw["sample_id"], dtype=np.int64)
    generated = np.asarray(raw["generated_trajectories"], dtype=np.float32)
    raw_real = np.asarray(raw["real_trajectories"], dtype=np.float32)

if not np.array_equal(ids, expected_ids):
    raise SystemExit(f"Chunk {start}: sample_id alignment failed")
if not np.array_equal(raw_real, expected_real):
    raise SystemExit(f"Chunk {start}: canonical real trajectory alignment failed")
if generated.shape != expected_real.shape:
    raise SystemExit(f"Chunk {start}: generated shape {generated.shape} != {expected_real.shape}")
if not np.isfinite(generated).all():
    raise SystemExit(f"Chunk {start}: generated trajectories contain NaN or Inf")

endpoint_error = max(
    float(np.max(np.abs(generated[:, 0] - expected_real[:, 0]))),
    float(np.max(np.abs(generated[:, -1] - expected_real[:, -1]))),
)
if endpoint_error > 1e-5:
    raise SystemExit(f"Chunk {start}: endpoint max error {endpoint_error:.8g} exceeds 1e-5 degrees")

output_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    output_path,
    sample_id=ids.astype(np.int64),
    generated_trajectories=generated.astype(np.float32),
)
print(
    f"CHUNK PASSED: start={start:,}, n={count:,}, "
    f"ids={int(ids[0])}...{int(ids[-1])}, endpoint_max_error={endpoint_error:.3g}",
    flush=True,
)
PY
}

merge_chunks() {
  "$PYTHON_BIN" - "$TRAIN_REAL" "$CHUNK_ROOT" "$FINAL_NPZ" "$MANIFEST" \
    "$CHECKPOINT" "$PROTOCOL_MANIFEST" "$NUM_SAMPLES" "$CHUNK_SIZE" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

real_path, chunk_root, final_path, manifest_path, checkpoint, protocol_path = map(Path, sys.argv[1:7])
n, chunk_size = map(int, sys.argv[7:9])
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

with np.load(real_path, allow_pickle=False) as canonical:
    expected_ids = np.asarray(canonical["sample_id"][:n], dtype=np.int64)
    expected_shape = tuple(canonical["real_trajectories"][:n].shape)

all_ids = []
all_generated = []
for start in range(0, n, chunk_size):
    count = min(chunk_size, n - start)
    chunk_path = chunk_root / f"chunk_{start:06d}_{start + count - 1:06d}.npz"
    if not chunk_path.is_file():
        raise SystemExit(f"Missing completed chunk: {chunk_path}")
    with np.load(chunk_path, allow_pickle=False) as payload:
        all_ids.append(np.asarray(payload["sample_id"], dtype=np.int64))
        all_generated.append(np.asarray(payload["generated_trajectories"], dtype=np.float32))

sample_ids = np.concatenate(all_ids, axis=0)
generated = np.concatenate(all_generated, axis=0)
if not np.array_equal(sample_ids, expected_ids):
    raise SystemExit("Merged sample IDs do not match canonical train order")
if generated.shape != expected_shape:
    raise SystemExit(f"Merged shape {generated.shape} != canonical shape {expected_shape}")
if len(np.unique(sample_ids)) != n:
    raise SystemExit("Merged sample IDs contain duplicates")

final_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    final_path,
    sample_id=sample_ids.astype(np.int64),
    generated_trajectories=generated.astype(np.float32),
)

id_bytes = sample_ids.astype("<i8", copy=False).tobytes(order="C")
manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Final fixed-16x16 HDM-Traj K=4 Porto train source for the fixed-100k downstream utility experiment",
    "dataset": "porto",
    "split": "train",
    "num_samples": int(n),
    "sample_selection": f"first_{n}_canonical_train_samples",
    "canonical_real_path": str(real_path),
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": protocol["checkpoint_sha256"],
    "candidate_topm": 4,
    "feature_mode": "od",
    "seed": 2026,
    "conditioning_protocol": "od_chord_only",
    "candidate_selection": "independent_per_sample",
    "selection_policy": "weighted_sum",
    "semantic_weight": 0.0,
    "semantic_fallback_policy": "none",
    "selection_target_length_weight": 0.0,
    "refine_target_length_weight": 0.0,
    "batch_refinement": False,
    "postprocessing": False,
    "grid_num": protocol["grid_num"],
    "length_bins": protocol["length_bins"],
    "out_of_bounds_policy": protocol["out_of_bounds_policy"],
    "bounds_json": protocol["bounds_json"],
    "bounds_source": protocol["bounds_source"],
    "trajectory_shape": list(map(int, generated.shape)),
    "trajectory_dtype": str(generated.dtype),
    "first_sample_id": int(sample_ids[0]),
    "last_sample_id": int(sample_ids[-1]),
    "sample_id_sha256_little_endian_int64": hashlib.sha256(id_bytes).hexdigest(),
    "chunk_size": int(chunk_size),
    "num_chunks": int(len(all_ids)),
    "prediction_path": str(final_path),
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("FINAL EXPORT PASSED", flush=True)
print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
PY
}

run_pipeline() {
  cd "$REPO_ROOT"
  mkdir -p "$CHUNK_ROOT" "$RAW_ROOT" "$LOG_ROOT"
  precheck
  prepare_protocol_manifest

  local start count end chunk_name compact_npz job_root raw_npz
  for ((start=0; start<NUM_SAMPLES; start+=CHUNK_SIZE)); do
    count="$CHUNK_SIZE"
    if (( start + count > NUM_SAMPLES )); then
      count=$((NUM_SAMPLES - start))
    fi
    end=$((start + count - 1))
    chunk_name="chunk_$(printf '%06d' "$start")_$(printf '%06d' "$end")"
    compact_npz="$CHUNK_ROOT/$chunk_name.npz"
    job_root="$RAW_ROOT/$chunk_name"
    raw_npz="$job_root/export/semantic_reranker_trajectories.npz"

    if [[ -f "$compact_npz" ]]; then
      echo "[$(date '+%F %T')] SKIP completed $chunk_name"
      continue
    fi

    echo "================================================================"
    echo "[$(date '+%F %T')] GENERATE $chunk_name ($count trajectories)"
    echo "================================================================"
    rm -rf -- "$job_root"
    mkdir -p "$job_root"

    "$PYTHON_BIN" -u scripts/train_semantic_route_memory_reranker.py \
      --memory-real-npz "$TRAIN_REAL" \
      --train-query-real-npz "$TRAIN_REAL" \
      --val-real-npz "$VAL_REAL" \
      --test-real-npz "$TRAIN_REAL" \
      --train-input "$TRAIN_META" \
      --val-input "$VAL_META" \
      --test-input "$TRAIN_META" \
      --val-counterfactual-input "$VAL_META" \
      --test-counterfactual-input "$TRAIN_META" \
      --memory-samples "$MEMORY_SAMPLES" \
      --train-query-samples 1 \
      --prior-samples "$PRIOR_SAMPLES" \
      --candidate-topm 4 \
      --feature-mode od \
      --retrieval-method kdtree \
      --target-length-scale 1.15 \
      --target-length-knn 16 \
      --target-length-weight 10.0 \
      --grid-num "$GRID_NUM" \
      --od-grid 6 \
      --length-bins "$LENGTH_BINS" \
      --bounds-json "$BOUNDS_JSON" \
      --bounds-source "$BOUNDS_SOURCE" \
      --out-of-bounds-policy "$OUT_OF_BOUNDS_POLICY" \
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
      --seed 2026 \
      --device "$DEVICE" \
      --skip-table3 \
      --skip-table4 \
      --eval-only \
      --checkpoint "$CHECKPOINT" \
      --output-dir "$job_root" \
      --eval-split test \
      --test-output-name export \
      --eval-start "$start" \
      --eval-samples "$count" \
      --candidate-cache-chunk-size 2048

    require_file "$raw_npz"
    compact_chunk "$start" "$count" "$raw_npz" "$compact_npz"
    rm -rf -- "$job_root"
    echo "[$(date '+%F %T')] SAVED $compact_npz"
  done

  merge_chunks
  echo "[$(date '+%F %T')] PIPELINE COMPLETE"
  echo "Download these two files:"
  echo "  $FINAL_NPZ"
  echo "  $MANIFEST"
}

start_background() {
  mkdir -p "$LOG_ROOT"
  if [[ -f "$PID_FILE" ]]; then
    local old_pid
    old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Pipeline is already running: PID $old_pid"
      echo "Log: $LOG_FILE"
      exit 0
    fi
  fi
  : > "$LOG_FILE"
  nohup bash "$SCRIPT_PATH" run > "$LOG_FILE" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "Pipeline started: PID $pid"
  echo "Log: $LOG_FILE"
  echo "Follow: bash $SCRIPT_PATH logs"
}

show_status() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file: $PID_FILE"
    return 1
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "RUNNING: PID $pid"
    ps -o pid,ppid,etime,%cpu,%mem,cmd -p "$pid" || true
  else
    echo "NOT RUNNING: stale PID $pid"
    return 1
  fi
}

kill_pipeline() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file: $PID_FILE"
    return 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    sleep 2
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
    echo "Stopped pipeline PID $pid"
  else
    echo "Pipeline PID $pid is not running"
  fi
  rm -f -- "$PID_FILE"
}

show_summary() {
  require_file "$FINAL_NPZ"
  require_file "$MANIFEST"
  cat "$MANIFEST"
  echo
  ls -lh "$FINAL_NPZ" "$MANIFEST"
}

mode="${1:-}"
case "$mode" in
  check)
    cd "$REPO_ROOT"
    precheck
    ;;
  start)
    start_background
    ;;
  run)
    run_pipeline
    ;;
  logs)
    mkdir -p "$LOG_ROOT"
    touch "$LOG_FILE"
    tail -n 100 -F "$LOG_FILE"
    ;;
  status)
    show_status
    ;;
  summary)
    show_summary
    ;;
  kill)
    kill_pipeline
    ;;
  *)
    usage
    exit 2
    ;;
esac
