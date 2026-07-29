#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

LOG_DIR="logs/v42_scratch_curriculum"
DONE_DIR="$LOG_DIR/done"
RESULT_DIR="results/metrics/v42_scratch_curriculum"
FOUNDATION_DIR="checkpoints/continuous_adaptive_tf_mamba_v42_scratch_foundation"
ADAPTIVE_DIR="checkpoints/continuous_adaptive_tf_mamba_v42_scratch_adaptive"
TEST_METADATA="data/processed/voicebank_demand/metadata/test.csv"
mkdir -p "$LOG_DIR" "$DONE_DIR" "$RESULT_DIR"

exec 9>"$LOG_DIR/pipeline.lock"
if ! flock -n 9; then
    echo "Another V4.2 scratch curriculum is already running." >&2
    exit 1
fi

run_step() {
    local name="$1"
    shift
    echo
    echo "[$(date --iso-8601=seconds)] START $name"
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    echo "[$(date --iso-8601=seconds)] DONE  $name"
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file is missing: $1" >&2
        exit 1
    fi
}

train_stage() {
    local name="$1"
    local config="$2"
    local checkpoint_dir="$3"
    local done_marker="$DONE_DIR/${name}.done"
    local resume_arguments=()
    if [[ -f "$done_marker" ]]; then
        echo "SKIP $name: completion marker exists"
        return
    fi
    if [[ -f "$checkpoint_dir/latest.pt" ]]; then
        resume_arguments=(--resume "$checkpoint_dir/latest.pt")
        echo "RESUME $name from $checkpoint_dir/latest.pt"
    fi
    run_step "$name" \
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/train.py --config "$config" --device cuda \
        "${resume_arguments[@]}"
    require_file "$checkpoint_dir/best.pt"
    touch "$done_marker"
}

train_stage \
    train_foundation \
    configs/v4/train_v42_scratch_foundation.yaml \
    "$FOUNDATION_DIR"

train_stage \
    train_adaptive \
    configs/v4/train_v42_scratch_adaptive.yaml \
    "$ADAPTIVE_DIR"

FOUNDATION_SUMMARY="$RESULT_DIR/foundation_test_full/summary.json"
if [[ ! -f "$FOUNDATION_SUMMARY" ]]; then
    run_step evaluate_foundation \
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py \
        --checkpoint "$FOUNDATION_DIR/best.pt" \
        --metadata "$TEST_METADATA" \
        --output-dir "$RESULT_DIR/foundation_test_full" \
        --device cuda
else
    echo "SKIP foundation evaluation: summary exists"
fi

ADAPTIVE_SUMMARY="$RESULT_DIR/adaptive_test_full/summary.json"
if [[ ! -f "$ADAPTIVE_SUMMARY" ]]; then
    run_step evaluate_adaptive \
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py \
        --checkpoint "$ADAPTIVE_DIR/best.pt" \
        --metadata "$TEST_METADATA" \
        --output-dir "$RESULT_DIR/adaptive_test_full" \
        --device cuda
else
    echo "SKIP adaptive evaluation: summary exists"
fi

echo
echo "[$(date --iso-8601=seconds)] V4.2 SCRATCH CURRICULUM COMPLETE"
echo "Foundation: $FOUNDATION_SUMMARY"
echo "Adaptive:   $ADAPTIVE_SUMMARY"
