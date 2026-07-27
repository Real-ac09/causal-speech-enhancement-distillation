#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

LOG_DIR="logs/v42_hybrid_vq_scratch_curriculum"
DONE_DIR="$LOG_DIR/done"
RESULT_DIR="results/metrics/v42_hybrid_vq_scratch_curriculum"
FOUNDATION_DIR="checkpoints/continuous_adaptive_tf_mamba_v42_hybrid_vq_scratch_foundation"
ADAPTIVE_DIR="checkpoints/continuous_adaptive_tf_mamba_v42_hybrid_vq_scratch_adaptive"
TEST_METADATA="data/processed/voicebank_demand/metadata/test.csv"
mkdir -p "$LOG_DIR" "$DONE_DIR" "$RESULT_DIR"

exec 9>"$LOG_DIR/pipeline.lock"
if ! flock -n 9; then
    echo "Another V4.2 hybrid-VQ scratch curriculum is already running." >&2
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

evaluate_stage() {
    local name="$1"
    local checkpoint="$2"
    local output_dir="$3"
    if [[ -f "$output_dir/summary.json" ]]; then
        echo "SKIP $name: summary exists"
        return
    fi
    run_step "$name" \
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py \
        --checkpoint "$checkpoint" \
        --metadata "$TEST_METADATA" \
        --output-dir "$output_dir" \
        --device cuda
}

train_stage \
    train_foundation \
    configs/v4/train_v42_hybrid_vq_scratch_foundation.yaml \
    "$FOUNDATION_DIR"

train_stage \
    train_adaptive \
    configs/v4/train_v42_hybrid_vq_scratch_adaptive.yaml \
    "$ADAPTIVE_DIR"

evaluate_stage \
    evaluate_foundation \
    "$FOUNDATION_DIR/best.pt" \
    "$RESULT_DIR/foundation_test_full"

evaluate_stage \
    evaluate_adaptive \
    "$ADAPTIVE_DIR/best.pt" \
    "$RESULT_DIR/adaptive_test_full"

run_step compare_results \
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/compare_evaluations.py \
    --reference v42_scratch=results/metrics/v42_scratch_curriculum/adaptive_test_full \
    --candidate v41_leader=results/metrics/noise_adaptive_tf_mamba_v41_segment_vq_muon_test_full \
    --candidate hybrid_vq_foundation="$RESULT_DIR/foundation_test_full" \
    --candidate hybrid_vq_adaptive="$RESULT_DIR/adaptive_test_full" \
    --output-dir "$RESULT_DIR/comparison"

echo
echo "[$(date --iso-8601=seconds)] V4.2 HYBRID-VQ SCRATCH CURRICULUM COMPLETE"
echo "Results: $RESULT_DIR"
