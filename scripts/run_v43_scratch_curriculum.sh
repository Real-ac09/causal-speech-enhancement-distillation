#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

LOG_DIR="logs/v43_scratch_curriculum"
DONE_DIR="$LOG_DIR/done"
RESULT_DIR="results/metrics/v43_scratch_curriculum"
FOUNDATION_DIR="checkpoints/auxiliary_gated_tf_mamba_v43_scratch_foundation"
ADAPTIVE_DIR="checkpoints/auxiliary_gated_tf_mamba_v43_scratch_adaptive"
TEST_METADATA="data/processed/voicebank_demand/metadata/test.csv"
mkdir -p "$LOG_DIR" "$DONE_DIR" "$RESULT_DIR"

exec 9>"$LOG_DIR/pipeline.lock"
if ! flock -n 9; then
    echo "Another V4.3 scratch curriculum is already running." >&2
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

train_stage() {
    local name="$1"
    local config="$2"
    local checkpoint_dir="$3"
    local resume_arguments=()
    if [[ -f "$DONE_DIR/${name}.done" ]]; then
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
    [[ -f "$checkpoint_dir/best.pt" ]] || {
        echo "Missing best checkpoint for $name" >&2
        exit 1
    }
    touch "$DONE_DIR/${name}.done"
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
    configs/v4/train_v43_scratch_foundation.yaml \
    "$FOUNDATION_DIR"

train_stage \
    train_adaptive \
    configs/v4/train_v43_scratch_adaptive.yaml \
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
    --candidate v42_fixed_hybrid=results/metrics/v42_hybrid_vq_scratch_curriculum/adaptive_test_full \
    --candidate v43_foundation="$RESULT_DIR/foundation_test_full" \
    --candidate v43_adaptive="$RESULT_DIR/adaptive_test_full" \
    --output-dir "$RESULT_DIR/comparison"

echo
echo "[$(date --iso-8601=seconds)] V4.3 SCRATCH CURRICULUM COMPLETE"
