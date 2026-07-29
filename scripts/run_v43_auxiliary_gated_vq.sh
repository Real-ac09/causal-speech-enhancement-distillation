#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

LOG_DIR="logs/v43_auxiliary_gated_vq"
DONE_DIR="$LOG_DIR/done"
RESULT_DIR="results/metrics/v43_auxiliary_gated_vq"
CALIBRATION_DIR="checkpoints/auxiliary_gated_tf_mamba_v43_codebook_calibration"
TRAIN_DIR="checkpoints/auxiliary_gated_tf_mamba_v43_gate_only"
CONFIG="configs/v4/train_v43_auxiliary_gated_vq.yaml"
SOURCE="checkpoints/continuous_adaptive_tf_mamba_v42_scratch_adaptive/best.pt"
TEST_METADATA="data/processed/voicebank_demand/metadata/test.csv"
mkdir -p "$LOG_DIR" "$DONE_DIR" "$RESULT_DIR" "$CALIBRATION_DIR"

exec 9>"$LOG_DIR/pipeline.lock"
if ! flock -n 9; then
    echo "Another V4.3 auxiliary-gated VQ run is already active." >&2
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

CALIBRATED="$CALIBRATION_DIR/calibrated.pt"
if [[ ! -f "$CALIBRATED" ]]; then
    run_step calibrate_codebook \
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/calibrate_vq_codebook.py \
        --config "$CONFIG" \
        --source-checkpoint "$SOURCE" \
        --output "$CALIBRATED" \
        --max-batches 500 \
        --device cuda
else
    echo "SKIP codebook calibration: checkpoint exists"
fi

if [[ ! -f "$RESULT_DIR/calibrated_test_full/summary.json" ]]; then
    run_step evaluate_calibrated \
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py \
        --checkpoint "$CALIBRATED" \
        --metadata "$TEST_METADATA" \
        --output-dir "$RESULT_DIR/calibrated_test_full" \
        --device cuda
fi

resume_arguments=()
if [[ ! -f "$DONE_DIR/train_gate_only.done" ]]; then
    if [[ -f "$TRAIN_DIR/latest.pt" ]]; then
        resume_arguments=(--resume "$TRAIN_DIR/latest.pt")
        echo "RESUME gate-only training from $TRAIN_DIR/latest.pt"
    fi
    run_step train_gate_only \
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/train.py --config "$CONFIG" --device cuda \
        "${resume_arguments[@]}"
    [[ -f "$TRAIN_DIR/best.pt" ]] || {
        echo "Missing gate-only best checkpoint" >&2
        exit 1
    }
    touch "$DONE_DIR/train_gate_only.done"
else
    echo "SKIP gate-only training: completion marker exists"
fi

if [[ ! -f "$RESULT_DIR/gate_only_test_full/summary.json" ]]; then
    run_step evaluate_gate_only \
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py \
        --checkpoint "$TRAIN_DIR/best.pt" \
        --metadata "$TEST_METADATA" \
        --output-dir "$RESULT_DIR/gate_only_test_full" \
        --device cuda
fi

run_step compare_results \
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/compare_evaluations.py \
    --reference v42_scratch=results/metrics/v42_scratch_curriculum/adaptive_test_full \
    --candidate v41_leader=results/metrics/noise_adaptive_tf_mamba_v41_segment_vq_muon_test_full \
    --candidate calibrated="$RESULT_DIR/calibrated_test_full" \
    --candidate gate_only="$RESULT_DIR/gate_only_test_full" \
    --output-dir "$RESULT_DIR/comparison"

echo
echo "[$(date --iso-8601=seconds)] V4.3 AUXILIARY-GATED VQ COMPLETE"
