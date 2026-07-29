#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
CONDA_BIN="${CONDA_BIN:-conda}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

LOG_DIR="logs/v42_controlled_ablation"
DONE_DIR="$LOG_DIR/done"
RESULT_DIR="results/metrics/v42_controlled_ablation"
TEST_METADATA="data/processed/voicebank_demand/metadata/test.csv"
mkdir -p "$LOG_DIR" "$DONE_DIR" "$RESULT_DIR"

exec 9>"$LOG_DIR/pipeline.lock"
if ! flock -n 9; then
    echo "Another V4.2 controlled ablation is already running." >&2
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

run_arm() {
    local name="$1"
    local config="$2"
    local checkpoint_dir="$3"
    local output_dir="$RESULT_DIR/${name}_test_full"
    local resume_arguments=()

    if [[ ! -f "$DONE_DIR/${name}_train.done" ]]; then
        if [[ -f "$checkpoint_dir/latest.pt" ]]; then
            resume_arguments=(--resume "$checkpoint_dir/latest.pt")
            echo "RESUME $name from $checkpoint_dir/latest.pt"
        fi
        run_step "train_${name}" \
            "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
            python scripts/train.py --config "$config" --device cuda \
            "${resume_arguments[@]}"
        [[ -f "$checkpoint_dir/best.pt" ]] || {
            echo "Missing best checkpoint for $name" >&2
            exit 1
        }
        touch "$DONE_DIR/${name}_train.done"
    else
        echo "SKIP $name training: completion marker exists"
    fi

    if [[ ! -f "$output_dir/summary.json" ]]; then
        run_step "evaluate_${name}" \
            "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
            python scripts/evaluate.py \
            --checkpoint "$checkpoint_dir/best.pt" \
            --metadata "$TEST_METADATA" \
            --output-dir "$output_dir" \
            --device cuda
    else
        echo "SKIP $name evaluation: summary exists"
    fi
}

run_arm \
    perceptual_no_gan \
    configs/v4/train_v42_perceptual_no_gan.yaml \
    checkpoints/continuous_adaptive_tf_mamba_v42_perceptual_no_gan

run_arm \
    metricgan_regression \
    configs/v4/train_v42_metricgan_regression.yaml \
    checkpoints/continuous_adaptive_tf_mamba_v42_metricgan_regression

run_arm \
    hybrid_vq \
    configs/v4/train_v42_hybrid_vq_finetune.yaml \
    checkpoints/continuous_adaptive_tf_mamba_v42_hybrid_vq_finetune

echo
echo "[$(date --iso-8601=seconds)] V4.2 CONTROLLED ABLATION COMPLETE"
echo "Results: $RESULT_DIR"
