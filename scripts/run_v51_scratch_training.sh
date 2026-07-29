#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

CONFIG="configs/v5/generated/v51_recovery_main/adamw_loss_rescue_scratch.yaml"
RUN_DIR="checkpoints/v51_optimizer_recovery/v51_recovery_main/adamw_loss_rescue_scratch"
OUTPUT="results/v51_optimizer_recovery/v51_recovery_main/scratch"
LOCKED400="data/processed/voicebank_demand/metadata/val_v5_locked_400.csv"
V43_REFERENCE="results/metrics/v5/apples_to_apples_locked400/v43_adaptive"

mkdir -p "$RUN_DIR" "$OUTPUT" logs/v51_optimizer_recovery
exec 9>logs/v51_optimizer_recovery/scratch.lock
flock -n 9 || { echo "Another V5.1 scratch run is active" >&2; exit 1; }

train_args=(python scripts/train.py --config "$CONFIG" --device cuda)
if [[ -f "$RUN_DIR/latest.pt" ]]; then
    train_args+=(--resume "$RUN_DIR/latest.pt")
fi
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" "${train_args[@]}"

BEST="$RUN_DIR/best.pt"
[[ -f "$BEST" ]] || BEST="$RUN_DIR/latest.pt"

evaluate() {
    local output="$1" phase="$2" magnitude="$3"
    if [[ ! -f "$output/summary.json" || ! -f "$output/per_file_metrics.csv" ]]; then
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
            python scripts/evaluate.py --checkpoint "$BEST" --metadata "$LOCKED400" \
            --output-dir "$output" --device cuda \
            --phase-residual-scale "$phase" --magnitude-residual-scale "$magnitude"
    fi
}

evaluate "$OUTPUT/best_deployment" 0.0 0.95
evaluate "$OUTPUT/best_native" 1.0 1.0

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/compare_evaluations.py \
    --reference "v43=$V43_REFERENCE" \
    --candidate "scratch_deployment=$OUTPUT/best_deployment" \
    --candidate "scratch_native=$OUTPUT/best_native" \
    --output-dir "$OUTPUT/comparison" --bootstrap-samples 10000

echo "Scratch training and evaluation complete: $OUTPUT/comparison/comparison.json"
