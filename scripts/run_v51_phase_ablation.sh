#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

CHECKPOINT="checkpoints/v51_optimizer_recovery/v51_recovery_main/adamw_1e4_loss_rescue/best.pt"
METADATA="data/processed/voicebank_demand/metadata/val_v51_search_100.csv"
ROOT_OUT="results/v51_optimizer_recovery/v51_recovery_main/phase_ablation"
REFERENCE="results/v51_optimizer_recovery/v51_recovery_main/full_utterance/source"
mkdir -p "$ROOT_OUT" logs/v51_optimizer_recovery
exec 9>logs/v51_optimizer_recovery/phase_ablation.lock
flock -n 9 || { echo "Another V5.1 phase ablation is active" >&2; exit 1; }

candidate_args=()
for scale in 0.00 0.25 0.50 0.75 1.00; do
    label="phase_${scale/./_}"
    output="$ROOT_OUT/$label"
    if [[ ! -f "$output/summary.json" || ! -f "$output/per_file_metrics.csv" ]]; then
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
            python scripts/evaluate.py \
            --checkpoint "$CHECKPOINT" \
            --metadata "$METADATA" \
            --output-dir "$output" \
            --device cuda \
            --phase-residual-scale "$scale"
    fi
    candidate_args+=(--candidate "$label=$output")
done

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/compare_evaluations.py \
    --reference "source=$REFERENCE" \
    "${candidate_args[@]}" \
    --output-dir "$ROOT_OUT/comparison" \
    --bootstrap-samples 10000

echo "Phase ablation complete: $ROOT_OUT/comparison/comparison.json"
