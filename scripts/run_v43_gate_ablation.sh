#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

CHECKPOINT="checkpoints/auxiliary_gated_tf_mamba_v43_scratch_adaptive/best.pt"
METADATA="data/processed/voicebank_demand/metadata/test.csv"
OUTPUT_DIR="results/metrics/v43_scratch_curriculum/gate_ablation_test_full"
LOG_DIR="logs/v43_gate_ablation"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

exec 9>"$LOG_DIR/pipeline.lock"
if ! flock -n 9; then
    echo "Another V4.3 gate ablation is already running." >&2
    exit 1
fi

echo "[$(date --iso-8601=seconds)] START V4.3 gate ablation"
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/evaluate_v4_ablations.py \
    --checkpoint "$CHECKPOINT" \
    --metadata "$METADATA" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --gate-values learned off 0.01 0.05 0.1 0.25 0.5 1 \
    2>&1 | tee "$LOG_DIR/evaluate.log"
echo "[$(date --iso-8601=seconds)] DONE V4.3 gate ablation"
