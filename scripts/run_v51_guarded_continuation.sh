#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

CONFIG="configs/v5/generated/v51_recovery_main/adamw_1e4_loss_rescue_guarded_1000.yaml"
SOURCE_CHECKPOINT="checkpoints/v51_optimizer_recovery/v51_recovery_main/adamw_1e4_loss_rescue/latest.pt"
RUN_DIR="checkpoints/v51_optimizer_recovery/v51_recovery_main/adamw_1e4_loss_rescue_guarded_1000"
RESUME_CHECKPOINT="$RUN_DIR/seed_epoch_001.pt"
OUTPUT_ROOT="results/v51_optimizer_recovery/v51_recovery_main/guarded_1000"
REFERENCE="results/v51_optimizer_recovery/v51_recovery_main/full_utterance/source"
METADATA="data/processed/voicebank_demand/metadata/val_v51_search_100.csv"

mkdir -p "$RUN_DIR" "$OUTPUT_ROOT" logs/v51_optimizer_recovery
exec 9>logs/v51_optimizer_recovery/guarded_1000.lock
flock -n 9 || { echo "Another guarded V5.1 continuation is active" >&2; exit 1; }

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
    echo "Missing recovery checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 1
fi

# Seed the isolated run once so AdamW moments, RNG state, and epoch numbering survive.
if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
    cp --reflink=auto "$SOURCE_CHECKPOINT" "$RESUME_CHECKPOINT"
fi

if [[ ! -f "$RUN_DIR/latest.pt" ]]; then
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/train.py \
        --config "$CONFIG" \
        --device cuda \
        --epochs 2 \
        --max-train-batches 1000 \
        --max-val-batches 50 \
        --resume "$RESUME_CHECKPOINT"
fi

CANDIDATE_CHECKPOINT="$RUN_DIR/latest.pt"
CANDIDATE_OUTPUT="$OUTPUT_ROOT/candidate_phase_0_mag_0_95"
if [[ ! -f "$CANDIDATE_OUTPUT/summary.json" || ! -f "$CANDIDATE_OUTPUT/per_file_metrics.csv" ]]; then
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py \
        --checkpoint "$CANDIDATE_CHECKPOINT" \
        --metadata "$METADATA" \
        --output-dir "$CANDIDATE_OUTPUT" \
        --device cuda \
        --phase-residual-scale 0.0 \
        --magnitude-residual-scale 0.95
fi

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/compare_evaluations.py \
    --reference "source=$REFERENCE" \
    --candidate "guarded_1000=$CANDIDATE_OUTPUT" \
    --output-dir "$OUTPUT_ROOT/comparison" \
    --bootstrap-samples 10000

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import json
from pathlib import Path

root = Path("results/v51_optimizer_recovery/v51_recovery_main/guarded_1000")
comparison = json.loads((root / "comparison/comparison.json").read_text())
paired = comparison["paired_bootstrap"]["guarded_1000"]
row = next(item for item in comparison["results"] if item["name"] == "guarded_1000")
gates = {
    "pesq_positive_ci": paired["enhanced_pesq"]["ci95"][0] > 0.0,
    "si_sdr_no_harm": row["delta_enhanced_si_sdr"] >= -0.15,
    "stoi_no_harm": row["delta_enhanced_stoi"] >= -0.002,
    "estoi_no_harm": row["delta_enhanced_estoi"] >= -0.003,
}
decision = {
    "status": "passed_guarded_1000" if all(gates.values()) else "stopped_by_no_harm_gate",
    "inference": {"phase_residual_scale": 0.0, "magnitude_residual_scale": 0.95},
    "gates": gates,
    "candidate": row,
    "paired_bootstrap": paired,
}
(root / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
print(json.dumps(decision, indent=2))
PY

echo "Guarded continuation complete: $OUTPUT_ROOT/decision.json"
