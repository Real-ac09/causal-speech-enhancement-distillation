#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"

PILOT_CONFIG="configs/v5/v52_teacher_fast_pilot.yaml"
FULL_CONFIG="configs/v5/v52_teacher_fast_full.yaml"
PILOT_DIR="checkpoints/v52_fast/teacher_pilot_1000"
FULL_DIR="checkpoints/v52_fast/teacher_full"
OUTPUT="results/v52_fast"
mkdir -p "$OUTPUT" logs/v52_fast
exec 9>logs/v52_fast/programme.lock
flock -n 9 || { echo "Another V5.2 programme is active" >&2; exit 1; }

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/check_v51_candidate.py --config "$PILOT_CONFIG" \
    --output "$OUTPUT/structural_gates.json" --device cuda

if [[ ! -f "$PILOT_DIR/latest.pt" ]]; then
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/train.py --config "$PILOT_CONFIG" --device cuda \
        --max-train-batches 1000 --max-val-batches 50
fi

if [[ ! -f "$OUTPUT/pilot_100/summary.json" ]]; then
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py --checkpoint "$PILOT_DIR/best.pt" \
        --metadata data/processed/voicebank_demand/metadata/val_v51_search_100.csv \
        --output-dir "$OUTPUT/pilot_100" --device cuda \
        --phase-residual-scale 1.0 --magnitude-residual-scale 1.0
fi

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -c '
import json
from pathlib import Path
p = Path("results/v52_fast")
s = json.loads((p / "pilot_100/summary.json").read_text())["metrics"]
g = {
    "pesq_gain": s["enhanced_pesq"] >= s["noisy_pesq"] + 0.05,
    "si_sdr_gain": s["si_sdr_improvement"] >= 1.0,
    "stoi_no_harm": s["enhanced_stoi"] >= s["noisy_stoi"] - 0.002,
    "estoi_no_harm": s["enhanced_estoi"] >= s["noisy_estoi"] - 0.003,
}
d = {"status": "promote" if all(g.values()) else "stop", "gates": g, "metrics": s}
(p / "pilot_decision.json").write_text(json.dumps(d, indent=2) + "\n")
print(json.dumps(d, indent=2))
raise SystemExit(0 if all(g.values()) else 3)
'

# The pilot is deliberately isolated. A promoted model begins a reproducible
# full scratch run rather than treating a truncated epoch as epoch one.
if [[ -f "$FULL_DIR/latest.pt" ]]; then
    train_args=(--resume "$FULL_DIR/latest.pt")
else
    train_args=()
fi
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/train.py --config "$FULL_CONFIG" --device cuda "${train_args[@]}"

BEST="$FULL_DIR/best.pt"
if [[ ! -f "$OUTPUT/full_best_locked400/summary.json" ]]; then
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
        python scripts/evaluate.py --checkpoint "$BEST" \
        --metadata data/processed/voicebank_demand/metadata/val_v5_locked_400.csv \
        --output-dir "$OUTPUT/full_best_locked400" --device cuda \
        --phase-residual-scale 1.0 --magnitude-residual-scale 1.0
fi
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/compare_evaluations.py \
    --reference v43=results/metrics/v5/apples_to_apples_locked400/v43_adaptive \
    --candidate v52=results/v52_fast/full_best_locked400 \
    --output-dir "$OUTPUT/comparison" --bootstrap-samples 10000

echo "V5.2 fast programme complete: $OUTPUT/comparison/comparison.json"
