#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
SEARCH_BATCHES="${V61_SEARCH_BATCHES:-600}"
mkdir -p checkpoints/v61/search results/v61/search results/v61/comparisons logs/v61
exec 9>logs/v61/search.lock
flock -n 9 || { echo "Another V6.1 search is active" >&2; exit 1; }

CONFIG=configs/v6/v61_teacher_search_fullband.yaml
CHECKPOINT=checkpoints/v61/search/fullband_polar/best.pt

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
  python scripts/check_v51_candidate.py --config "$CONFIG" \
  --output results/v61/structural_gates.json --device cuda

if [[ ! -f "$CHECKPOINT" ]]; then
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/train.py --config "$CONFIG" --device cuda \
    --max-train-batches "$SEARCH_BATCHES"
fi

for scale in 050 075 100; do
  case "$scale" in
    050) value=0.50 ;;
    075) value=0.75 ;;
    100) value=1.00 ;;
  esac
  output="results/v61/search/fullband_polar_phase_$scale"
  if [[ ! -f "$output/summary.json" ]]; then
    "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
      python scripts/evaluate.py --checkpoint "$CHECKPOINT" \
      --metadata data/processed/voicebank_demand/metadata/val_v51_search_100.csv \
      --output-dir "$output" --device cuda --phase-residual-scale "$value"
  fi
done

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import json
from pathlib import Path

root = Path("results/v61/search")
rows = []
for scale in ("050", "075", "100"):
    directory = root / f"fullband_polar_phase_{scale}"
    metrics = json.loads((directory / "summary.json").read_text())["metrics"]
    rows.append({"name": f"v61_fullband_phase_{scale}", "directory": str(directory), **metrics})
winner = max(rows, key=lambda row: row["enhanced_pesq"])
report = {"winner": winner["name"], "candidates": rows}
Path("results/v61/search_decision.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

WINNER="$(jq -r '.winner | split("_") | last' results/v61/search_decision.json)"
if [[ ! "$WINNER" =~ ^(050|075|100)$ ]]; then
  echo "Invalid winning phase suffix: $WINNER" >&2
  exit 2
fi
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
  python scripts/compare_evaluations.py \
  --reference v6_polar075=results/v6/search/polar_phase_075 \
  --candidate v61_fullband="results/v61/search/fullband_polar_phase_$WINNER" \
  --output-dir results/v61/comparisons/v6_vs_fullband \
  --bootstrap-samples 10000

echo "V6.1 controlled search complete"
