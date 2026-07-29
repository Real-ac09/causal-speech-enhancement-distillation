#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CONDA_BIN="${CONDA_BIN:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${CNVQG_ENV_NAME:-cnvqg}"
ENV_PREFIX="${CNVQG_ENV_PREFIX:-$HOME/miniconda3/envs/$ENV_NAME}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
SEARCH_BATCHES="${V44_SEARCH_BATCHES:-600}"
mkdir -p checkpoints/v44/search results/v44/search results/v44/comparisons logs/v44
exec 9>logs/v44/search.lock
flock -n 9 || { echo "Another V4.4 causal search is active" >&2; exit 1; }

CONFIG=configs/v4/train_v44_causal_v43_recipe_search.yaml
CHECKPOINT=checkpoints/v44/search/causal_v43_recipe/best.pt

"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
  python scripts/check_v51_candidate.py --config "$CONFIG" \
  --output results/v44/structural_gates.json --device cuda

if [[ ! -f "$CHECKPOINT" ]]; then
  "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
    python scripts/train.py --config "$CONFIG" --device cuda \
    --max-train-batches "$SEARCH_BATCHES"
fi

for scale in 000 050 100; do
  case "$scale" in
    000) value=0.00 ;;
    050) value=0.50 ;;
    100) value=1.00 ;;
  esac
  output="results/v44/search/phase_$scale"
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

rows = []
for scale in ("000", "050", "100"):
    directory = Path(f"results/v44/search/phase_{scale}")
    metrics = json.loads((directory / "summary.json").read_text())["metrics"]
    rows.append({"name": f"v44_phase_{scale}", "directory": str(directory), **metrics})
winner = max(rows, key=lambda row: row["enhanced_pesq"])
Path("results/v44/search_decision.json").write_text(
    json.dumps({"winner": winner["name"], "candidates": rows}, indent=2) + "\n"
)
print(json.dumps({"winner": winner["name"], "candidates": rows}, indent=2))
PY

WINNER="$(jq -r '.winner | split("_") | last' results/v44/search_decision.json)"
if [[ ! "$WINNER" =~ ^(000|050|100)$ ]]; then
  echo "Invalid winning phase suffix: $WINNER" >&2
  exit 2
fi
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" \
  python scripts/compare_evaluations.py \
  --reference v6_polar075=results/v6/search/polar_phase_075 \
  --candidate v44_causal="results/v44/search/phase_$WINNER" \
  --output-dir results/v44/comparisons/v6_vs_v44 --bootstrap-samples 10000

echo "V4.4 causal V4.3-recipe search complete"
