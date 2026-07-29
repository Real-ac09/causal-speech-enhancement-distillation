#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:-v11_decoder_main}"
SAFE_ID="$(printf '%s' "$RUN_ID" | tr -c 'A-Za-z0-9_.-' '-')"
LOG_PATH="$ROOT_DIR/logs/v11/${RUN_ID}.log"
mkdir -p "$ROOT_DIR/logs/v11"
systemd-run --user --unit="cnvqg-v11-decoder-${SAFE_ID}" --collect \
  --property="WorkingDirectory=$ROOT_DIR" \
  --property="StandardOutput=append:$LOG_PATH" \
  --property="StandardError=append:$LOG_PATH" \
  /usr/bin/bash scripts/run_v11_decoder_stage.sh --run-id "$RUN_ID"
printf 'Queued cnvqg-v11-decoder-%s\nLog: %s\n' "$SAFE_ID" "$LOG_PATH"
