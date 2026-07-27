#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:-v11_magnitude_main}"
SAFE_ID="$(printf '%s' "$RUN_ID" | tr -c 'A-Za-z0-9_.-' '-')"
LOG_DIR="$ROOT_DIR/logs/v11"
LOG_PATH="$LOG_DIR/${RUN_ID}.log"
mkdir -p "$LOG_DIR"

systemd-run --user --unit="cnvqg-v11-${SAFE_ID}" --collect \
  --property="WorkingDirectory=$ROOT_DIR" \
  --property="StandardOutput=append:$LOG_PATH" \
  --property="StandardError=append:$LOG_PATH" \
  /usr/bin/bash scripts/run_v11_magnitude_recovery.sh --run-id "$RUN_ID"

printf 'Queued cnvqg-v11-%s\n' "$SAFE_ID"
printf 'Log: %s\n' "$LOG_PATH"
