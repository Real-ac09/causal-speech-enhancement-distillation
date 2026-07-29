#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:-v10_$(date +%Y%m%d_%H%M%S)}"
shift || true

SAFE_ID="$(printf '%s' "$RUN_ID" | tr -c 'A-Za-z0-9_.-' '-')"
UNIT="cnvqg-v10-${SAFE_ID}"
LOG_DIR="$ROOT_DIR/logs/v10"
LOG_PATH="$LOG_DIR/${RUN_ID}.log"
mkdir -p "$LOG_DIR"

systemd-run --user --unit="$UNIT" --collect \
  --property="WorkingDirectory=$ROOT_DIR" \
  --property="StandardOutput=append:$LOG_PATH" \
  --property="StandardError=append:$LOG_PATH" \
  /usr/bin/bash scripts/run_v10_programme.sh \
  --wait-for-idle --run-id "$RUN_ID" "$@"

printf 'Queued %s\n' "$UNIT"
printf 'Log: %s\n' "$LOG_PATH"
printf 'Status: systemctl --user status %s\n' "$UNIT"
printf 'Follow: tail -f %s\n' "$LOG_PATH"
printf 'Stop: systemctl --user stop %s\n' "$UNIT"
