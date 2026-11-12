#!/usr/bin/env bash
# Cron entry point: collect Slurm state, rebuild the dashboard, push it to gh-pages.
#   */5 * * * * /path/to/compute-monitor/scripts/cron_update.sh
# Set GPUMON_NO_PUBLISH=1 to only rebuild locally.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/usr/bin:/usr/local/bin:/bin:$PATH"
mkdir -p "$REPO/data"
LOG="$REPO/data/cron.log"
exec 9>"$REPO/data/.lock"
flock -n 9 || { echo "$(date -Is) previous run still going, skipping" >> "$LOG"; exit 0; }
cd "$REPO"
{
  echo "$(date -Is) start"
  if [[ "${GPUMON_NO_PUBLISH:-0}" == "1" ]]; then
    python3 -m gpumon update
  else
    python3 -m gpumon publish
  fi
} >> "$LOG" 2>&1
# keep the log small
tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
