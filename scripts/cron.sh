#!/usr/bin/env bash
# Cron entrypoint for the pickengine daily paper-trading loop.
#
# Usage: cron.sh daily | daily-settle
#
# Install on the VPS (crontab -e), times in UTC — check the server's TZ or
# set CRON_TZ=UTC:
#
#   0 14 * * * cd /opt/pickengine && ./scripts/cron.sh daily
#   0 12 * * * cd /opt/pickengine && ./scripts/cron.sh daily-settle
#
# Requirements on the VPS: uv installed and on cron's PATH (or symlink into
# /usr/local/bin), ODDS_API_KEY exported (e.g. via /etc/environment or a
# `. /opt/pickengine/.env` line added below), repo cloned at /opt/pickengine,
# `uv sync` run once. All output (stdout+stderr) is appended to
# ./logs/<command>-YYYY-MM-DD.log; a non-zero exit propagates to cron so
# failures show up in cron's mail/monitoring.

set -euo pipefail
cd "$(dirname "$0")/.."

CMD="${1:?usage: cron.sh daily|daily-settle}"
case "$CMD" in
  daily|daily-settle) ;;
  *) echo "unknown command: $CMD (expected daily or daily-settle)" >&2; exit 2 ;;
esac

mkdir -p logs
LOG="logs/${CMD}-$(date -u +%F).log"

{
  echo "=== $(date -u '+%Y-%m-%d %H:%M:%S') UTC — pickengine ${CMD} ==="
  STATUS=0
  uv run python -m pickengine "$CMD" || STATUS=$?
  echo "=== exit ${STATUS} at $(date -u '+%Y-%m-%d %H:%M:%S') UTC ==="
  exit "$STATUS"
} >>"$LOG" 2>&1
