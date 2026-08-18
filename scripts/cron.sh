#!/usr/bin/env bash
# Cron entrypoint for the pickengine daily paper-trading loop.
#
# Usage: cron.sh daily | daily-settle | capture-odds
#
# Install on the VPS (crontab -e), times in UTC — check the server's TZ or
# set CRON_TZ=UTC:
#
#   0 14 * * *  cd /opt/pickengine && ./scripts/cron.sh daily
#   30 16 * * * cd /opt/pickengine && ./scripts/cron.sh capture-odds
#   0 18 * * *  cd /opt/pickengine && ./scripts/cron.sh capture-odds
#   0 22 * * *  cd /opt/pickengine && ./scripts/cron.sh capture-odds
#   30 0 * * *  cd /opt/pickengine && ./scripts/cron.sh capture-odds
#   0 12 * * *  cd /opt/pickengine && ./scripts/cron.sh daily-settle
#
# The capture-odds passes are odds-only snapshots taken closer to first pitch
# than the 14:00 pull that prices the picks: 16:30 covers early day games
# (Sunday 13:05 ET starts at 17:05 UTC would otherwise close on the 14:00
# pull, a ~3h-stale gap that trips the report's staleness warning benignly),
# 18:00 later afternoon games, 22:00 evening ET starts, 00:30 west coast
# starts. Without them the closing line is just the pick-time snapshot
# re-flagged and paper CLV is 0 by construction. API budget: 5 pulls/day
# (daily + 4 captures) * ~30 days ~= 150 requests/month against The Odds API
# free tier's 500 — still comfortable. A paid tier would allow tighter
# pre-game captures (e.g. hourly or per-game T-5min) for sharper closing
# lines.
#
# Requirements on the VPS: uv installed and on cron's PATH (or symlink into
# /usr/local/bin), ODDS_API_KEY exported (e.g. via /etc/environment or a
# `. /opt/pickengine/.env` line added below), repo cloned at /opt/pickengine,
# `uv sync` run once. All output (stdout+stderr) is appended to
# ./logs/<command>-YYYY-MM-DD.log; a non-zero exit propagates to cron so
# failures show up in cron's mail/monitoring.

set -euo pipefail
cd "$(dirname "$0")/.."

CMD="${1:?usage: cron.sh daily|daily-settle|capture-odds}"
case "$CMD" in
  daily|daily-settle|capture-odds) ;;
  *) echo "unknown command: $CMD (expected daily, daily-settle, or capture-odds)" >&2; exit 2 ;;
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
