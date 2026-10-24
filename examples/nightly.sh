#!/usr/bin/env bash
# Nightly: process new recordings, rebuild the review reel, refresh the schedule audit.
# Edit the three paths, then add to cron (see docs/INTEGRATION.md).
set -euo pipefail

REPO="${REPO:-$HOME/service-trimmer}"
AUDIO="${AUDIO:-$HOME/recordings}"
WORK="${WORK:-$HOME/service-trimmer-work}"
REEL="${REEL:-$HOME/service-trimmer-work/sermon-review.mp3}"

# shellcheck disable=SC1091
source "$REPO/venv/bin/activate"
mkdir -p "$WORK/logs"

echo "== $(date '+%F %T') batch"
service-trimmer batch --audio-dir "$AUDIO" --workdir "$WORK" --extract --jobs 2

echo "== $(date '+%F %T') reel"
service-trimmer reel --workdir "$WORK" -o "$REEL"

echo "== $(date '+%F %T') schedule audit"
service-trimmer schedule --audio-dir "$AUDIO" --workdir "$WORK" -o "$WORK/SCHEDULE-AUDIT.txt" >/dev/null
echo "== $(date '+%F %T') done"
