#!/bin/bash
# Step-4 official run launcher (lead 19:45, fixed 19:47) — waits for the OPS window.
set -u
cd "$(dirname "$0")/.."
LOG=reports/step4-launcher.log
MESH="node /Users/cgarrot/.pi/agent/npm/node_modules/pi-mesh-extension/dist/src/cli/mesh.js send tagi-1"
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; $MESH "STEP4-LAUNCHER: $*" --room default >/dev/null 2>&1 || true; }

secs_until() {  # $1=HHMM local
  now=$(date +%H%M); now_s=$((10#${now:0:2}*3600 + 10#${now:2:2}*60))
  tgt_s=$((10#${1:0:2}*3600 + 10#${1:2:2}*60))
  d=$((tgt_s - now_s)); [ "$d" -le 0 ] && d=$((d + 86400)); echo "$d"
}

load_ok() { python3 -c "import os;print(1 if os.getloadavg()[0]<2.0 else 0)"; }

run_now() {
  say "LOAD OK — launching parallel run (4 workers)"
  export V1BIS_STEP4_GO=LEAD_APPROVED
  .venv/bin/python -c "
import json, hashlib
from ucm.v1.runner_parallel import run_v1bis_parallel
proto=open('artifacts/freeze-v1bis-official-v10.sha256').read().strip()  # ancre externe append-only (anti-tautologie)
r=run_v1bis_parallel('artifacts/freeze-v1bis-official-v10.json', proto, dry_run=False, n_workers=4)
print('run complete — results persisted by runner (see artifacts/v1bis-run/)')
" >> reports/step4-run-output.log 2>&1
  say "run finished rc=$? (see reports/step4-run-output.log)"
}

# Window 1: tonight 22:30 → 23:59
say "armed: waiting for tonight's window (22:30-23:59 local)"
sleep "$(secs_until 2230)"
say "night window open — polling load<2.0 (2 samples/60s)"
while [ "$(date +%H%M)" -lt 2359 ]; do
  a=$(load_ok); sleep 60; b=$(load_ok)
  if [ "$a" = "1" ] && [ "$b" = "1" ]; then run_now; exit 0; fi
done
say "night window closed without launch — re-arming for morning"
# Window 2: tomorrow 06:30 → 08:00
sleep "$(secs_until 0630)"
say "morning window open — polling load<2.0"
while [ "$(date +%H%M)" -lt 0800 ]; do
  a=$(load_ok); sleep 60; b=$(load_ok)
  if [ "$a" = "1" ] && [ "$b" = "1" ]; then run_now; exit 0; fi
done
say "ABORT: no calm window — repli série à décider (lead)"
