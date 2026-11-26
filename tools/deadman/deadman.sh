#!/bin/bash
# Destroy a Vast.ai instance and verify it is gone.
# usage: deadman.sh <instance-id>
# env: VASTAI (CLI, default vastai), LOG (default ~/deadman.log),
#      RETRY_SLEEP (seconds between attempts, default 30), MAX_ATTEMPTS (default 10)
set -u
ID=${1:?usage: deadman.sh <instance-id>}
[[ $ID =~ ^[0-9]+$ ]] || { echo "instance id must be numeric" >&2; exit 2; }
V=${VASTAI:-vastai}
LOG=${LOG:-$HOME/deadman.log}
MAX=${MAX_ATTEMPTS:-10}
log() { echo "$(date -u +%FT%TZ) DEADMAN $*" | tee -a "$LOG" >&2; }

listed() { # 0 if the id is still listed; a failed query counts as still listed
  local out
  out=$("$V" show instances --raw 2>>"$LOG") || return 0
  [ "$(jq --argjson id "$ID" '[.[] | select(.id == $id)] | length' <<<"$out" 2>>"$LOG")" = 0 ] && return 1
  return 0
}

log "fired for instance $ID"
for ((i = 1; i <= MAX; i++)); do
  "$V" destroy instance "$ID" >>"$LOG" 2>&1 || log "destroy attempt $i failed"
  if ! listed; then
    log "VERIFIED: instance $ID not listed (attempt $i)"
    exit 0
  fi
  [ "$i" -lt "$MAX" ] && sleep "${RETRY_SLEEP:-30}"
done
log "FAILED: instance $ID STILL LISTED after $MAX attempts"
notify-send -u critical "DEADMAN FAILED $ID" 2>/dev/null || true
exit 1
