#!/bin/bash
# Remove the dead-man units for an instance.
# usage: disarm-deadman.sh [--force] <instance-id>
# Safe order: `vastai destroy instance <id> -y`, confirm the id is absent from
# `vastai show instances --raw`, THEN disarm. Refuses (exit 3) while the instance is
# still listed or the listing fails, unless --force. env: VASTAI (CLI, default vastai).
set -u
FORCE=0
if [ "${1:-}" = --force ]; then FORCE=1; shift; fi
ID=${1:?usage: disarm-deadman.sh [--force] <instance-id>}
[[ $ID =~ ^[0-9]+$ ]] || { echo "instance id must be numeric" >&2; exit 2; }
V=${VASTAI:-vastai}
if [ $FORCE = 0 ]; then
  if ! out=$("$V" show instances --raw 2>&1); then
    echo "REFUSING to disarm: 'vastai show instances --raw' failed; cannot confirm $ID is destroyed (use --force to override)" >&2
    exit 3
  fi
  n=$(jq --argjson id "$ID" '[.[] | select(.id == $id)] | length' <<<"$out" 2>/dev/null) || n=
  if [ -z "$n" ]; then
    echo "REFUSING to disarm: could not parse instance list; cannot confirm $ID is destroyed (use --force to override)" >&2
    exit 3
  fi
  if [ "$n" != 0 ]; then
    echo "REFUSING to disarm: instance $ID is still listed. Run 'vastai destroy instance $ID -y' and verify first (or use --force)" >&2
    exit 3
  fi
fi
PREFIX=${UNIT_PREFIX:-vast-deadman}
DIR=${UNIT_DIR:-$HOME/.config/systemd/user}
NAME=$PREFIX-$ID
systemctl --user disable --now "$NAME.timer" 2>/dev/null
systemctl --user stop "$NAME-inhibit.service" "$NAME.service" 2>/dev/null
systemctl --user disable "$NAME.service" "$NAME-inhibit.service" 2>/dev/null
rm -f "$DIR/$NAME.timer" "$DIR/$NAME.service" "$DIR/$NAME-inhibit.service"
systemctl --user daemon-reload
systemctl --user reset-failed "$NAME.timer" "$NAME.service" "$NAME-inhibit.service" 2>/dev/null
echo "disarmed $NAME"
exit 0
