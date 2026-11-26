#!/bin/bash
# Arm a persistent user timer that destroys a Vast.ai instance at an absolute time.
# usage: arm-deadman.sh <instance-id> <"YYYY-MM-DD HH:MM:SS" UTC | +minutes>
# env: UNIT_PREFIX (default vast-deadman), UNIT_DIR (default ~/.config/systemd/user),
#      and VASTAI, LOG, RETRY_SLEEP, MAX_ATTEMPTS passed through to deadman.sh.
set -euo pipefail
ID=${1:?usage: arm-deadman.sh <instance-id> <UTC time|+minutes>}
WHEN=${2:?usage: arm-deadman.sh <instance-id> <UTC time|+minutes>}
[[ $ID =~ ^[0-9]+$ ]] || { echo "instance id must be numeric" >&2; exit 2; }
PREFIX=${UNIT_PREFIX:-vast-deadman}
DIR=${UNIT_DIR:-$HOME/.config/systemd/user}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME=$PREFIX-$ID

if [[ $WHEN == +* ]]; then
  DEADLINE=$(date -u -d "+${WHEN#+} minutes" '+%Y-%m-%d %H:%M:%S')
else
  DEADLINE=$(date -u -d "$WHEN UTC" '+%Y-%m-%d %H:%M:%S')
fi
END=$(date -u -d "$DEADLINE UTC" +%s)
NOW=$(date -u +%s)
[ "$END" -gt "$NOW" ] || { echo "deadline $DEADLINE UTC is in the past" >&2; exit 2; }

ENVLINES="Environment=PATH=$PATH"
for v in VASTAI LOG RETRY_SLEEP MAX_ATTEMPTS FAKE_DIR; do  # FAKE_DIR: test-only
  [ -n "${!v:-}" ] && ENVLINES+=$'\n'"Environment=$v=${!v}"
done

mkdir -p "$DIR"
cat > "$DIR/$NAME.service" <<UNIT
[Unit]
Description=Dead-man: destroy Vast instance $ID

[Service]
Type=oneshot
$ENVLINES
ExecStart=$HERE/deadman.sh $ID
UNIT
cat > "$DIR/$NAME.timer" <<UNIT
[Unit]
Description=Dead-man timer for Vast instance $ID

[Timer]
OnCalendar=$DEADLINE UTC
Persistent=true
AccuracySec=1s
RemainAfterElapse=no

[Install]
WantedBy=timers.target
UNIT
cat > "$DIR/$NAME-inhibit.service" <<UNIT
[Unit]
Description=Block host sleep until dead-man deadline for instance $ID

[Service]
ExecStart=$(command -v systemd-inhibit) --what=sleep --who=$NAME --why="dead-man armed for $ID" --mode=block $(command -v sleep) $((END - NOW + 5))
UNIT

# A custom UNIT_DIR is outside systemd's search path: link the units in.
if [ "$DIR" != "$HOME/.config/systemd/user" ]; then
  systemctl --user link "$DIR/$NAME.service" "$DIR/$NAME-inhibit.service" "$DIR/$NAME.timer"
fi
systemctl --user daemon-reload
systemctl --user enable --now "$NAME.timer"
systemctl --user start "$NAME-inhibit.service"

LINE=$(systemctl --user list-timers --all --no-pager | grep -F "$NAME.timer" || true)
[ -n "$LINE" ] || { echo "ERROR: $NAME.timer not listed after arming" >&2; exit 1; }
echo "armed $NAME for $DEADLINE UTC"
echo "NEXT: $LINE"
