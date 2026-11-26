#!/bin/bash
# Remove the dead-man units for an instance. usage: disarm-deadman.sh <instance-id>
set -u
ID=${1:?usage: disarm-deadman.sh <instance-id>}
[[ $ID =~ ^[0-9]+$ ]] || { echo "instance id must be numeric" >&2; exit 2; }
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
