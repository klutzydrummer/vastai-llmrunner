#!/bin/bash
# Simulated downtime catch-up test with a fake vastai. Never suspends or reboots.
# Models "host off" by stopping the timer past its deadline, then restarting it.
# This shows the Persistent= logic, NOT a real suspend/reboot. dummy-deadman units only.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d)
export UNIT_PREFIX=dummy-deadman UNIT_DIR=$T/units FAKE_DIR=$T/fake
export VASTAI=$HERE/fakevast.sh LOG=$T/deadman.log RETRY_SLEEP=0 MAX_ATTEMPTS=3
mkdir -p "$FAKE_DIR"
ID=998
NAME=dummy-deadman-$ID
fail=0
check() { if eval "$2"; then echo "PASS: $1"; else echo "FAIL: $1"; fail=1; fi; }
cleanup() {
  "$HERE/disarm-deadman.sh" "$ID" >/dev/null 2>&1
  systemctl --user stop "dummy-deadman-jump.timer" "dummy-deadman-jump.service" 2>/dev/null
  rm -rf "$T"
}
# shellcheck disable=SC2329
trap cleanup EXIT
reset_state() { rm -rf "$FAKE_DIR"; mkdir -p "$FAKE_DIR"; : > "$LOG"; }
wait_log() { for _ in $(seq 1 "$1"); do grep -q 'VERIFIED\|FAILED' "$LOG" 2>/dev/null && return; sleep 1; done; }
arm_in() { "$HERE/arm-deadman.sh" "$ID" "$(date -u -d "+$1 seconds" '+%Y-%m-%d %H:%M:%S')"; }

echo "== A: timer stopped across the deadline (host off), then restarted"
reset_state
arm_in 25
systemctl --user stop "$NAME.timer"
check "timer inactive while 'off'" "! systemctl --user is-active -q $NAME.timer"
sleep 35
check "nothing fired while off" "! grep -q fired $LOG"
systemctl --user start "$NAME.timer"
wait_log 30
check "fired after restart" "grep -q 'fired for instance $ID' $LOG"
check "VERIFIED after restart" "grep -q VERIFIED $LOG"
"$HERE/disarm-deadman.sh" "$ID" >/dev/null

echo "== B: user manager re-exec across the deadline"
reset_state
arm_in 25
systemctl --user daemon-reexec
sleep 35
wait_log 30
check "fired despite daemon-reexec" "grep -q 'fired for instance $ID' $LOG"
check "VERIFIED" "grep -q VERIFIED $LOG"
"$HERE/disarm-deadman.sh" "$ID" >/dev/null

echo "== C: reboot model: timer and inhibitor stopped but still enabled, restart past deadline"
reset_state
arm_in 25
systemctl --user stop "$NAME.timer" "$NAME-inhibit.service" 2>/dev/null
systemctl --user daemon-reload
check "timer still enabled" "systemctl --user is-enabled -q $NAME.timer"
sleep 35
systemctl --user start "$NAME.timer"
wait_log 30
check "fired after restart" "grep -q 'fired for instance $ID' $LOG"
check "VERIFIED" "grep -q VERIFIED $LOG"
"$HERE/disarm-deadman.sh" "$ID" >/dev/null

echo "== D: monotonic vs calendar clocks, both armed 20s ahead (no suspend; baseline only)"
reset_state
J=dummy-deadman-jump
systemd-run --user --quiet --timer-property=AccuracySec=1s --unit="$J-mono" --on-active=20s "$(command -v sh)" -c "echo mono >> $T/clock.out"
systemd-run --user --quiet --timer-property=AccuracySec=1s --unit="$J-cal" --on-calendar="$(date -u -d '+20 seconds' '+%F %T') UTC" "$(command -v sh)" -c "echo cal >> $T/clock.out"
sleep 25
check "mono timer fired awake" "grep -q mono $T/clock.out"
check "calendar timer fired awake (they diverge only across a real suspend)" "grep -q cal $T/clock.out"
systemctl --user stop "$J-mono.timer" "$J-cal.timer" 2>/dev/null

echo "== cleanup"
"$HERE/disarm-deadman.sh" "$ID" >/dev/null
check "no dummy timers left" "! systemctl --user list-timers --all --no-pager | grep -q dummy-deadman"
check "no dummy units left" "! systemctl --user list-units --all --no-legend 'dummy-deadman*' | grep -q ."

[ $fail = 0 ] && echo "ALL PASSED" || echo "SOME FAILED"
exit $fail
