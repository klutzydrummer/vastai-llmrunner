#!/bin/bash
# Dummy test with a fake vastai: no network, no real instances, dummy-deadman units only.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d)
export UNIT_PREFIX=dummy-deadman UNIT_DIR=$T/units FAKE_DIR=$T/fake
export VASTAI=$HERE/fakevast.sh LOG=$T/deadman.log RETRY_SLEEP=0 MAX_ATTEMPTS=3
mkdir -p "$FAKE_DIR"
ID=999
fail=0
check() { if eval "$2"; then echo "PASS: $1"; else echo "FAIL: $1"; fail=1; fi; }
cleanup() { "$HERE/disarm-deadman.sh" "$ID" >/dev/null 2>&1; rm -rf "$T"; }
# shellcheck disable=SC2329
trap cleanup EXIT

echo "== positive: destroy fails 2x then succeeds"
"$HERE/arm-deadman.sh" "$ID" +1 >"$T/arm.out" 2>&1; cat "$T/arm.out"

check "timer listed" "systemctl --user list-timers --all --no-pager | grep -q dummy-deadman-$ID.timer"
check "unit files persisted" "[ -f $UNIT_DIR/dummy-deadman-$ID.timer ] && [ -f $UNIT_DIR/dummy-deadman-$ID.service ]"
check "inhibitor active" "systemctl --user is-active -q dummy-deadman-$ID-inhibit.service"
for _ in $(seq 1 90); do grep -q 'VERIFIED\|FAILED' "$LOG" 2>/dev/null && break; sleep 1; done
check "destroy retried (2 failures logged)" "[ \"\$(grep -c 'destroy attempt .* failed' $LOG)\" = 2 ]"
check "VERIFIED logged" "grep -q VERIFIED $LOG"
"$HERE/disarm-deadman.sh" "$ID"
check "no dummy units left" "! systemctl --user list-units --all --no-legend 'dummy-deadman*' | grep -q ."
check "no dummy timers left" "! systemctl --user list-timers --all --no-pager | grep -q dummy-deadman"
check "no dummy files left" "[ -z \"\$(ls $UNIT_DIR)\" ]"

echo "== negative: destroy never succeeds"
rm -rf "$FAKE_DIR"; mkdir -p "$FAKE_DIR"; : > "$LOG"
FAKE_OK_AFTER=never "$HERE/deadman.sh" "$ID" >/dev/null 2>&1; rc=$?
check "exit 1" "[ $rc = 1 ]"
check "FAILED logged" "grep -q FAILED $LOG"

[ $fail = 0 ] && echo "ALL PASSED" || echo "SOME FAILED"
exit $fail
