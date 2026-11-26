#!/bin/sh
# Dummy stand-in for vastai. State lives in $FAKE_DIR (required; never the script dir).
# destroy fails until FAKE_OK_AFTER attempts (default 3), then succeeds.
# FAKE_OK_AFTER=never makes destroy always fail. `show instances` lists id 999 until destroyed.
[ -n "${FAKE_DIR:-}" ] || { echo "FAKE_DIR required" >&2; exit 2; }
D=$FAKE_DIR
n=$(cat "$D/count" 2>/dev/null || echo 0)
case "$1 $2" in
  "destroy instance")
    n=$((n+1)); echo "$n" > "$D/count"
    if [ "${FAKE_OK_AFTER:-3}" != never ] && [ "$n" -ge "${FAKE_OK_AFTER:-3}" ]; then touch "$D/gone"; exit 0; fi
    echo "api error" >&2; exit 1;;
  "show instances")
    if [ -e "$D/gone" ]; then echo '[]'; else echo '[{"id": 999}]'; fi;;
esac
