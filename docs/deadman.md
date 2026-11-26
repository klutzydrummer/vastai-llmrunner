# Dead-man guard for Vast.ai instances

Auto-destroys a rented instance at an absolute UTC time, retrying and verifying.

## Why
RUN-1 was billed about 2.1 h against a planned 85 min. The cause is unproven.
The likely one (doc-based, not observed) is a monotonic `--on-active` timer
pausing while the host was suspended. Absolute `OnCalendar=` uses the realtime
clock and catches up after resume.
Source: https://www.freedesktop.org/software/systemd/man/latest/systemd.timer.html

## Usage
```
tools/deadman/arm-deadman.sh <instance-id> "2026-11-26 19:30:00"   # UTC
tools/deadman/arm-deadman.sh <instance-id> +85                     # minutes from now
tools/deadman/disarm-deadman.sh <instance-id>
```
Arming writes `vast-deadman-<id>.{service,timer}` under `~/.config/systemd/user`
(`Persistent=true`, `AccuracySec=1s`, `RemainAfterElapse=no`), enables the timer,
prints the `NEXT` line, and holds a sleep inhibitor
(`vast-deadman-<id>-inhibit.service`) until the deadline.
`deadman.sh` tries `vastai destroy instance` up to `MAX_ATTEMPTS` (10) times,
verifies with `vastai show instances --raw | jq`, logs to `LOG`, and sends
`notify-send` and exits 1 on failure. Env: `VASTAI`, `LOG`, `RETRY_SLEEP`,
`MAX_ATTEMPTS`, `UNIT_PREFIX`, `UNIT_DIR`. Disarm after a normal teardown.

## Tested vs not tested
- TESTED (awake, fake `vastai`): `tools/deadman/test-dummy.sh` covers arm, timer
  listed, unit file persisted, inhibitor active, two failed destroys then a
  verified one, disarm leaving nothing, and a never-succeeds case exiting 1.
- NOT tested: suspend/resume catch-up and reboot catch-up (doc-based only).
- NOT tested: a real `vastai` destroy.

## Credentials
Never put the home account key on an untrusted worker: it has full account access.

## Instance-side watchdog (UNVERIFIED)
Idea: onstart runs `sleep N; vastai stop instance $CONTAINER_ID` using only the
per-instance `CONTAINER_API_KEY`. Its permission scope is undocumented. Stop
also keeps storage charges, so home-side destroy stays primary.

## Audit
At the end of every plan the middle manager runs `vastai show instances` and
confirms nothing of ours is listed; do not trust a timer's absence.
