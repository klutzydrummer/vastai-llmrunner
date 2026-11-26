# Dead-man guard: real suspend and reboot tests (run by you)

Both tests use the dummy units (`dummy-deadman-*`) and the fake `vastai`. No real
Vast instance, no money. Status: **UNRUN** until you report the results.

Run every command from the repository root. Keep the same terminal for the setup
block so the exported variables stay set.

## Setup (both tests)

```sh
# Create a scratch folder and point the guard at the fake vastai
export T=$HOME/deadman-hosttest UNIT_PREFIX=dummy-deadman; mkdir -p $T/fake; export FAKE_DIR=$T/fake VASTAI=$PWD/tools/deadman/fakevast.sh LOG=$T/deadman.log RETRY_SLEEP=0 MAX_ATTEMPTS=3; echo ready
```

## Test 1: suspend catch-up (UNRUN)

The deadline passes while the host sleeps. Expected: the guard fires within about
60 seconds of resume. The host will not wake itself for the user timer
(`WakeSystem=` is system-manager only), so the wake comes from rtcwake.

```sh
# Arm the dummy guard for 3 minutes from now (fake vastai, no real instance)
tools/deadman/arm-deadman.sh 777 +3
```

Expected: `armed dummy-deadman-777 for <UTC time>` and a `NEXT:` line about 3 minutes ahead.

The armed inhibitor holds a sleep block, so a plain `systemctl suspend` may be
refused. Remove only the inhibitor first (the timer stays armed):

```sh
# Release the sleep block so the host can suspend (timer stays armed)
systemctl --user stop dummy-deadman-777-inhibit.service
```

Option A, suspend until about 4 minutes from now with an RTC wake (needs sudo; rtcwake is in /run/current-system/sw/bin):

```sh
# Suspend to RAM and let the clock wake the host in 240 seconds
sudo rtcwake -m mem -s 240
```

Option B, suspend and wake it yourself (press a key or the power button after over 3 minutes):

```sh
# Suspend to RAM; wake it by hand after the 3-minute deadline has passed
systemctl suspend
```

After resume, check it fired:

```sh
# Show the guard's log lines and the unit journal after resume
cat $LOG; journalctl --user -u dummy-deadman-777.service --no-pager | tail -15; date -u
```

Expected: `DEADMAN fired for instance 777`, two `destroy attempt N failed` lines,
`VERIFIED`, and the `fired` timestamp no more than 60 s after the resume time
(compare against `date -u` and the resume time in `journalctl -b -u systemd-suspend` if needed).
PASS means fired and VERIFIED within 60 s of resume. If rtcwake printed an error the RTC may not
support alarms; use Option B.

Cleanup:

```sh
# Remove the dummy guard and scratch files
tools/deadman/disarm-deadman.sh 777; rm -rf $T; systemctl --user list-timers --all | grep dummy-deadman || echo clean
```

## Test 2: reboot catch-up (UNRUN)

The deadline passes while the host is powered off. Linger is enabled here
(`loginctl show-user $USER -p Linger` printed `Linger=yes` on 2026-11-26), so the
user manager starts at boot without a login. A `Persistent=true` timer that never
fired before has no stored timestamp, so this is the case the test is for.

```sh
# Arm the dummy guard for 4 minutes from now
tools/deadman/arm-deadman.sh 778 +4
```

Reboot at once, so the host is off at the deadline; wait at least 5 minutes before logging in if you can (linger starts the manager anyway):

```sh
# Reboot now (needs sudo)
sudo systemctl reboot
```

After boot, in a new terminal from the repository root:

```sh
# Show the guard's log and journal after the reboot
export T=$HOME/deadman-hosttest; cat $T/deadman.log; journalctl --user -u dummy-deadman-778.service --no-pager | tail -15; uptime -s; date -u
```

Expected: `fired for instance 778` and `VERIFIED` within about 60 s of boot (the timer
fires when the user manager starts). The setup variables were lost with the reboot;
the log path above is the same file. If the log is missing or empty, the catch-up did
not run: report that with `systemctl --user list-timers --all | grep dummy`.

Cleanup:

```sh
# Remove the dummy guard and scratch files
export UNIT_PREFIX=dummy-deadman; tools/deadman/disarm-deadman.sh 778; rm -rf $HOME/deadman-hosttest; systemctl --user list-timers --all | grep dummy-deadman || echo clean
```

Note for the reboot test: the arm script ran without `FAKE_DIR`/`VASTAI` in a fresh
shell, so run the Setup block once before arming (the values are baked into the
service unit and persist across the reboot).

## Reporting

Send back the output of the check commands and the resume or boot time. Until then
both tests stay UNRUN in `docs/deadman.md`.
