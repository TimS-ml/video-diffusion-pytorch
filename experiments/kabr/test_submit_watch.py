"""Check when the unattended loop is willing to spend a session.

`submit.py watch` runs for days with nobody reading it, and both ways it can be wrong are
expensive: not pushing means the run quietly stops overnight, pushing when it should not
means the week's GPU hours go into a chunk that fails the same way the last one did. Neither
shows up for eleven hours, so they are tested against a fake clock instead.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SUBMIT = Path(__file__).resolve().parent / "kaggle" / "submit.py"
spec = importlib.util.spec_from_file_location("kabr_submit", SUBMIT)
submit = importlib.util.module_from_spec(spec)
sys.modules["kabr_submit"] = submit
spec.loader.exec_module(submit)


class ScriptEnded(Exception):
    """The status script ran out while the loop was still willing to keep watching."""


def run_watch(monkeypatch, states, chunks=2, min_minutes=45, push_returns=0):
    """Drive `watch` through a scripted status sequence on a clock we control.

    `states` are `(state, minutes_to_advance_before_reporting_it)`. The clock only moves
    when the loop polls, so a chunk's measured length is whatever the script says it is,
    and eleven hours of it cost nothing to test.

    Returns `(pushes, stopped)`. Not every case ends with the loop stopping - after its last
    push it goes on watching that chunk, which is the point - so running out of script is a
    result rather than an error.
    """
    now, pushes = [0.0], []
    script = list(states)

    monkeypatch.setattr(submit.time, "time", lambda: now[0])
    monkeypatch.setattr(submit.time, "sleep", lambda s: None)
    monkeypatch.setattr(submit.time, "strftime", lambda f: "test")
    monkeypatch.setattr(submit, "kaggle_username", lambda: "someone")
    monkeypatch.setattr(submit, "push",
                        lambda ns, ov, quiet=False: pushes.append(now[0]) or push_returns)

    def next_status(ref):
        if not script:
            raise ScriptEnded
        state, minutes = script.pop(0)
        now[0] += minutes * 60
        return state, state

    monkeypatch.setattr(submit, "kernel_status", next_status)
    ns = SimpleNamespace(slug="s", chunks=chunks, interval=1, min_minutes=min_minutes,
                         public=False, accelerator="NvidiaTeslaT4", timeout=43_200)
    try:
        submit.watch(ns, {})
    except ScriptEnded:
        return pushes, False
    return pushes, True


def test_status_parses_whatever_the_cli_spells_it(monkeypatch):
    """The enum prefix is not load bearing, and the terminal states are not one word."""
    for raw, want in [('x has status "KernelWorkerStatus.RUNNING"', "RUNNING"),
                      ('x has status "complete"', "COMPLETE"),
                      ('x has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"',
                       "CANCEL_ACKNOWLEDGED"),
                      ("403 - Forbidden", "UNKNOWN")]:
        monkeypatch.setattr(submit.subprocess, "run",
                            lambda *a, **k: SimpleNamespace(stdout=raw, stderr=""))
        assert submit.kernel_status("x")[0] == want, raw
    print("[ok] status parses through the enum prefix and the several spellings of cancelled")


def test_a_finished_chunk_launches_the_next(monkeypatch):
    pushes, _ = run_watch(monkeypatch, [
        ("RUNNING", 60), ("COMPLETE", 60),          # the chunk that was already in flight
        ("RUNNING", 660), ("COMPLETE", 0),          # the one this loop launched
        ("RUNNING", 60),
    ], chunks=2)
    assert len(pushes) == 2, pushes
    print("[ok] each finished chunk launches the next")


def test_the_cap_is_a_cap(monkeypatch):
    pushes, stopped = run_watch(monkeypatch, [
        ("RUNNING", 60), ("COMPLETE", 0),
        ("RUNNING", 660), ("COMPLETE", 0),          # would be push two, but the cap is one
    ], chunks=1)
    assert (len(pushes), stopped) == (1, True), (pushes, stopped)
    print("[ok] --chunks bounds what an unattended loop can spend")


def test_a_chunk_that_dies_young_stops_the_loop(monkeypatch):
    """The expensive case: a bad token or a broken commit fails in minutes, every time.

    Without this the loop would spend the week's quota rediscovering the same failure. The
    first chunk is exempt because the loop did not start it and cannot time it, so the
    script here has to get past that one first.
    """
    pushes, stopped = run_watch(monkeypatch, [
        ("RUNNING", 5), ("ERROR", 0),               # in flight already: resubmit once
        ("RUNNING", 2), ("ERROR", 0),               # died in 2 min: stop
    ], chunks=4)
    assert (len(pushes), stopped) == (1, True), (pushes, stopped)
    print("[ok] a chunk that fails in minutes stops the loop instead of repeating itself")


def test_a_chunk_killed_late_is_still_worth_resuming(monkeypatch):
    """Kaggle killing a chunk at hour ten is not a reason to stop: the trainer pushes to
    the hub every 2000 steps, so the next chunk starts from the last of those."""
    pushes, _ = run_watch(monkeypatch, [
        ("RUNNING", 60), ("COMPLETE", 0),
        ("RUNNING", 600), ("CANCEL_ACKNOWLEDGED", 0),
        ("RUNNING", 600), ("COMPLETE", 0),
        ("RUNNING", 60),
    ], chunks=3)
    assert len(pushes) == 3, pushes
    print("[ok] a chunk killed after real training is resumed, not abandoned")


def test_a_stale_terminal_status_does_not_push_twice(monkeypatch):
    """A push does not take effect instantly. For a minute after one the API still reports
    the previous version's COMPLETE, and believing it would push twice for one chunk."""
    pushes, _ = run_watch(monkeypatch, [
        ("RUNNING", 60), ("COMPLETE", 0),           # a real finish: one push
        ("COMPLETE", 1), ("COMPLETE", 1),           # the same stale answer, still settling
        ("RUNNING", 660), ("COMPLETE", 0),          # it really started, then really finished
        ("RUNNING", 60),
    ], chunks=3)
    assert len(pushes) == 2, pushes
    print("[ok] a stale COMPLETE right after a push is not counted as a finished chunk")


def test_settling_gives_up_rather_than_watching_a_dead_kernel(monkeypatch):
    """The other half of that: if the terminal status is still there long after the push,
    the push did not take and waiting for a RUNNING that never comes is worse than acting."""
    pushes, _ = run_watch(monkeypatch, [
        ("RUNNING", 60), ("COMPLETE", 0),           # push one
        ("COMPLETE", 5), ("COMPLETE", 5),           # settling, ignored
        ("COMPLETE", 10), ("COMPLETE", 0),          # past the settle window: believed
    ], chunks=3)
    assert len(pushes) == 2, pushes
    print("[ok] a terminal status that outlasts the settle window is believed, not waited on")


def test_an_unreadable_status_is_not_a_finished_chunk(monkeypatch):
    """A network blip must not read as "done, push the next one", and must not stop the
    watch either - it has to survive eleven hours of whatever the connection does."""
    pushes, _ = run_watch(monkeypatch, [
        ("UNKNOWN", 1), ("RUNNING", 60), ("UNKNOWN", 1), ("COMPLETE", 0),
        ("RUNNING", 660), ("COMPLETE", 0),
        ("RUNNING", 60),
    ], chunks=2)
    assert len(pushes) == 2, pushes
    print("[ok] an unreadable status is retried, not treated as a finished chunk")


def test_a_failed_push_stops_rather_than_spins(monkeypatch):
    """Out of weekly quota is the usual cause, and it does not clear by retrying."""
    pushes, stopped = run_watch(monkeypatch, [
        ("RUNNING", 60), ("COMPLETE", 0),
    ], chunks=4, push_returns=1)
    assert (len(pushes), stopped) == (1, True), (pushes, stopped)
    print("[ok] a rejected push stops the loop instead of retrying every five minutes")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
