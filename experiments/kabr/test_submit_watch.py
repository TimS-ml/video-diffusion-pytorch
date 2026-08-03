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


def run_watch(monkeypatch, states, chunks=2, min_minutes=45, push_failures=None):
    """Drive `watch` through a scripted status sequence on a clock we control.

    `states` are `(state, minutes_to_advance_before_reporting_it)`. The clock only moves
    when the loop polls, so a chunk's measured length is whatever the script says it is, and
    eleven hours of it cost nothing to test. Every terminal reading the loop is meant to act
    on needs a second scripted entry right after it, because that is what `watch` now polls
    for before believing the first one - see `RECONFIRM_SECONDS`.

    `push_failures` is consumed one per `push()` call - `None` for a push that succeeds, a
    string containing the rejection text otherwise. Calls past the end of the list succeed.

    Returns `(calls, stopped)`, where `calls` is `[(sim_time, failure_or_None), ...]` for
    every push attempted. Not every case ends with the loop stopping - after its last push
    it goes on watching that chunk, and a quota rejection is meant to be retried rather than
    given up on - so running out of script is a result rather than an error.
    """
    now, calls = [0.0], []
    script = list(states)
    failures = list(push_failures) if push_failures else []

    monkeypatch.setattr(submit.time, "time", lambda: now[0])
    monkeypatch.setattr(submit.time, "sleep", lambda s: None)
    monkeypatch.setattr(submit.time, "strftime", lambda f: "test")
    monkeypatch.setattr(submit, "kaggle_username", lambda: "someone")

    def fake_push(ns, ov, quiet=False):
        outcome = failures.pop(0) if failures else None
        calls.append((now[0], outcome))
        return outcome

    monkeypatch.setattr(submit, "push", fake_push)

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
        return calls, False
    return calls, True


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


def test_push_trusts_kaggles_text_over_its_exit_code(monkeypatch):
    """`kaggle kernels push` returns 0 even when Kaggle itself rejects the push - measured,
    for a weekly-quota rejection. An exit-code check alone reads that as a success."""
    monkeypatch.setattr(submit, "kaggle_username", lambda: "someone")
    monkeypatch.setattr(submit, "stage", lambda slug, private, overrides: Path("/tmp/x"))
    monkeypatch.setattr(submit, "shutil", SimpleNamespace(rmtree=lambda *a, **k: None))
    monkeypatch.setattr(submit, "kaggle", lambda *a: (
        0, "Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.\n"))
    ns = SimpleNamespace(slug="s", public=False, accelerator="NvidiaTeslaT4", timeout=43_200)
    failure = submit.push(ns, {}, quiet=True)
    assert failure and "quota" in failure.lower(), failure
    print("[ok] a rejection is read from kaggle's text, not trusted to show up as an exit code")


def test_a_finished_chunk_launches_the_next(monkeypatch):
    calls, _ = run_watch(monkeypatch, [
        ("RUNNING", 60), ("COMPLETE", 60), ("COMPLETE", 0),   # in flight, finishes, confirmed
        ("RUNNING", 660), ("COMPLETE", 0), ("COMPLETE", 0),   # the one this loop launched
        ("RUNNING", 60),
    ], chunks=2)
    assert len(calls) == 2, calls
    print("[ok] each finished chunk launches the next")


def test_the_cap_is_a_cap(monkeypatch):
    calls, stopped = run_watch(monkeypatch, [
        ("RUNNING", 60), ("COMPLETE", 0), ("COMPLETE", 0),
        ("RUNNING", 660), ("COMPLETE", 0), ("COMPLETE", 0),   # would be push two, cap is one
    ], chunks=1)
    assert (len(calls), stopped) == (1, True), (calls, stopped)
    print("[ok] --chunks bounds what an unattended loop can spend")


def test_a_chunk_that_dies_young_stops_the_loop(monkeypatch):
    """The expensive case: a bad token or a broken commit fails in minutes, every time.

    Without this the loop would spend the week's quota rediscovering the same failure. The
    first chunk is exempt because the loop did not start it and cannot time it, so the
    script here has to get past that one first.
    """
    calls, stopped = run_watch(monkeypatch, [
        ("RUNNING", 5), ("ERROR", 0), ("ERROR", 0),           # in flight: resubmit once
        ("RUNNING", 2), ("ERROR", 0), ("ERROR", 0),           # died in 2 min: stop
    ], chunks=4)
    assert (len(calls), stopped) == (1, True), (calls, stopped)
    print("[ok] a chunk that fails in minutes stops the loop instead of repeating itself")


def test_a_chunk_killed_late_is_still_worth_resuming(monkeypatch):
    """Kaggle killing a chunk at hour ten is not a reason to stop: the trainer pushes to
    the hub every 2000 steps, so the next chunk starts from the last of those."""
    calls, _ = run_watch(monkeypatch, [
        ("RUNNING", 60), ("COMPLETE", 0), ("COMPLETE", 0),
        ("RUNNING", 600), ("CANCEL_ACKNOWLEDGED", 0), ("CANCEL_ACKNOWLEDGED", 0),
        ("RUNNING", 600), ("COMPLETE", 0), ("COMPLETE", 0),
        ("RUNNING", 60),
    ], chunks=3)
    assert len(calls) == 3, calls
    print("[ok] a chunk killed after real training is resumed, not abandoned")


def test_a_stale_terminal_status_does_not_push_twice(monkeypatch):
    """A push does not take effect instantly. For a minute after one the API still reports
    the previous version's COMPLETE, and believing it would push twice for one chunk."""
    calls, _ = run_watch(monkeypatch, [
        ("RUNNING", 60), ("COMPLETE", 0), ("COMPLETE", 0),    # a real finish: one push
        ("COMPLETE", 1), ("COMPLETE", 1),                      # stale answer, still settling
        ("RUNNING", 660), ("COMPLETE", 0), ("COMPLETE", 0),   # really started, really finished
        ("RUNNING", 60),
    ], chunks=3)
    assert len(calls) == 2, calls
    print("[ok] a stale COMPLETE right after a push is not counted as a finished chunk")


def test_settling_gives_up_rather_than_watching_a_dead_kernel(monkeypatch):
    """The other half of that: if the terminal status is still there long after the push,
    the push did not take and waiting for a RUNNING that never comes is worse than acting."""
    calls, _ = run_watch(monkeypatch, [
        ("RUNNING", 60), ("COMPLETE", 0), ("COMPLETE", 0),    # push one
        ("COMPLETE", 5), ("COMPLETE", 5),                      # settling, ignored
        ("COMPLETE", 10), ("COMPLETE", 0),                     # past the settle window: acted on
        ("RUNNING", 60),
    ], chunks=3)
    assert len(calls) == 2, calls
    print("[ok] a terminal status that outlasts the settle window is believed, not waited on")


def test_a_single_bad_reading_does_not_end_the_run(monkeypatch):
    """Measured once: the status API reported COMPLETE for four minutes solid while the
    kernel kept logging steps to wandb without a break. One re-poll catches this before it
    burns a resubmit on a chunk that was still hours from done."""
    calls, _ = run_watch(monkeypatch, [
        ("RUNNING", 300), ("COMPLETE", 4), ("RUNNING", 1),     # the flaky read, then correct
        ("RUNNING", 360), ("COMPLETE", 0), ("COMPLETE", 0),    # the real finish, confirmed
        ("RUNNING", 60),
    ], chunks=2)
    assert len(calls) == 1, calls
    print("[ok] a terminal reading that does not repeat on re-poll is not believed")


def test_an_unreadable_status_is_not_a_finished_chunk(monkeypatch):
    """A network blip must not read as "done, push the next one", and must not stop the
    watch either - it has to survive eleven hours of whatever the connection does."""
    calls, _ = run_watch(monkeypatch, [
        ("UNKNOWN", 1), ("RUNNING", 60), ("UNKNOWN", 1), ("COMPLETE", 0), ("COMPLETE", 0),
        ("RUNNING", 660), ("COMPLETE", 0), ("COMPLETE", 0),
        ("RUNNING", 60),
    ], chunks=2)
    assert len(calls) == 2, calls
    print("[ok] an unreadable status is retried, not treated as a finished chunk")


def test_a_non_quota_push_failure_stops_the_loop(monkeypatch):
    """Auth broke, or the staged kernel is malformed: retrying every five minutes for a
    week does not fix either, so this is the case that should stop and wait for a person."""
    calls, stopped = run_watch(monkeypatch, [
        ("RUNNING", 60), ("COMPLETE", 0), ("COMPLETE", 0),
    ], chunks=4, push_failures=["403 Forbidden: token expired"])
    assert (len(calls), stopped) == (1, True), (calls, stopped)
    assert calls[0][1] and "quota" not in calls[0][1].lower()
    print("[ok] a non-quota push rejection stops the loop instead of spinning on it")


def test_a_quota_rejection_backs_off_and_keeps_trying(monkeypatch):
    """The failure mode that actually happened: four chunks of budget burned on pushes
    that were silently rejected. Quota is not a bug and resolves on its own on the weekly
    reset, so this should retry rather than stop - and the retry should not count against
    `--chunks`, since a rejected push never launched anything."""
    calls, stopped = run_watch(monkeypatch, [
        ("RUNNING", 60), ("COMPLETE", 0), ("COMPLETE", 0),     # quota-rejected
        ("COMPLETE", 0), ("COMPLETE", 0),                       # retried: still terminal, confirmed
        ("RUNNING", 60),                                        # the retry succeeded
    ], chunks=1, push_failures=[
        "Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.", None])
    assert len(calls) == 2, calls
    assert calls[0][1] and "quota" in calls[0][1].lower()
    assert calls[1][1] is None
    assert stopped is False, "a quota rejection must not stop the loop"
    print("[ok] a quota rejection is retried rather than treated as a reason to give up")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
