"""Push a training chunk to Kaggle as a batch kernel, then watch it from here.

Batch rather than an interactive notebook, for three reasons that matter to a run measured
in sessions: the limit is 12 hours instead of 9, the job survives the terminal closing, and
quota is charged for the time the code runs rather than for the time a browser tab stays
open. The cost is no live output, which is already covered because the run reports to wandb.

    python experiments/kabr/kaggle/submit.py push
    python experiments/kabr/kaggle/submit.py status
    python experiments/kabr/kaggle/submit.py output -o /tmp/kabr-kernel

A run is a chain of chunks, and the chain only advances when the next one is pushed. `watch`
does that push, so the run continues through a night rather than stopping at whichever hour
the last chunk happened to end. It is a loop around the same `push` above, left running
detached, and it stops rather than retrying when a chunk dies young - see `watch()`.

    nohup python experiments/kabr/kaggle/submit.py watch --chunks 4 >> watch.log 2>&1 &

`kernel-metadata.json` is generated rather than committed. It carries a `<username>/<slug>`
id, and a stale one committed to a public repo is a paper cut for anyone who forks this and
pushes to an id that is not theirs.

Credentials come from `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`, the same
places the CLI reads. Nothing is written into the repository.

Getting a token into the kernel does not go through Kaggle Secrets, because that does not
survive this workflow. Secrets are attached per kernel through the editor page, the save API
has no field for them, and pushing a new version clears whatever was attached: measured, by
attaching them and watching `KAGGLE_KERNEL_INTEGRATIONS` come back empty on the next push. In
a run that is nine sessions long and every session is a push, that is nine trips to a web
page, each of which silently costs a chunk if forgotten.

So the tokens travel in a private dataset instead. `dataset_sources` is a field the save API
does control, so it reattaches itself on every push and there is nothing to remember:

    export HF_TOKEN=...  WANDB_API_KEY=...
    python submit.py secrets          # once, and again whenever a token is rotated

The exposure is the same as a Kaggle secret - a private dataset only its owner can read - and
neither the token nor the dataset ever enters the repository. Scope the HF token to just the
one dataset repo it needs to write, so a leak costs a revoke and nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENTRY = HERE / "kabr_session.py"
DEFAULT_SLUG = "kabr-video-diffusion-session"

# A private dataset holding the tokens, mounted read-only at /kaggle/input/<slug>/.
SECRETS_DATASET = "kabr-secrets"
SECRETS_FILE = "tokens.json"
# WANDB_API_KEY is optional: without it a chunk still trains and still saves, it just does
# not draw. HF_TOKEN is not, because it is what makes the session's work survive it.
SECRET_NAMES = ("HF_TOKEN", "WANDB_API_KEY")


def kaggle_username() -> str:
    """Whoever the CLI is authenticated as, whichever of the three ways that happened.

    There are two credential formats in circulation: the older `kaggle.json` holding a
    username and key, and an access token, which carries the identity without spelling it
    out. `kaggle config view` reports the resolved username under either, so it is the
    fallback rather than the first thing tried - it costs a subprocess.
    """
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cfg_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    cfg = cfg_dir / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["username"]
    # The CLI touches the network on its first call, so this occasionally times out on a
    # machine that is otherwise authenticated. Retry once, and report what actually went
    # wrong instead of the generic "not authenticated" message, which sends you looking in
    # the wrong place.
    problem = "no '- username:' line in the output"
    for _ in range(2):
        try:
            out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True,
                                 check=True, timeout=60).stdout
            for line in out.splitlines():
                if line.strip().startswith("- username:"):
                    name = line.split(":", 1)[1].strip()
                    if name and name != "None":
                        return name
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
    raise SystemExit(
        f"cannot resolve a Kaggle username from `kaggle config view` ({problem}).\n"
        f"Authenticate the CLI (an access token at {cfg_dir / 'access_token'}, or "
        f"kaggle.json), or set KAGGLE_USERNAME to skip the lookup")


class SecretsUnverified(RuntimeError):
    """The secrets dataset could not be confirmed present, which is not the same as absent."""


def dataset_exists(ref: str, attempts: int = 3, pause: float = 20.0) -> bool:
    """Whether the dataset is readable. Raises `SecretsUnverified` if that is not knowable.

    Kaggle answers a bare `403 Client Error: Forbidden` both for a dataset that does not
    exist and for one this credential is not currently allowed to read - measured, by asking
    for a dataset that was definitely never created and getting a response identical to the
    one a lapsed token gets for a dataset that definitely does. Nothing in the reply
    distinguishes them, so a failed probe means "could not tell" rather than "not there".

    That distinction is the whole point. The caller uses this to decide whether to attach the
    token dataset, and reading a transient failure as absence attaches nothing: the kernel
    then trains with no `HF_TOKEN`, stops itself at minute one by design, and the watch loop
    sees a chunk that died young and stops for the night. Retrying first, then refusing to
    guess, turns that into a delayed push instead of a lost session.
    """
    for attempt in range(attempts):
        try:
            out = subprocess.run(["kaggle", "datasets", "files", ref], capture_output=True,
                                 text=True, timeout=60)
            if out.returncode == 0:
                return True
            problem = (out.stdout + out.stderr).strip() or f"exit {out.returncode}"
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < attempts:
            time.sleep(pause)
    raise SecretsUnverified(
        f"could not confirm whether {ref} exists after {attempts} attempts: {problem}")


def metadata(slug: str, private: bool, allow_missing_secrets: bool = False) -> dict:
    # Attached here rather than through the editor, because this field survives a push and
    # an Add-ons -> Secrets toggle does not. `allow_missing_secrets` is the escape hatch for
    # a fork that has never run `secrets`, where the probe failing really does mean absent.
    secrets_ref = f"{kaggle_username()}/{SECRETS_DATASET}"
    try:
        sources = [secrets_ref] if dataset_exists(secrets_ref) else []
    except SecretsUnverified:
        if not allow_missing_secrets:
            raise
        sources = []
    return {
        "id": f"{kaggle_username()}/{slug}",
        "title": slug,
        "code_file": ENTRY.name,
        "language": "python",
        # A script, not a notebook: batch kernels run top to bottom with no cell state and
        # no way to answer a prompt, which is exactly what this job is.
        "kernel_type": "script",
        "is_private": private,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": sources,
        "competition_sources": [],
        "kernel_sources": [],
    }


def push_secrets() -> None:
    """Put the tokens in a private dataset, reading them from this shell's environment.

    Never from a file in the repository, and the staging directory is removed even if the
    upload fails, so the only copies are the one in the environment and the one on Kaggle.
    """
    values = {name: os.environ[name] for name in SECRET_NAMES if os.environ.get(name)}
    if "HF_TOKEN" not in values:
        raise SystemExit(
            "HF_TOKEN is not set in this shell, and it is the one that matters: without it a\n"
            "session trains for its full length and then has nowhere to put the result.\n"
            "  export HF_TOKEN=...        # write scope on the dataset repo\n"
            "  export WANDB_API_KEY=...   # optional, curves only")
    ref = f"{kaggle_username()}/{SECRETS_DATASET}"
    folder = Path(tempfile.mkdtemp(prefix="kabr-secrets-"))
    try:
        (folder / SECRETS_FILE).write_text(json.dumps(values, indent=2) + "\n")
        (folder / "dataset-metadata.json").write_text(json.dumps(
            {"title": SECRETS_DATASET, "id": ref, "licenses": [{"name": "other"}]},
            indent=2) + "\n")
        print(f"uploading {sorted(values)} to {ref} (private)")
        if dataset_exists(ref):
            code = kaggle("datasets", "version", "-p", str(folder), "-m", "rotate tokens")
        else:
            code = kaggle("datasets", "create", "-p", str(folder))
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    if code != 0:
        raise SystemExit(code)
    print(f"\n{ref} holds {sorted(values)}")
    print("  it attaches itself to every push from now on, nothing to click")


def with_overrides(source: str, overrides: dict[str, str]) -> str:
    """Inject `KABR_*` settings into the staged copy of the entry script.

    A kernel has no environment to set from the outside and a script kernel is one file, so
    the settings have to travel inside it. `setdefault` rather than assignment, so a real
    environment variable still wins if one ever exists.

    The block goes after the jupytext header rather than at the top of the file, which keeps
    the staged copy parseable as a notebook for anyone who downloads the kernel.
    """
    if not overrides:
        return source
    block = ["import os as _os  # injected by submit.py"]
    block += [f"_os.environ.setdefault({k!r}, {v!r})" for k, v in overrides.items()]
    injected = "\n".join(block) + "\n"

    lines = source.splitlines(keepends=True)
    end = 0
    if lines and lines[0].startswith("# ---"):
        for i, line in enumerate(lines[1:], start=1):
            if line.startswith("# ---"):
                end = i + 1
                break
    return "".join(lines[:end]) + "\n" + injected + "".join(lines[end:])


def stage(slug: str, private: bool, overrides: dict[str, str], meta: dict | None = None) -> Path:
    """Build the upload folder: the entry script and nothing else.

    Only the entry point is uploaded. It clones the repository at a branch, so the code that
    runs is the code in git rather than a copy that drifted, and the kernel diff stays
    readable.

    `meta` is passed in by `push`, which needs to look at it before deciding to upload at
    all; building it here as well would probe the secrets dataset a second time.
    """
    folder = Path(tempfile.mkdtemp(prefix="kabr-kernel-"))
    (folder / ENTRY.name).write_text(with_overrides(ENTRY.read_text(), overrides))
    (folder / "kernel-metadata.json").write_text(
        json.dumps(meta if meta is not None else metadata(slug, private), indent=2) + "\n")
    return folder


def kaggle(*args: str) -> tuple[int, str]:
    cmd = ["kaggle", *args]
    print("$", " ".join(cmd), flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True)
    text = out.stdout + out.stderr
    print(text, end="" if text.endswith("\n") else "\n", flush=True)
    return out.returncode, text


def push(ns, overrides: dict[str, str], quiet: bool = False) -> str | None:
    """Stage the entry script and submit it as a new version of the kernel.

    A push is what continues the run: the chunk that starts finds its predecessor's
    checkpoint on the hub and picks up from that step, so this is the same command whether
    it is session one or session nine.

    Returns `None` on success, or the failure text otherwise. Not an exit code, because
    `kaggle kernels push` returns 0 even when Kaggle rejects the push - "Kernel push error:
    Maximum weekly GPU quota of 30.00 hours reached." comes back on an exit code of 0, and a
    caller that only checked the code would log four of these as four successful pushes,
    which is exactly what happened the first time this ran unattended. The rejection has to
    be read out of the text instead, so `push -> "successfully pushed"` is what is trusted
    rather than the process's own opinion of whether it worked.
    """
    ref = f"{kaggle_username()}/{ns.slug}"
    # Before uploading, not after: a kernel pushed without the token dataset attached is a
    # chunk that trains nothing and stops at minute one, and by then the session slot is
    # spent. Refusing to push is the cheaper half of that trade.
    try:
        meta = metadata(ns.slug, not ns.public,
                        allow_missing_secrets=getattr(ns, "allow_missing_secrets", False))
    except SecretsUnverified as exc:
        return (f"{exc}. Not pushing: a chunk with no HF_TOKEN stops at minute one and "
                f"spends a session slot to do it. Pass --allow-missing-secrets to override.")
    folder = stage(ns.slug, not ns.public, overrides, meta)
    if not quiet:
        print(f"staged {folder}" + (f" with {overrides}" if overrides else ""))
        if not meta["dataset_sources"]:
            print(f"  no {SECRETS_DATASET} dataset: this chunk cannot save what it earns.")
            print(f"  fix it with: python {Path(__file__).name} secrets")
    code, text = kaggle("kernels", "push", "-p", str(folder),
                        "--accelerator", ns.accelerator, "-t", str(ns.timeout))
    shutil.rmtree(folder, ignore_errors=True)
    if code != 0:
        return text.strip() or f"exit {code}"
    if "successfully pushed" not in text.lower():
        return text.strip() or "no confirmation in kaggle's output"
    if not quiet:
        print(f"\npushed {ref}")
        print(f"  https://www.kaggle.com/code/{ref}")
        print(f"  watch it with: python {Path(__file__).name} status --slug {ns.slug}")
    return None


# %% ---------------------------------------------------------------- watch
#
# A chunk ends when its 11 hours are up, and the next one has to be pushed for the run to
# continue. Doing that by hand costs a session every time it is missed overnight, and the
# whole point of the chunking is that the run outlives the person watching it.

# The states that mean "not finished yet". Anything else is terminal, including the several
# spellings of cancelled, which is how a chunk that Kaggle killed comes back.
WAITING = {"QUEUED", "RUNNING"}
# Read back from a terminal chunk before believing it. A push does not take effect
# instantly, so for a minute or so after one the API still reports the *previous* version's
# terminal status - resubmitting on that would push twice for one finished chunk.
SETTLE_SECONDS = 900
# A second reason not to act on the first terminal read: the status API is not only stale
# right after a push, it is occasionally just wrong. Measured once, well outside the settle
# window - it reported COMPLETE for four minutes straight while training kept logging to
# wandb without a break, so a chunk that was hours from done looked finished. One re-poll
# after a short pause is what a person would do before believing it, and it costs one poll.
RECONFIRM_SECONDS = 90
# How long to wait before trying a push again after Kaggle rejects one for quota. This is
# not a failure the loop should give up on - the quota is weekly and resolves on its own -
# so it backs off and keeps trying rather than needing to be started by hand again once the
# week turns over. Coarser than the training poll interval because there is nothing to
# learn by asking again in five minutes.
QUOTA_RETRY_SECONDS = 1800
# Quota is not the only rejection that passes on its own. Measured 08-03: the OAuth access
# token reached its expiry mid-run and for thirty minutes every Kaggle call - `kernels
# status`, `datasets files` - came back "Permission ... denied", until the CLI refreshed the
# token by itself and the same commands worked again untouched. Treating that as fatal ends
# the run overnight over something that repaired itself before anyone could have looked. It
# is still bounded, because a revoked credential looks the same on any single reading and
# should not be retried forever.
TRANSIENT_RETRY_SECONDS = 900
TRANSIENT_PUSH_ATTEMPTS = 3


def push_failure_kind(failure: str) -> str:
    """`quota`, `transient` or `fatal` - how much patience a rejected push has earned.

    Text matching, because these arrive as prose on an exit code of 0 and there is no
    structured field to read. Deliberately generous about what counts as transient: the cost
    of waiting out a fatal error is `TRANSIENT_PUSH_ATTEMPTS` delays before stopping anyway,
    while the cost of giving up on a transient one is the rest of the night.
    """
    low = failure.lower()
    if "quota" in low:
        return "quota"
    transient = ("could not confirm", "denied", "403", "forbidden", "401", "unauthorized",
                 "expired", "timed out", "timeout", "connection", "temporarily")
    return "transient" if any(s in low for s in transient) else "fatal"


def kernel_status(ref: str) -> tuple[str, str]:
    """`(state, raw)`, with the state normalised to a bare upper-case word.

    The CLI prints `... has status "KernelWorkerStatus.RUNNING"`, and the enum prefix has
    not always been there across versions, so it is stripped rather than matched.
    """
    try:
        out = subprocess.run(["kaggle", "kernels", "status", ref], capture_output=True,
                             text=True, timeout=180)
        raw = (out.stdout + out.stderr).strip()
    except Exception as exc:
        return "UNKNOWN", f"{type(exc).__name__}: {exc}"
    if m := re.search(r'status "([^"]+)"', raw):
        return m.group(1).rsplit(".", 1)[-1].upper(), raw
    return "UNKNOWN", raw


def watch(ns, overrides: dict[str, str]) -> None:
    """Poll the kernel and push the next chunk when the current one finishes.

    Meant to be left running detached - it is a loop around the same `push` that a person
    would run, not a second way of submitting. Three things it refuses to do, because all
    three turn an unattended loop into either a quota fire or a run that quietly stopped:

    - Resubmit after a chunk that died young. A chunk that ends in minutes ended for a
      reason a resubmit will hit again - a missing token, a bad commit on the branch - and
      the loop would burn the week's hours rediscovering it. Anything past `--min-minutes`
      has real training behind it, so a crash there is worth continuing from.
    - Act on one terminal reading. The status API is stale for a while right after this
      loop's own push (`SETTLE_SECONDS`), and separately - measured once - just wrong:
      COMPLETE for four minutes while the kernel kept training. A second read after a short
      pause is what catches both.
    - Give up on a rejection that passes by itself. Weekly quota is one, and is retried
      indefinitely because it is not a failure at all next week. An expired credential is
      the other - measured, a token that lapsed mid-run made every Kaggle call fail for
      thirty minutes and then start working again untouched - and is retried
      `TRANSIENT_PUSH_ATTEMPTS` times before stopping, because a revoked one reads the same
      on any single attempt. A rejection that looks like neither - a malformed kernel - still
      stops the loop immediately.

    `--chunks` still caps how many chunks it will actually launch, so the worst case if
    something upstream starts failing in a way that looks like success stays bounded. Neither
    kind of backoff counts against the cap, since they launch nothing.

    A chunk killed mid-flight is still resubmitted, because the trainer pushes to the hub
    every 2000 steps: the next chunk restarts from the last of those, not from zero.
    """
    ref = f"{kaggle_username()}/{ns.slug}"
    def log(msg: str) -> None:
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)

    log(f"watching {ref}, will launch at most {ns.chunks} more chunk(s), "
        f"polling every {ns.interval}s")
    # The chunk already in flight was launched by hand, so its start time is unknown. It is
    # exempt from the young-death check for that reason: at worst that costs one resubmit,
    # which the check then catches on the round after.
    started, launched, seen_waiting = None, 0, False
    # Consecutive push rejections that looked like a credential problem. Reset by any push
    # that goes through, so a lapse today and another next week are not added together.
    transient = 0
    # A loop that prints nothing until something happens is indistinguishable from a loop
    # that died, and this one is meant to be left alone overnight. Say the state on the
    # first poll, whenever it changes, and hourly regardless, so `tail` answers "is it
    # still watching" without attaching to anything.
    last_state, last_note = None, 0.0
    while True:
        state, raw = kernel_status(ref)
        if state != last_state or time.time() - last_note > 3600:
            log(f"{state}" + (f", {(time.time() - started) / 60:.0f} min in" if started else ""))
            last_state, last_note = state, time.time()
        if state in WAITING:
            seen_waiting = True
            time.sleep(ns.interval)
            continue
        if state == "UNKNOWN":
            # A blip in the network is not a reason to stop watching an 11 hour job.
            log(f"  unreadable, retrying: {raw}")
            time.sleep(ns.interval)
            continue
        if not seen_waiting and started is not None and time.time() - started < SETTLE_SECONDS:
            time.sleep(ns.interval)
            continue

        # One terminal reading is not enough to act on - re-poll after a short pause and
        # only proceed if it is still terminal. Anything that comes back WAITING here was
        # the flaky case, and the outer loop treats it exactly like any other still-running
        # chunk from that point.
        time.sleep(RECONFIRM_SECONDS)
        confirm_state, confirm_raw = kernel_status(ref)
        if confirm_state in WAITING or confirm_state == "UNKNOWN":
            log(f"  {state} did not hold on re-poll ({confirm_state}); still watching")
            continue
        state, raw = confirm_state, confirm_raw

        ran = None if started is None else (time.time() - started) / 60
        log(f"chunk finished: {state}" + (f" after {ran:.0f} min" if ran else ""))
        if state != "COMPLETE":
            # Verbatim, because this is where the weekly GPU quota running out will show up
            # and nobody has seen what that looks like from here yet.
            log(f"  {raw}")
        if state != "COMPLETE" and ran is not None and ran < ns.min_minutes:
            log(f"  it lasted under {ns.min_minutes} min, so this is a failure a resubmit "
                f"would repeat rather than a chunk that ran out of clock. Stopping.")
            log(f"  look at it with: python {Path(__file__).name} log --slug {ns.slug}")
            return
        if launched >= ns.chunks:
            log(f"  reached the --chunks {ns.chunks} cap, stopping. "
                f"Resubmit by hand or start another watch.")
            return

        log(f"pushing chunk {launched + 1}/{ns.chunks}")
        failure = push(ns, overrides, quiet=True)
        kind = push_failure_kind(failure) if failure else None
        if kind == "quota":
            log(f"  {failure}")
            log(f"  weekly quota, not a bug - retrying in {QUOTA_RETRY_SECONDS // 60} min "
                f"rather than giving up on the run")
            time.sleep(QUOTA_RETRY_SECONDS)
            continue
        if kind == "transient":
            transient += 1
            log(f"  {failure}")
            if transient > TRANSIENT_PUSH_ATTEMPTS:
                log(f"  that is {transient} in a row, so it is not a credential refreshing "
                    f"itself. Stopping.")
                return
            log(f"  looks like a credential or the API being briefly unreadable "
                f"({transient}/{TRANSIENT_PUSH_ATTEMPTS}) - retrying in "
                f"{TRANSIENT_RETRY_SECONDS // 60} min")
            time.sleep(TRANSIENT_RETRY_SECONDS)
            continue
        if failure:
            log(f"  push rejected, stopping so it does not spin: {failure}")
            return
        transient = 0
        started, launched, seen_waiting = time.time(), launched + 1, False
        log(f"  pushed, https://www.kaggle.com/code/{ref}")
        time.sleep(ns.interval)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["push", "watch", "status", "output", "log",
                                       "metadata", "secrets"])
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    ap.add_argument("--accelerator", default="NvidiaTeslaT4",
                    help="NvidiaTeslaT4 or NvidiaTeslaP100")
    ap.add_argument("--timeout", type=int, default=43_200, help="kernel wall clock cap")
    ap.add_argument("--public", action="store_true", help="push as a public kernel")
    ap.add_argument("-o", "--out", default="kaggle-output")
    ap.add_argument("--set", action="append", default=[], metavar="KABR_X=value",
                    help="override a KABR_* setting inside the kernel; repeatable")
    ap.add_argument("--chunks", type=int, default=4,
                    help="watch: how many further chunks to launch before stopping")
    ap.add_argument("--interval", type=int, default=300,
                    help="watch: seconds between status polls")
    ap.add_argument("--min-minutes", type=int, default=45,
                    help="watch: a failed chunk shorter than this stops the loop")
    ap.add_argument("--allow-missing-secrets", action="store_true",
                    help="push even when the secrets dataset cannot be confirmed. For a fork "
                         "that has never run `secrets`; a chunk pushed this way cannot save "
                         "a checkpoint unless it gets HF_TOKEN some other way")
    ns = ap.parse_args()

    overrides = {}
    for item in ns.set:
        key, _, value = item.partition("=")
        if not _ or not key.startswith("KABR_"):
            raise SystemExit(f"--set wants KABR_KEY=value, got {item!r}")
        overrides[key] = value

    ref = f"{kaggle_username()}/{ns.slug}"
    if ns.action == "secrets":
        return push_secrets()
    if ns.action == "metadata":
        print(json.dumps(metadata(ns.slug, not ns.public, ns.allow_missing_secrets), indent=2))
        return
    if ns.action == "status":
        raise SystemExit(kaggle("kernels", "status", ref)[0])
    if ns.action in ("output", "log"):
        Path(ns.out).mkdir(parents=True, exist_ok=True)
        raise SystemExit(kaggle("kernels", "output", ref, "-p", ns.out)[0])
    if ns.action == "watch":
        return watch(ns, overrides)

    failure = push(ns, overrides)
    if failure:
        print(f"\npush rejected:\n{failure}")
    raise SystemExit(1 if failure else 0)


if __name__ == "__main__":
    main()
