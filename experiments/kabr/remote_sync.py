"""Where a chunked run's resume state goes, and where the next chunk looks for it.

Two backends exist and they are not redundant:

    hf      a dataset repo. Storage is effectively free, the cache already lives there, and
            it is readable without a token, so a checkpoint stays reachable even from a
            machine that has no credentials at all.
    wandb   an artifact next to the run's own curves. One identity for a run instead of two,
            and the artifact is versioned per chunk rather than overwritten.

`--ckpt-remote hf,wandb` writes both. Order matters on the way back in: resume takes the
first backend that has something, so put the one you trust first.

The reason to prefer hf as the default is that it puts resumability and metrics logging on
separate failures. A wandb key that is missing, rate limited, or out of storage quota
currently costs the whole chunk, because the checkpoint had nowhere else to go. Splitting
them means the worst case is losing the curves, which are cheap to regenerate, instead of
losing eleven hours of training, which are not.

A checkpoint here is 572 MB. wandb's free tier is 100 GB of artifact storage, which two arms
pushing every chunk will reach sooner than the experiment finishes.
"""

from __future__ import annotations

from pathlib import Path

from kabr import hf_sync, wandb_sync

BACKENDS = ("hf", "wandb")


def backends(cfg) -> list[str]:
    """The configured backends, in order, rejecting typos rather than silently doing nothing.

    Silently is the failure that matters: a misspelled backend that parses as "none" trains
    for eleven hours and pushes nowhere, and looks exactly like success until the next chunk
    starts from scratch.
    """
    names = [s.strip() for s in (cfg.ckpt_remote or "").split(",") if s.strip()]
    if bad := [n for n in names if n not in BACKENDS]:
        raise SystemExit(f"unknown --ckpt-remote {bad}, expected some of {list(BACKENDS)}")
    return names


def push_checkpoint(cfg, run, checkpoint: Path, step: int, metadata: dict | None = None) -> None:
    """Push to every configured backend, and do not let one failure hide the others.

    Each backend is tried even if an earlier one raised, because the whole point of writing
    to two places is that one of them can be broken.
    """
    for name in backends(cfg):
        try:
            if name == "hf":
                hf_sync.push_chunk(cfg, checkpoint, step, metadata)
            else:
                wandb_sync.push_checkpoint(cfg, run, checkpoint, step, metadata)
        except Exception as exc:
            print(f"  {name} push failed: {type(exc).__name__}: {exc}", flush=True)


def resolve_resume(cfg) -> str:
    """Turn `--resume auto` into a path: the first backend that has one, else local disk."""
    if cfg.resume != "auto":
        return cfg.resume
    for name in backends(cfg):
        try:
            found = hf_sync.pull_chunk(cfg) if name == "hf" else wandb_sync.pull_checkpoint(cfg)
        except Exception as exc:
            print(f"{name} pull failed: {type(exc).__name__}: {exc}")
            continue
        if found is not None:
            return str(found)
    local = wandb_sync.latest_local(cfg.run_dir)
    if local is not None:
        print(f"nothing on the remotes, resuming from local {local}")
        return str(local)
    print("nothing to resume from, starting at step 0")
    return ""


def preflight(cfg) -> list[str]:
    """Problems that would only otherwise surface at the end of the chunk.

    A push is the last thing a session does, so a credential that cannot write is discovered
    when there is no time left to do anything about it. Returns human-readable problems,
    empty if the run can save what it is about to earn.
    """
    problems = []
    for name in backends(cfg):
        if name == "hf":
            if problem := hf_sync.preflight(cfg):
                problems.append(problem)
        elif name == "wandb":
            import os

            if cfg.wandb_mode != "online":
                problems.append(f"wandb is {cfg.wandb_mode}, so its artifact will not be pushed")
            elif not os.environ.get("WANDB_API_KEY"):
                problems.append("no WANDB_API_KEY in the environment")
    return problems
