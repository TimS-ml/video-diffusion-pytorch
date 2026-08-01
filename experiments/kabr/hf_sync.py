"""A HF dataset repo as the thing a chunked run survives in: the cache, and the resume state.

Training runs in fixed blocks of wall clock on machines that keep nothing afterwards, so
everything needed to carry on has to live somewhere the next session can reach:

    cache/<slug>_<size>px/    frames.u8, index.npz, manifest.json
    chunks/<run name>/        resume state, overwritten every chunk
    runs/<run name>/          publish: the selected weights of a finished run

`chunks/` and `runs/` are separate on purpose. A resume point is 572 MB of a step nobody
chose, rewritten every chunk and interesting only until the next one lands. Publishing is a
decision made once at the end. Mixing them means either the publish path fills with garbage
or the resume path gets curated, and neither is what you want at 3am on session five.

The resume checkpoint is always written as `ckpt-latest.pt` rather than under its step
number. Overwriting means the newest one is found by name, with no listing, no sorting and
no tie to break when two chunks raced. Hub commits are atomic, so a session that dies
mid-upload leaves the previous chunk intact rather than a half-written file. `state.json`
carries the step, so progress can be read without pulling 572 MB to find out.

Checkpoints could live in wandb instead, and `wandb_sync` still does that. The reason this
exists is that it puts resumability and metrics logging on separate failures: a run whose
wandb key is missing, offline or out of storage quota still leaves something the next chunk
can start from.

`frames.u8` is a decoded uint8 memmap rather than the original JPEGs. That trades repo size
for the ability to start training within a minute of a session opening, which is the
scarcer resource when the session is billed in wall clock.

Usage (KABR_HF_REPO overrides the default repo):

    python -m kabr.hf_sync push-cache --image-size 96
    python -m kabr.hf_sync pull-cache --image-size 96
    python -m kabr.hf_sync state      --run-name giraffe-96px-...   # cheap progress check
    python -m kabr.hf_sync push-run   --run-name giraffe-96px-...   # publish, not resume
    python -m kabr.hf_sync ls
"""


from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from kabr.config import Config, parse_config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

DEFAULT_REPO = "TimS-ml/kabr-video-diffusion"
REPO_TYPE = "dataset"

# What a finished run is worth publishing: the selected weights and what produced them.
# Rolling checkpoints exist to survive a crash inside one session and are not part of this.
RUN_KEEP = ("config.json", "wandb_id.txt", "best-*.pt")

# Everything a next chunk needs to carry on as if the previous one had not stopped. The
# best-*.pt record matters as much as the weights: without it the first metric block of a
# new chunk calls a worse result a new best and overwrites a better checkpoint.
CHUNK_KEEP = ("best-*.pt", "config.json", "wandb_id.txt")
LATEST = "ckpt-latest.pt"
STATE = "state.json"


def repo_id() -> str:
    return os.environ.get("KABR_HF_REPO", DEFAULT_REPO)


def chunk_prefix(cfg) -> str:
    return f"chunks/{cfg.run_name}"


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _api():
    from huggingface_hub import HfApi

    api = HfApi(token=_token())
    api.create_repo(repo_id(), repo_type=REPO_TYPE, exist_ok=True, private=False)
    return api


def _latest_checkpoint(run_dir: Path) -> Path | None:
    numbered = sorted(run_dir.glob("ckpt-[0-9]*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    if numbered:
        return numbered[-1]
    final = run_dir / "ckpt-final.pt"
    return final if final.exists() else None


# ---------------------------------------------------------------- cache
def push_cache(cfg: Config) -> None:
    src = cfg.cache_dir
    if not (src / "index.npz").exists():
        raise SystemExit(f"no cache at {src}; build it with `python -m kabr.prepare_data`")
    prefix = f"cache/{src.name}"
    size = sum(f.stat().st_size for f in src.iterdir() if f.is_file())
    print(f"uploading {src.name} ({size / 2**30:.2f} GiB) to {repo_id()}:{prefix}")
    _api().upload_folder(folder_path=str(src), path_in_repo=prefix,
                         repo_id=repo_id(), repo_type=REPO_TYPE,
                         commit_message=f"cache {src.name}")
    print("done")


def pull_cache(cfg: Config) -> Path:
    from huggingface_hub import snapshot_download

    dst = cfg.cache_dir
    name = dst.name
    if (dst / "index.npz").exists():
        print(f"cache already present at {dst}")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {repo_id()}:cache/{name} -> {dst}")
    local = snapshot_download(repo_id(), repo_type=REPO_TYPE,
                              allow_patterns=[f"cache/{name}/*"],
                              token=os.environ.get("HF_TOKEN"))
    src = Path(local) / "cache" / name
    if not (src / "index.npz").exists():
        raise SystemExit(f"{repo_id()} has no cache/{name}; push it from a machine that has one")
    # Symlink rather than copy: the hub cache already holds the only copy that matters, the
    # memmap is opened read-only, and a hosted session's disk is not big enough to want two.
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        raise SystemExit(f"{dst} exists and is not a symlink; move it aside first")
    dst.symlink_to(src, target_is_directory=True)
    print(f"cache ready at {dst} -> {src}")
    return dst


# ---------------------------------------------------------------- chunks (resume state)
def _chunk_files(run_dir: Path, checkpoint: Path) -> list[Path]:
    out = [checkpoint]
    for pattern in CHUNK_KEEP:
        out += sorted(run_dir.glob(pattern))
    return [p for p in dict.fromkeys(out) if p.is_file()]


def push_chunk(cfg, checkpoint: Path, step: int, metadata: dict | None = None) -> None:
    """Write this chunk's resume state as one commit.

    One commit rather than a file at a time, because a commit either lands whole or not at
    all. Uploading the checkpoint and the best-*.pt record separately leaves a window where
    the two disagree, and a resume that picks up in that window trains against a best record
    from a different step.
    """
    from huggingface_hub import CommitOperationAdd

    if not checkpoint.is_file():
        raise SystemExit(f"no checkpoint at {checkpoint}")
    prefix = chunk_prefix(cfg)
    state = {"step": step, "run_name": cfg.run_name,
             "updated": _now(), **(metadata or {})}
    files = _chunk_files(cfg.run_dir, checkpoint)
    ops = [CommitOperationAdd(f"{prefix}/{LATEST}", str(checkpoint))]
    ops += [CommitOperationAdd(f"{prefix}/{f.name}", str(f))
            for f in files if f != checkpoint]
    ops.append(CommitOperationAdd(f"{prefix}/{STATE}",
                                  json.dumps(state, indent=2).encode()))
    _api().create_commit(repo_id(), repo_type=REPO_TYPE, operations=ops,
                         commit_message=f"{cfg.run_name} @ step {step}")
    names = ", ".join([f"{checkpoint.name} -> {LATEST}"]
                      + [f.name for f in files if f != checkpoint])
    print(f"  pushed {repo_id()}:{prefix} step {step} [{names}]", flush=True)


def chunk_state(cfg) -> dict | None:
    """The step a run reached, without pulling the 572 MB that proves it."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id(), f"{chunk_prefix(cfg)}/{STATE}",
                               repo_type=REPO_TYPE, token=_token())
    except Exception:
        return None
    return json.loads(Path(path).read_text())


def pull_chunk(cfg) -> Path | None:
    """Restore a run's resume state into its run directory. Returns what to resume from.

    Into the run directory rather than a scratch path, because `wandb_id.txt` and the
    best-*.pt record have to be in place before training starts, not after.
    """
    from huggingface_hub import snapshot_download

    prefix = chunk_prefix(cfg)
    try:
        local = snapshot_download(repo_id(), repo_type=REPO_TYPE,
                                  allow_patterns=[f"{prefix}/*"], token=_token())
    except Exception as exc:  # a run nobody has pushed yet is the normal first case
        print(f"no chunk {prefix} on {repo_id()} ({type(exc).__name__}), starting fresh")
        return None
    src = Path(local) / "chunks" / cfg.run_name
    if not (src / LATEST).exists():
        print(f"no chunk {prefix} on {repo_id()}, starting fresh")
        return None

    import shutil
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_file():
            shutil.copy2(f, cfg.run_dir / f.name)
    state = json.loads((src / STATE).read_text()) if (src / STATE).exists() else {}
    resume = cfg.run_dir / LATEST
    print(f"pulled {prefix} (step {state.get('step')}) -> {resume}")
    return resume


def preflight(cfg) -> str | None:
    """Prove we can write before spending a session finding out we cannot.

    A token that is missing, expired, or scoped read-only fails identically to a working one
    until the first push, which is at the end of the chunk. Eleven hours is too late to learn
    this, and the check costs one small commit.
    """
    if not _token():
        return "no HF_TOKEN in the environment, so the chunk cannot push its resume state"
    from huggingface_hub import CommitOperationAdd

    # Outside the run's own prefix, so the probe does not come back down with the resume
    # state on the next pull.
    probe = {"run_name": cfg.run_name, "checked": _now()}
    try:
        _api().create_commit(
            repo_id(), repo_type=REPO_TYPE,
            operations=[CommitOperationAdd(f"chunks/_preflight/{cfg.run_name}.json",
                                           json.dumps(probe, indent=2).encode())],
            commit_message=f"preflight {cfg.run_name}")
    except Exception as exc:
        return f"HF_TOKEN cannot write to {repo_id()}: {type(exc).__name__}: {exc}"
    return None


# ---------------------------------------------------------------- runs
def push_run(cfg: Config, checkpoints: str = "none") -> None:
    """Publish a run. Rolling checkpoints are excluded by default.

    Resume comes from `chunks/`, not from here, so what this publishes is the selected
    weights of a finished run rather than its resume state. A rolling checkpoint is 572 MB
    of a step nobody chose.
    """
    run_dir = cfg.run_dir
    if not run_dir.exists():
        raise SystemExit(f"no run at {run_dir}")
    latest = _latest_checkpoint(run_dir)
    patterns = list(RUN_KEEP)
    if checkpoints == "all":
        patterns.append("ckpt-*.pt")
    elif checkpoints == "latest" and latest is not None:
        patterns.append(latest.name)
    print(f"uploading {run_dir.name} -> {repo_id()}:runs/{run_dir.name}")
    print(f"  patterns: {patterns}")
    _api().upload_folder(folder_path=str(run_dir), path_in_repo=f"runs/{run_dir.name}",
                         repo_id=repo_id(), repo_type=REPO_TYPE,
                         allow_patterns=patterns,
                         commit_message=f"run {run_dir.name} @ {latest.name if latest else 'no ckpt'}")
    print("done")


def pull_run(cfg: Config) -> Path | None:
    """Fetch a published run's files. Resume does not go through here, `pull_chunk` does."""
    from huggingface_hub import snapshot_download

    name = cfg.run_dir.name
    try:
        local = snapshot_download(repo_id(), repo_type=REPO_TYPE,
                                  allow_patterns=[f"runs/{name}/*"],
                                  token=os.environ.get("HF_TOKEN"))
    except Exception as exc:  # a run that has never been published is the normal first case
        print(f"no run {name} on {repo_id()} ({type(exc).__name__})")
        return None
    src = Path(local) / "runs" / name
    if not src.exists():
        print(f"no run {name} on {repo_id()}")
        return None

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in sorted(src.iterdir()):
        if f.is_file():
            shutil.copy2(f, cfg.run_dir / f.name)
    print(f"pulled {name}: {[f.name for f in sorted(cfg.run_dir.iterdir()) if f.is_file()]}")
    return _latest_checkpoint(cfg.run_dir)


def list_repo() -> None:
    api = _api()
    files = api.list_repo_files(repo_id(), repo_type=REPO_TYPE)
    for f in sorted(files):
        print(f)


def main() -> None:
    known = argparse.ArgumentParser(add_help=False)
    known.add_argument("action", choices=["push-cache", "pull-cache", "push-run",
                                          "pull-run", "pull-chunk", "state", "ls"])
    known.add_argument("--checkpoints", choices=["none", "latest", "all"], default="none",
                       help="rolling checkpoints to include when publishing a run")
    ns, rest = known.parse_known_args()
    if ns.action == "ls":
        return list_repo()
    cfg = parse_config(rest)
    if ns.action == "push-cache":
        push_cache(cfg)
    elif ns.action == "pull-cache":
        pull_cache(cfg)
    elif ns.action == "push-run":
        push_run(cfg, checkpoints=ns.checkpoints)
    elif ns.action == "pull-chunk":
        pull_chunk(cfg)
    elif ns.action == "state":
        state = chunk_state(cfg)
        print(json.dumps(state, indent=2) if state
              else f"no chunk pushed for {cfg.run_name}")
    else:
        pull_run(cfg)


if __name__ == "__main__":
    main()
