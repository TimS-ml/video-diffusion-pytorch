"""The prepared clip cache on a HF dataset repo, so a hosted session can start from it.

Training runs in fixed blocks of wall clock on machines that keep nothing afterwards. Two
things have to survive between them, and they belong in different places:

    the cache        large, static, shared by every run, and useful to anyone reproducing
                     this work. That is a dataset, and it lives here.
    the checkpoints  per run, rewritten every chunk, only meaningful next to the curves
                     that produced them. Those are wandb artifacts, see `wandb_sync`.

    cache/<slug>_<size>px/    frames.u8, index.npz, manifest.json
    runs/<run name>/          the publish path: final weights for a finished run

`frames.u8` is a decoded uint8 memmap rather than the original JPEGs. That trades repo size
for the ability to start training within a minute of a session opening, which is the
scarcer resource when the session is billed in wall clock.

Usage (KABR_HF_REPO overrides the default repo):

    python -m kabr.hf_sync push-cache --image-size 96
    python -m kabr.hf_sync pull-cache --image-size 96
    python -m kabr.hf_sync push-run   --run-name giraffe-96px-...   # publish, not resume
    python -m kabr.hf_sync ls
"""


from __future__ import annotations

import argparse
import os
from pathlib import Path

from kabr.config import Config, parse_config

DEFAULT_REPO = "TimS-ml/kabr-video-diffusion"
REPO_TYPE = "dataset"

# What a finished run is worth publishing: the selected weights and what produced them.
# Rolling checkpoints exist to survive a crash inside one session and are not part of this.
RUN_KEEP = ("config.json", "wandb_id.txt", "best-*.pt")


def repo_id() -> str:
    return os.environ.get("KABR_HF_REPO", DEFAULT_REPO)


def _api():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
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


# ---------------------------------------------------------------- runs
def push_run(cfg: Config, checkpoints: str = "none") -> None:
    """Publish a run. Rolling checkpoints are excluded by default.

    Resume does not come from here any more, it comes from the wandb artifact, so what this
    publishes is the selected weights of a finished run rather than its resume state. A
    rolling checkpoint is 570 MB of a step nobody chose.
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
    """Fetch a published run's files. Resume does not go through here, `wandb_sync` does."""
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
                                          "pull-run", "ls"])
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
    else:
        pull_run(cfg)


if __name__ == "__main__":
    main()
