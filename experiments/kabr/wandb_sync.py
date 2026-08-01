"""Checkpoints as wandb artifacts, so a chunked run can resume on a different machine.

A hosted session keeps nothing when it ends, so the resume state has to live somewhere the
next session can reach. wandb is already holding this run's curves; putting the checkpoint
in the same place keeps one identity for a run instead of two, and artifact storage is
content addressed, so the `best-*.pt` files that did not change between chunks cost nothing
to include again.

One artifact version per chunk, named `ckpt-<run name>`, holding:

    ckpt-<step>.pt   the resume state: live weights, EMA, optimiser, scaler, best record
    best-*.pt        the selected checkpoints, so a new chunk's first metric block cannot
                     overwrite a better result it never saw
    config.json      what produced them
    wandb_id.txt     read before `wandb.init`, so chunks land on one continuous set of curves

Aliases are `latest` and `step-<n>`. Resume takes `latest`; `step-<n>` is what you name when
you want a specific point back.
"""

from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_TYPE = "model"
# The checkpoint plus everything needed to carry on as if the previous chunk had not stopped.
RESUME_FILES = ("best-*.pt", "config.json", "wandb_id.txt")


def artifact_name(cfg) -> str:
    return f"ckpt-{cfg.run_name}"


def _entity() -> str | None:
    import wandb

    if os.environ.get("WANDB_ENTITY"):
        return os.environ["WANDB_ENTITY"]
    try:
        return wandb.Api().default_entity
    except Exception:
        return None


def _resume_candidates(run_dir: Path) -> list[Path]:
    out = []
    for pattern in RESUME_FILES:
        out += sorted(run_dir.glob(pattern))
    return [p for p in out if p.is_file()]


def latest_local(run_dir: Path) -> Path | None:
    """Newest numbered checkpoint on disk, falling back to the one a finished run writes."""
    numbered = sorted(run_dir.glob("ckpt-[0-9]*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    if numbered:
        return numbered[-1]
    final = run_dir / "ckpt-final.pt"
    return final if final.exists() else None


def push_checkpoint(cfg, run, checkpoint: Path, step: int, metadata: dict | None = None) -> None:
    """Log one artifact version holding `checkpoint` and the run's resume files."""
    import wandb

    art = wandb.Artifact(artifact_name(cfg), type=ARTIFACT_TYPE,
                         metadata={"step": step, "run_name": cfg.run_name, **(metadata or {})})
    files = [checkpoint, *_resume_candidates(cfg.run_dir)]
    for f in dict.fromkeys(files):  # the checkpoint can also match a glob above
        art.add_file(str(f))
    run.log_artifact(art, aliases=["latest", f"step-{step}"])
    names = ", ".join(f.name for f in dict.fromkeys(files))
    print(f"  logged artifact {artifact_name(cfg)}:step-{step} [{names}]", flush=True)


def pull_checkpoint(cfg, alias: str = "latest") -> Path | None:
    """Download the newest artifact into the run directory. Returns what to resume from.

    Downloading into the run directory rather than a scratch path is deliberate: it also
    restores `wandb_id.txt` and the `best-*.pt` record, both of which have to be in place
    before training starts rather than after.
    """
    import wandb

    entity = _entity()
    if entity is None:
        print("no wandb entity available, cannot pull a checkpoint")
        return None
    ref = f"{entity}/{cfg.wandb_project}/{artifact_name(cfg)}:{alias}"
    try:
        art = wandb.Api().artifact(ref, type=ARTIFACT_TYPE)
    except Exception as exc:
        print(f"no artifact {ref} ({type(exc).__name__}), starting fresh")
        return None
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    art.download(root=str(cfg.run_dir))
    resume = latest_local(cfg.run_dir)
    print(f"pulled {ref} (step {art.metadata.get('step')}) -> {resume}")
    return resume


def resolve_resume(cfg) -> str:
    """Turn `--resume auto` into a path: the hub artifact, else whatever is already on disk."""
    if cfg.resume != "auto":
        return cfg.resume
    remote = pull_checkpoint(cfg)
    if remote is not None:
        return str(remote)
    local = latest_local(cfg.run_dir)
    if local is not None:
        print(f"no artifact, resuming from local {local}")
        return str(local)
    print("nothing to resume from, starting at step 0")
    return ""
