# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # KABR video diffusion, one Kaggle session
#
# This file is the entry point of a Kaggle **script kernel**, submitted with
# `experiments/kabr/kaggle/submit.py`. It is plain Python with no shell magics, so the same
# file also converts to a notebook when something needs stepping through:
#
#     jupytext --to notebook experiments/kabr/kaggle/kabr_session.py
#
# Batch is the right default here. A batch kernel gets 12 hours against an interactive
# session's 9, it runs after the terminal is closed, and quota is charged for the time the
# code actually runs rather than for the time a browser tab stays open. What it gives up is
# live output, which costs nothing in this case because the run reports to wandb anyway.
#
# A session keeps nothing when it ends, so a run is a chain of chunks and each chunk finds
# its own predecessor. Two stores, split by what the thing actually is:
#
# - the clip cache is large, static and shared by every run, so it is a HF dataset
# - the checkpoints are per run and only mean anything next to the curves that produced
#   them, so they are wandb artifacts
#
# Internet has to be on. Of the two secrets, only one actually gates anything:
#
# - `WANDB_API_KEY` is required. Without it the chunk trains and then has nowhere to leave
#   its checkpoint, so it cannot be resumed from and the session is a smoke test.
# - `HF_TOKEN` is optional, because the dataset holding the cache is public. It is only
#   needed to write back to it.
#
# Secrets cannot be attached over the API, so the first push has to be followed by one visit
# to the kernel's editor page.

# %%
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/TimS-ml/video-diffusion-pytorch"
BRANCH = os.environ.get("KABR_BRANCH", "feature/kabr-conditioning")
SRC = Path("/kaggle/working/video-diffusion-pytorch")
# Everything under /kaggle/working is saved as kernel output. The cache is 4 GB and each
# checkpoint is 570 MB, and both already have a home elsewhere, so they go to scratch.
SCRATCH = Path("/kaggle/temp")

# A batch kernel is cut at 12 hours. Stop the chunk early enough to write a checkpoint and
# push it as an artifact, which is the only thing that survives the session.
CHUNK_SECONDS = int(os.environ.get("KABR_CHUNK_SECONDS", 39_600))  # 11 h
ARMS = os.environ.get("KABR_ARMS", "conditioned,control")
IMAGE_SIZE = int(os.environ.get("KABR_IMAGE_SIZE", 96))
# Forwarded verbatim to every arm. A short smoke session wants the expensive blocks pulled
# forward - the metric block is where a 16 GB card is most likely to run out of memory, and
# a chunk that never reaches one has not tested the thing most likely to fail.
EXTRA = os.environ.get("KABR_EXTRA", "").split()


def sh(*args, **kw):
    """Run a command and fail loudly. No shell magics, so this file stays a valid script."""
    print("$", " ".join(str(a) for a in args), flush=True)
    return subprocess.run([str(a) for a in args], check=True, **kw)


# %% [markdown]
# ## Environment
#
# The bf16 line is the one that matters. A T4 is sm_75 and cannot do bf16, so `amp_dtype:
# auto` resolves to fp16 and brings the gradient scaler with it. Pinning bf16 on this card
# is not an error in torch, it is a very slow emulation.

# %%
sh("nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader")

import torch

print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| devices", torch.cuda.device_count())
print("bf16 supported:", torch.cuda.is_bf16_supported(),
      "-> amp_dtype auto picks", "bf16" if torch.cuda.is_bf16_supported() else "fp16")

# %%
# Only what the Kaggle image does not already carry.
sh(sys.executable, "-m", "pip", "install", "-q",
   "ema-pytorch", "rotary-embedding-torch", "einops-exts", "hf-transfer")

# %%
if not SRC.exists():
    sh("git", "clone", "--depth", "1", "-b", BRANCH, REPO, SRC)
else:
    sh("git", "-C", SRC, "fetch", "--depth", "1", "origin", BRANCH)
    sh("git", "-C", SRC, "reset", "--hard", f"origin/{BRANCH}")
sh("git", "-C", SRC, "log", "--oneline", "-1")

# %% [markdown]
# ## Secrets and paths
#
# `KABR_DATA_ROOT` stays unset on purpose. It points at the raw KABR JPEGs, which only the
# machine that built the cache ever needed, and the config only asks for it when something
# actually tries to decode them.

# %%
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()


def secret(name: str) -> str | None:
    try:
        return secrets.get_secret(name)
    except Exception:
        return None


# Reading the cache off a public dataset needs no token, so a missing HF_TOKEN is only a
# problem for writing back, which this session does not do.
if token := secret("HF_TOKEN"):
    os.environ["HF_TOKEN"] = token
else:
    print("no HF_TOKEN secret: public reads still work, pushing to the dataset will not")

if key := secret("WANDB_API_KEY"):
    os.environ["WANDB_API_KEY"] = key
else:
    # Without wandb there is nowhere to put the checkpoint, so the chunk cannot be resumed
    # from and the session is a smoke test rather than a step of the run.
    os.environ["WANDB_MODE"] = "offline"
    print("no WANDB_API_KEY secret: logging offline, THIS CHUNK WILL NOT BE RESUMABLE")

os.environ["KABR_OUT_ROOT"] = str(SCRATCH / "kabr_out")
os.environ["KABR_HF_REPO"] = os.environ.get("KABR_HF_REPO", "TimS-ml/kabr-video-diffusion")
os.environ["HF_HOME"] = str(SCRATCH / "hf")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTHONPATH"] = f"{SRC}:{SRC}/experiments"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path[:0] = [str(SRC), str(SRC / "experiments")]
Path(os.environ["KABR_OUT_ROOT"]).mkdir(parents=True, exist_ok=True)
print("out root", os.environ["KABR_OUT_ROOT"], "| hf home", os.environ["HF_HOME"])

# %% [markdown]
# ## Pull the cache
#
# About 4 GB at 96px. It is symlinked out of the hub cache rather than copied, because the
# memmap is opened read-only and the session disk is not big enough to want two copies.

# %%
from kabr import hf_sync
from kabr.config import Config
from kabr.data import ClipIndex

cfg = Config(image_size=IMAGE_SIZE)
t0 = time.time()
hf_sync.pull_cache(cfg)
print(f"cache ready in {time.time() - t0:.0f}s")

idx = ClipIndex(cfg.cache_dir, 16, 4)
print("cache version", idx.version,
      "| train", len(idx.usable("train")), "| val", len(idx.usable("val")),
      "| labelled frames", f"{(idx.frame_labels != 255).mean() * 100:.1f}%")

# %% [markdown]
# ## Train
#
# One arm per card, both stopping on wall clock, output interleaved and tagged. Each arm
# resumes from its own `latest` artifact, so resubmitting this kernel continues the run
# rather than restarting it. If the session comes up with one GPU instead of two, the runner
# drops the extra arms rather than oversubscribing a card.
#
# Expect roughly 2.5-3 s/step per T4 at 96px, so an 11 hour chunk is on the order of 14k
# steps and the 70k horizon is about five sessions per arm.

# %%
rc = subprocess.run(
    [sys.executable, "-u", "-m", "kabr.kaggle_runner",
     "--arms", ARMS, "--seconds", str(CHUNK_SECONDS), "--image-size", str(IMAGE_SIZE),
     *EXTRA],
    cwd=str(SRC), env=os.environ.copy(),
).returncode
print("session exit", rc, flush=True)

# %% [markdown]
# ## What came back
#
# The artifact push happens inside the trainer, at the end of the chunk and before wandb is
# closed, so a chunk that ran out of wall clock still leaves its progress behind. This is the
# receipt, and the only thing worth reading in the kernel log afterwards.

# %%
from kabr import wandb_sync
from kabr.kaggle_runner import arm_config

for arm in ARMS.split(","):
    arm_cfg = arm_config(arm, ["--image-size", str(IMAGE_SIZE), *EXTRA])
    print(f"{arm}: artifact {wandb_sync.artifact_name(arm_cfg)}"
          f" | local {wandb_sync.latest_local(arm_cfg.run_dir)}")

if rc != 0:
    raise SystemExit(rc)
