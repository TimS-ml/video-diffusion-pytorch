"""Run KABR video diffusion training on a rented Modal GPU.

The local box lost its eGPU mid-run, so the same run continues here from its last
checkpoint. Two things follow from renting the GPU instead of owning it:

  billed by the hour   A budget ledger on the artifact volume records what every chunk
                       spent. A chunk that would take the run past `BUDGET_USD` refuses
                       to start rather than discovering the overrun afterwards.
  killed on a timeout  No single invocation may outlive the platform's function timeout,
                       so the run is cut into chunks. Each chunk resumes from the newest
                       checkpoint on the volume and stops on its own wall clock, which
                       makes the driver loop restartable: if it dies, run it again.

Nothing about the training recipe changes. `train_steps` still sets the cosine horizon,
the metric protocol is untouched so the numbers stay comparable with the steps already
logged, and only `ckpt_keep` drops - volume storage costs money and the local default of
15 was sized for a machine that could lose power without warning.

Usage:
    modal run experiments/kabr/modal_runner.py::smoke      # 60 steps, measure s/step
    modal run experiments/kabr/modal_runner.py::main       # chunked run to completion
    modal run experiments/kabr/modal_runner.py::status     # steps done, dollars spent
    modal run experiments/kabr/modal_runner.py::fetch      # pull run dir back down
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

APP_NAME = "kabr-video-diffusion"
DATA_VOLUME = "kabr-data"    # the decoded uint8 memmap, uploaded once
RUNS_VOLUME = "kabr-runs"    # checkpoints, media, wandb state, spend ledger

GPU_TYPE = "L4"              # 24 GB, same Ada generation as the card this run started on
GPU_HOURLY_USD = 0.80
BUDGET_USD = 30.0

RUN_NAME = "giraffe-96px-16f-d64-pred_v-minsnr5"
IMAGE_SIZE = 96
CKPT_KEEP = 3

# One chunk is short enough that a failure costs little and the ledger gets a say between
# chunks, long enough that per-chunk overhead (container start, staging, compile warmup:
# measured at two to nine minutes) stays near one percent. The slack on top is what the
# platform timeout allows beyond the chunk itself: the wall-clock check only runs between
# steps, so a metric block that starts just before the deadline has to be able to finish.
CHUNK_SECONDS = 5 * 60 * 60 + 30 * 60
STARTUP_SLACK = 60 * 60

# Training flags for this run, mirroring the local invocation.
TRAIN_ARGS = [
    "--image-size", str(IMAGE_SIZE),
    "--batch-size", "4",
    "--grad-accum", "2",
    "--metric-batch", "8",
    "--lr-schedule", "cosine",
    "--metric-protocol", "v2",
]

# Container layout. `out_root` sits on local disk and only its `runs/` is a symlink onto
# the volume: the dataloader takes random windows out of a 4 GiB memmap, which a
# network-backed filesystem serves badly, so the cache is staged to local disk once per
# chunk and read from page cache after that.
WORKSPACE = "/workspace"
OUT_ROOT = "/root/out"
VOL_RUNS = "/vol/runs"
VOL_DATA = "/vol/data"
LEDGER = f"{VOL_RUNS}/spend.jsonl"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        # Pinned to what the run was training under locally, so resuming a checkpoint
        # does not also silently change the framework underneath it.
        "torch==2.10.0",
        "torchvision==0.25.0",
        "einops==0.8.2",
        "einops-exts==0.0.4",
        "rotary-embedding-torch==0.9.1",
        "ema-pytorch==0.7.9",
        "wandb==0.27.0",
        "torchmetrics==1.9.0",
        "torch-fidelity==0.4.0",
        "scipy==1.17.1",
        "numpy==2.2.6",
        "pillow==12.1.1",
        "huggingface-hub==1.14.0",
        "tqdm",
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{WORKSPACE}:{WORKSPACE}/experiments",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # I3D and InceptionV3 land here, on the volume, so the metric block downloads
        # them once for the whole run instead of once per chunk.
        "HF_HOME": f"{VOL_RUNS}/hf",
        "TORCH_HOME": f"{VOL_RUNS}/torch",
    })
)

# This module is imported again inside the container, where the repository it was written
# from does not exist. The source mounts are a local-side concern only.
if modal.is_local():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    image = (
        image
        .add_local_dir(REPO_ROOT / "video_diffusion_pytorch",
                       f"{WORKSPACE}/video_diffusion_pytorch",
                       ignore=["__pycache__", "*.pyc"], copy=False)
        .add_local_dir(REPO_ROOT / "experiments" / "kabr",
                       f"{WORKSPACE}/experiments/kabr",
                       ignore=["__pycache__", "*.pyc"], copy=False)
    )

app = modal.App(APP_NAME, image=image)

data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
runs_vol = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
wandb_secret = modal.Secret.from_name("kabr-wandb")


# ---------------------------------------------------------------- container helpers
def _latest_checkpoint(run_dir: Path) -> Path | None:
    """Newest numbered checkpoint, or None for a run that has not started."""
    ckpts = sorted(run_dir.glob("ckpt-[0-9]*.pt"),
                   key=lambda p: int(p.stem.split("-")[-1]))
    return ckpts[-1] if ckpts else None


def _stage_cache(species: str, image_size: int) -> None:
    """Copy the memmap off the volume onto local disk, once per container."""
    name = f"{species}_{image_size}px"
    src = Path(VOL_DATA) / "cache" / name
    dst = Path(OUT_ROOT) / "cache" / name
    if not src.exists():
        raise RuntimeError(f"no cache at {src} - upload it with the `upload_data` entrypoint")
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        target = dst / f.name
        if target.exists() and target.stat().st_size == f.stat().st_size:
            continue
        t0 = time.time()
        shutil.copy2(f, target)
        mb = target.stat().st_size / 1e6
        print(f"staged {f.name} ({mb:.0f} MB) in {time.time() - t0:.1f}s", flush=True)


def _link_runs() -> Path:
    """Point `$KABR_OUT_ROOT/runs` at the volume so checkpoints persist."""
    runs = Path(OUT_ROOT) / "runs"
    runs.parent.mkdir(parents=True, exist_ok=True)
    vol_runs = Path(VOL_RUNS) / "runs"
    vol_runs.mkdir(parents=True, exist_ok=True)
    if not runs.exists():
        os.symlink(str(vol_runs), str(runs), target_is_directory=True)
    return vol_runs


def _read_ledger() -> list[dict]:
    p = Path(LEDGER)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _spent_usd() -> float:
    return sum(e.get("usd", 0.0) for e in _read_ledger())


def _append_ledger(entry: dict) -> None:
    Path(LEDGER).parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


class _Committer:
    """Commit the volume on a timer, so a container that dies loses minutes, not hours.

    Checkpoint writes are atomic (temp file then rename), so a commit that lands in the
    middle of one captures the temp file and never a half-written checkpoint.
    """

    def __init__(self, vol, every: float = 600.0):
        self.vol, self.every = vol, every
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(self.every):
            try:
                self.vol.commit()
            except Exception as exc:  # a failed commit must not take the run down
                print(f"volume commit failed: {exc}", flush=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.vol.commit()


def _run_training(extra_args: list[str], run_name: str, chunk_seconds: float,
                  enforce_budget: bool) -> dict:
    t0 = time.time()
    if enforce_budget:
        spent = _spent_usd()
        if spent >= BUDGET_USD:
            print(f"budget exhausted: ${spent:.2f} of ${BUDGET_USD:.2f}", flush=True)
            return {"launched": False, "reason": "budget", "spent_usd": spent}

    _stage_cache("giraffe", IMAGE_SIZE)
    vol_runs = _link_runs()
    run_dir = vol_runs / RUN_NAME

    args = ["python", "-m", "kabr.train", *TRAIN_ARGS, *extra_args,
            "--run-name", run_name,
            "--ckpt-keep", str(CKPT_KEEP)]
    if not any(a == "--resume" for a in extra_args):
        ckpt = _latest_checkpoint(run_dir)
        if ckpt is not None:
            args += ["--resume", str(ckpt)]
            print(f"resuming from {ckpt.name}", flush=True)
        else:
            print("no checkpoint on the volume, starting from scratch", flush=True)
    if chunk_seconds:
        args += ["--stop-after-seconds", str(chunk_seconds)]

    env = os.environ.copy()
    env["KABR_OUT_ROOT"] = OUT_ROOT
    # Only prepare_data reads the dataset root; training works off the staged cache.
    env["KABR_DATA_ROOT"] = env.get("KABR_DATA_ROOT", OUT_ROOT)
    if not env.get("WANDB_API_KEY"):
        env["WANDB_MODE"] = "disabled"

    log_dir = Path(VOL_RUNS) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"

    print(f"launching: {' '.join(args)}", flush=True)
    status: dict = {}
    with _Committer(runs_vol), open(log_path, "a") as log_f:
        log_f.write(f"\n=== chunk start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        proc = subprocess.Popen(args, cwd=WORKSPACE, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for raw in iter(proc.stdout.readline, b""):
            text = raw.decode("utf-8", errors="replace")
            print(text, end="")
            log_f.write(text)
            log_f.flush()  # the volume copy is the only view once the client is gone
            if text.startswith("KABR_STATUS "):
                status = json.loads(text[len("KABR_STATUS "):])
        proc.wait()

    elapsed = time.time() - t0
    usd = elapsed / 3600.0 * GPU_HOURLY_USD
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_name": run_name, "gpu": GPU_TYPE,
        "seconds": round(elapsed, 1), "usd": round(usd, 4),
        "returncode": proc.returncode, **status,
    }
    if enforce_budget:
        _append_ledger(entry)
    runs_vol.commit()

    print(f"chunk used {elapsed/3600:.2f} h = ${usd:.2f}; "
          f"total ${_spent_usd():.2f} of ${BUDGET_USD:.2f}", flush=True)
    return {"launched": True, **entry}


def _step_on_volume() -> int | None:
    run_dir = Path(VOL_RUNS) / "runs" / RUN_NAME
    ckpt = _latest_checkpoint(run_dir) if run_dir.exists() else None
    return int(ckpt.stem.split("-")[-1]) if ckpt else None


# ---------------------------------------------------------------- remote functions
@app.function(
    gpu=GPU_TYPE, cpu=8.0, memory=32 * 1024,
    timeout=CHUNK_SECONDS + STARTUP_SLACK,
    volumes={VOL_RUNS: runs_vol, VOL_DATA: data_vol},
    secrets=[wandb_secret],
)
def train_chunk(train_steps: int, chunk_seconds: int = CHUNK_SECONDS,
                chunks_left: int = 20, retries_left: int = 3) -> dict:
    """One chunk, which queues the next one itself.

    The chain lives here rather than in a local loop because the link to this machine is
    not reliable enough to hold a thirty hour run together: a dropped connection kills an
    ephemeral app and any driver looping inside it. Spawning the successor from inside the
    container means the client is only needed to start the first chunk.
    """
    before = _step_on_volume()
    try:
        result = _run_training(
            extra_args=["--train-steps", str(train_steps)],
            run_name=RUN_NAME, chunk_seconds=chunk_seconds, enforce_budget=True,
        )
    except Exception as exc:
        result = {"launched": True, "error": repr(exc), "returncode": -1}
    after = _step_on_volume()

    result["step_before"], result["step_after"] = before, after
    progressed = after is not None and (before is None or after > before)
    failed = result.get("returncode", 0) != 0

    # A chunk that neither finished nor moved the step counter is a crash loop in the
    # making; spend a bounded number of retries on it and then stop rather than burning
    # the budget on containers that cannot train.
    if failed or not progressed:
        retries_left -= 1
        print(f"chunk made no progress (rc={result.get('returncode')}), "
              f"{retries_left} retries left", flush=True)

    if result.get("launched") is False:
        stop = "budget"
    elif result.get("done"):
        stop = "done"
    elif chunks_left <= 1:
        stop = "max_chunks"
    elif retries_left < 0:
        stop = "retries_exhausted"
    elif _spent_usd() >= BUDGET_USD:
        stop = "budget"
    else:
        stop = None

    if stop is None:
        train_chunk.spawn(train_steps=train_steps, chunk_seconds=chunk_seconds,
                          chunks_left=chunks_left - 1, retries_left=retries_left)
        print(f"queued next chunk ({chunks_left - 1} left)", flush=True)
    else:
        print(f"chain stopped: {stop}", flush=True)
    result["chain_stop"] = stop
    return result


@app.function(
    gpu=GPU_TYPE, cpu=8.0, memory=32 * 1024, timeout=45 * 60,
    volumes={VOL_RUNS: runs_vol, VOL_DATA: data_vol},
)
def smoke(steps: int = 60, train_steps: int = 80_000) -> dict:
    """Resume the real checkpoint into a throwaway run dir and time `steps` steps.

    Checks the two things that have to hold before spending the budget: that a locally
    written checkpoint loads here, and what a step actually costs on this GPU.
    """
    vol_runs = _link_runs()
    ckpt = _latest_checkpoint(vol_runs / RUN_NAME)
    if ckpt is None:
        raise RuntimeError(f"no checkpoint under {vol_runs / RUN_NAME}")
    blob_step = int(ckpt.stem.split("-")[-1])
    return _run_training(
        extra_args=["--train-steps", str(train_steps),
                    "--resume", str(ckpt),
                    "--stop-at-step", str(blob_step + steps),
                    "--wandb-mode", "disabled"],
        run_name=f"smoke-{IMAGE_SIZE}px", chunk_seconds=0, enforce_budget=False,
    )


@app.function(volumes={VOL_RUNS: runs_vol, VOL_DATA: data_vol}, timeout=10 * 60)
def report() -> dict:
    """Cheap, GPU-free look at where the run and the budget stand."""
    run_dir = Path(VOL_RUNS) / "runs" / RUN_NAME
    ckpt = _latest_checkpoint(run_dir) if run_dir.exists() else None
    ledger = _read_ledger()
    cache = Path(VOL_DATA) / "cache" / f"giraffe_{IMAGE_SIZE}px"
    return {
        "step": int(ckpt.stem.split("-")[-1]) if ckpt else None,
        "latest_ckpt": ckpt.name if ckpt else None,
        "checkpoints": sorted(p.name for p in run_dir.glob("*.pt")) if run_dir.exists() else [],
        "chunks": len(ledger),
        "spent_usd": round(_spent_usd(), 2),
        "budget_usd": BUDGET_USD,
        "gpu_hours": round(sum(e.get("seconds", 0) for e in ledger) / 3600, 2),
        "cache_files": sorted(p.name for p in cache.iterdir()) if cache.exists() else [],
        "last_chunk": ledger[-1] if ledger else None,
    }


# ---------------------------------------------------------------- local entrypoints
@app.local_entrypoint()
def status():
    print(json.dumps(report.remote(), indent=2))


@app.local_entrypoint()
def smoke_test(steps: int = 60):
    """Spawn the deployed smoke function, then wait on its result.

    Spawned rather than called inline so a dropped connection loses the printout, not the
    container: reconnecting and polling `status` still finds what it did.
    """
    fn = modal.Function.from_name(APP_NAME, "smoke")
    call = fn.spawn(steps=steps)
    print(f"smoke running as {call.object_id}")
    print(json.dumps(call.get(), indent=2))


@app.local_entrypoint()
def main(train_steps: int = 80_000, chunk_seconds: int = CHUNK_SECONDS,
         max_chunks: int = 20):
    """Start the chain and return. Deploy first, then poll with `status`.

        modal deploy experiments/kabr/modal_runner.py
        modal run experiments/kabr/modal_runner.py::main --train-steps 80000

    The client is not needed after this call: chunks queue their own successors.
    """
    before = report.remote()
    print(f"step {before['step']} | spent ${before['spent_usd']:.2f} of ${BUDGET_USD:.2f}")
    if before["step"] is not None and before["step"] >= train_steps:
        print("target already reached")
        return
    fn = modal.Function.from_name(APP_NAME, "train_chunk")
    call = fn.spawn(train_steps=train_steps, chunk_seconds=chunk_seconds,
                    chunks_left=max_chunks, retries_left=3)
    print(f"chain started, first chunk {call.object_id}")


@app.local_entrypoint()
def fetch(dest: str = "./modal-runs"):
    """Pull checkpoints, media and logs back down."""
    Path(dest).mkdir(parents=True, exist_ok=True)
    subprocess.run(["modal", "volume", "get", RUNS_VOLUME, "/", dest, "--force"], check=True)
    print(f"downloaded to {dest}")
