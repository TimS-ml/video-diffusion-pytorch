"""Drive one hosted training session on a 2x T4 box.

A session is a fixed block of wall clock that ends whether or not the run is finished, and
it keeps nothing afterwards. Everything here follows from that:

  cache from the hub          `hf_sync.pull_cache` once, symlinked out of the hub cache.
  resume from wandb           each arm runs with `--resume auto --ckpt-artifact`, so the
                              trainer pulls its own previous chunk and logs the next one.
                              The chunk stops on `stop_after_seconds`, not `train_steps`, so
                              it always writes a checkpoint before the session is cut.
  two cards, two arms         The two T4s run different configurations rather than one
                              data-parallel run. At this model size a T4 saturates around
                              batch 2 and the interesting comparison is between recipes, so
                              two independent arms buy an ablation per session where DDP
                              would buy about 1.7x on one of them.
  fp16, not bf16              T4 is sm_75. `amp_dtype: auto` resolves to fp16 there and
                              turns on the gradient scaler with it.

The arms below are the pair this round is built to compare: behaviour conditioning with
guidance against the same recipe without it. Everything else is held equal, including the
schedule shift, so the difference between them is attributable.

    python -m kabr.kaggle_runner --seconds 37800
    python -m kabr.kaggle_runner --arms conditioned --seconds 3600   # one card
    python -m kabr.kaggle_runner --list
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Held equal across arms: 96px on a 16 GB card wants batch 2 with accumulation 4, which is
# the same effective batch 8 the recorded runs used. Metric batch drops with it.
COMMON = [
    "--image-size", "96",
    "--batch-size", "2",
    "--grad-accum", "4",
    "--metric-batch", "4",
    "--train-steps", "70000",
    "--schedule-shift", "0.667",  # 64 / 96, the resolution the cosine schedule was tuned at
    "--weight-decay", "0.01",
    "--resume", "auto",       # find my own previous chunk
    "--ckpt-artifact",        # and leave the next one behind
    "--no-compile-model",     # a failed compile costs the whole session; turn it on once a
                              # session is known to work end to end
]

ARMS: dict[str, list[str]] = {
    "conditioned": COMMON + [
        "--cond-behaviour",
        "--null-cond-prob", "0.1",
        "--cond-scale", "2.0",
        "--metric-class-samples", "64",
    ],
    # The control. Same schedule, same optimiser, same protocol, no condition - without it
    # any improvement is attributable to four changes at once.
    "control": COMMON + [
        "--run-name", "giraffe-96px-16f-d64-pred_v-minsnr5-shift0.667-control",
    ],
}


def _env_for(gpu: int) -> dict:
    env = dict(os.environ)
    repo = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join([str(repo), str(repo / "experiments"),
                                         env.get("PYTHONPATH", "")])
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def _pump(stream, tag: str, sink: list[str]) -> None:
    for line in iter(stream.readline, ""):
        sink.append(line)
        print(f"[{tag}] {line}", end="", flush=True)
    stream.close()


def run_arm(name: str, gpu: int, seconds: int, extra: list[str]) -> subprocess.Popen:
    args = [sys.executable, "-u", "-m", "kabr.train",
            *ARMS[name], "--stop-after-seconds", str(seconds), *extra]
    print(f"launching {name} on gpu {gpu}: {' '.join(args[3:])}", flush=True)
    return subprocess.Popen(args, env=_env_for(gpu), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)


def arm_config(name: str, extra: list[str]):
    """The Config an arm will build, so the cache pull addresses the directory it reads."""
    from kabr.config import parse_config

    return parse_config(ARMS[name] + extra)


def visible_gpus() -> int:
    import torch

    return torch.cuda.device_count()


def session(names: list[str], seconds: int, extra: list[str], skip_cache: bool = False) -> int:
    from kabr import hf_sync

    # A session does not always come up with the accelerator that was asked for. One arm per
    # card and drop the rest: two arms sharing a 16 GB T4 would either OOM or halve both.
    cards = visible_gpus()
    if cards < len(names):
        print(f"{cards} gpu(s) visible, dropping {names[cards:]}")
        names = names[:cards]
    if not names:
        raise SystemExit("no gpu visible, nothing to run")

    if not skip_cache:
        hf_sync.pull_cache(arm_config(names[0], extra))

    procs, logs = {}, {}
    for gpu, n in enumerate(names):
        logs[n] = []
        procs[n] = run_arm(n, gpu, seconds, extra)
    threads = [threading.Thread(target=_pump, args=(p.stdout, n, logs[n]), daemon=True)
               for n, p in procs.items()]
    for t in threads:
        t.start()

    t0 = time.time()
    codes = {n: p.wait() for n, p in procs.items()}
    for t in threads:
        t.join(timeout=10)
    print(f"\nall arms finished in {(time.time() - t0) / 60:.1f} min: {codes}", flush=True)

    for n in names:
        status = [l for l in logs[n] if l.startswith("KABR_STATUS")]
        art = [l for l in logs[n] if "logged artifact" in l]
        print(f"{n}: exit {codes[n]}  {status[-1].strip() if status else 'no status line'}")
        print(f"  {art[-1].strip() if art else 'no artifact logged - this chunk is not resumable'}")
    return max(codes.values()) if codes else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="one hosted training session")
    ap.add_argument("--arms", default="conditioned,control",
                    help="comma separated, one per gpu, in gpu order")
    ap.add_argument("--seconds", type=int, default=37800,
                    help="wall clock for the chunk; leave the session margin to push")
    ap.add_argument("--skip-cache", action="store_true",
                    help="the cache is already local, do not touch the hub")
    ap.add_argument("--list", action="store_true")
    ns, rest = ap.parse_known_args()
    if ns.list:
        print(json.dumps({k: " ".join(v) for k, v in ARMS.items()}, indent=2))
        return
    names = [a.strip() for a in ns.arms.split(",") if a.strip()]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}, expected any of {sorted(ARMS)}")
    raise SystemExit(session(names, ns.seconds, rest, skip_cache=ns.skip_cache))


if __name__ == "__main__":
    main()
