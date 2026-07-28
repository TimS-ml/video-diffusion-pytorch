"""Run configuration for KABR video diffusion.

Paths never appear as literals in this repository. `data_root` and `out_root` come from
the environment (`KABR_DATA_ROOT`, `KABR_OUT_ROOT`) and can be overridden per invocation
on the command line.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SPECIES_PREFIX = {"giraffe": "G", "zebra_grevys": "ZG", "zebra_plains": "ZP"}


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    species: str = "giraffe"
    image_size: int = 64
    num_frames: int = 16
    frame_stride: int = 4  # 29.97 fps / 4 = 7.49 fps, 16 frames spans 2.14 s
    horizontal_flip: bool = True

    # ---- model ------------------------------------------------------------
    dim: int = 64
    dim_mults: tuple[int, ...] = (1, 2, 4, 8)
    attn_heads: int = 8
    attn_dim_head: int = 32

    # ---- diffusion --------------------------------------------------------
    timesteps: int = 1000
    objective: str = "pred_v"
    loss_type: str = "l2"
    min_snr_loss_weight: bool = True
    min_snr_gamma: float = 5.0
    sampling_timesteps: int = 50  # DDIM
    ddim_sampling_eta: float = 0.0

    # ---- optimisation -----------------------------------------------------
    batch_size: int = 4
    grad_accum: int = 2  # effective batch 8
    lr: float = 1e-4
    adam_betas: tuple[float, float] = (0.9, 0.99)
    warmup_steps: int = 1000
    # Holding the peak rate for the whole run cost the first attempt its best weights: fvd
    # bottomed out at step 60k and then climbed for the next 80k with nothing in the
    # schedule to settle it. Decay is the default now, and "constant" reproduces the old
    # behaviour for anyone rerunning that config.
    lr_schedule: str = "cosine"  # cosine | constant
    lr_final_ratio: float = 0.05  # cosine floor, as a fraction of lr
    max_grad_norm: float = 1.0
    train_steps: int = 300_000
    seed: int = 0

    ema_decay: float = 0.999
    ema_update_every: int = 10
    ema_update_after_step: int = 1000

    # ---- runtime ----------------------------------------------------------
    compile_model: bool = True
    num_workers: int = 4

    # A rented GPU is billed by the hour and its scheduler kills the container at a fixed
    # timeout, so a long run has to be cut into chunks that each stop on their own terms and
    # leave a checkpoint behind for the next one. Neither knob touches the lr schedule:
    # `train_steps` still sets the cosine horizon, these only say when this particular
    # process stops walking it. Both default to off, so a local run is unaffected.
    stop_after_seconds: float = 0.0
    stop_at_step: int = 0

    # ---- evaluation -------------------------------------------------------
    # L1: cheap deterministic held-out loss
    eval_every: int = 500
    eval_clips: int = 32  # per split
    eval_timesteps: int = 4  # stratified t per clip

    # L2: generative metrics. The whole tuple is the frozen protocol - FVD/FID at
    # n_sample=256 is biased and only comparable against runs using these exact values.
    metric_every: int = 10_000
    metric_samples: int = 256
    metric_batch: int = 16
    metric_seed: int = 1234
    metric_protocol: str = "v1"

    sample_every: int = 2_000
    sample_rows: int = 4

    # A slower, better-looking inference than the one the metrics use, purely to watch the
    # model with your own eyes. DDIM at 50 steps is a training-loop compromise; this shows
    # what the same weights do when sampling is not the bottleneck.
    preview_every: int = 10_000
    preview_rows: int = 2
    preview_timesteps: int = 250
    preview_scale: int = 4  # 64 px is unwatchable at native size

    # Sized for a machine that can lose power without warning: at ~0.3 s/step a 1k interval
    # caps the loss from a hard crash at about five minutes.
    ckpt_every: int = 1_000
    ckpt_keep: int = 15
    ckpt_milestone_every: int = 50_000  # these are never pruned

    # Metrics tracked into `best-<metric>.pt`, lower being better for all of them. `fvd/val`
    # only moves on the metric_every grid but is the one worth trusting; the held-out loss
    # updates every eval_every and mostly serves as an early warning that something broke.
    best_metrics: tuple = ("fvd/val", "eval/val_eps_mse")

    # ---- bookkeeping ------------------------------------------------------
    run_name: str = ""
    wandb_project: str = "kabr-video-diffusion"
    wandb_mode: str = "online"
    data_root: str = ""
    out_root: str = ""
    resume: str = ""

    def __post_init__(self):
        if not self.data_root:
            self.data_root = os.environ.get("KABR_DATA_ROOT", "")
        if not self.out_root:
            self.out_root = os.environ.get("KABR_OUT_ROOT", "")
        if not self.data_root:
            raise SystemExit("set KABR_DATA_ROOT (or pass --data-root) to the KABR dataset root")
        if not self.out_root:
            raise SystemExit("set KABR_OUT_ROOT (or pass --out-root) to a writable output dir")
        if self.species not in SPECIES_PREFIX:
            raise SystemExit(f"species must be one of {sorted(SPECIES_PREFIX)}")
        if not self.run_name:
            self.run_name = (
                f"{self.species}-{self.image_size}px-{self.num_frames}f"
                f"-d{self.dim}-{self.objective}"
                f"{'-minsnr' + str(int(self.min_snr_gamma)) if self.min_snr_loss_weight else ''}"
            )

    # ---- derived paths ----------------------------------------------------
    @property
    def prefix(self) -> str:
        return SPECIES_PREFIX[self.species]

    @property
    def image_dir(self) -> Path:
        return Path(self.data_root) / "image"

    @property
    def cache_dir(self) -> Path:
        """Where the decoded uint8 memmap for this (species, resolution) lives."""
        return Path(self.out_root) / "cache" / f"{self.species}_{self.image_size}px"

    @property
    def run_dir(self) -> Path:
        return Path(self.out_root) / "runs" / self.run_name

    def to_dict(self) -> dict:
        return asdict(self)


def _add_arg(parser: argparse.ArgumentParser, f) -> None:
    name = "--" + f.name.replace("_", "-")
    if f.type is bool or isinstance(f.default, bool):
        parser.add_argument(name, dest=f.name, action=argparse.BooleanOptionalAction, default=None)
    elif isinstance(f.default, tuple):
        parser.add_argument(name, dest=f.name, type=str, default=None,
                            help="comma separated")
    else:
        parser.add_argument(name, dest=f.name, type=type(f.default), default=None)


def parse_config(argv=None) -> Config:
    parser = argparse.ArgumentParser(description="KABR video diffusion")
    for f in fields(Config):
        _add_arg(parser, f)
    args = parser.parse_args(argv)

    overrides = {}
    for f in fields(Config):
        val = getattr(args, f.name)
        if val is None:
            continue
        if isinstance(f.default, tuple) and isinstance(val, str):
            cast = type(f.default[0])
            val = tuple(cast(p) for p in val.split(","))
        overrides[f.name] = val
    return Config(**overrides)
