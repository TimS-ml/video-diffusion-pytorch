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

# The eight classes of KABR/annotation/classes.json, in their published order. Frozen here
# because the conditioning vector is a fixed-width histogram over this list and a checkpoint
# trained against one ordering cannot be read back with another.
BEHAVIOURS = ("Walk", "Graze", "Browse", "Head Up", "Auto-Groom", "Trot", "Run", "Occluded")
UNLABELLED = 255  # sentinel inside the uint8 per-frame label array
CACHE_VERSION = 2  # v1 caches carry no labels and no species column


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    # Comma separated for a joint cache, e.g. "giraffe,zebra_grevys". Quadruped anatomy is
    # shared, so a second species is training signal for the first one's legs; the cache
    # and the conditioning vector both handle a set, and a run stays single-species only
    # because that is all the local dataset holds.
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

    # ---- conditioning -----------------------------------------------------
    # The condition is a fixed-width float vector: the behaviour histogram of the window,
    # optionally followed by a species one-hot. Feeding a one-hot to the U-Net's FiLM
    # projection is exactly an embedding table lookup, so no extra module and no extra
    # checkpoint state is needed. A histogram rather than a majority label because a
    # 61-frame window can straddle a behaviour change and averaging that into one class
    # would teach the model the wrong marginal.
    cond_behaviour: bool = False
    cond_species: bool = False
    # Fraction of steps that see the learned null embedding instead of the real condition.
    # Without it there is no unconditional branch and guidance is undefined.
    null_cond_prob: float = 0.1
    cond_scale: float = 1.0  # classifier-free guidance scale used by every sampler below
    # A window counts as class k for the per-class metric references only if k covers at
    # least this fraction of its frames; mixed windows are not a clean reference for any
    # single class.
    cond_pure_frac: float = 0.8

    # ---- diffusion --------------------------------------------------------
    timesteps: int = 1000
    objective: str = "pred_v"
    loss_type: str = "l2"
    min_snr_loss_weight: bool = True
    min_snr_gamma: float = 5.0
    sampling_timesteps: int = 50  # DDIM
    ddim_sampling_eta: float = 0.0
    # SNR' = SNR * schedule_shift^2 (https://arxiv.org/abs/2301.10972). The cosine schedule
    # is calibrated at 64px; above that, spatial redundancy makes the same nominal t easier
    # and the high-noise end stops forcing the model to learn global structure. Set to
    # 64 / image_size to hold the effective noise level fixed, 1.0 to leave the schedule
    # where it was for the two runs already on record.
    schedule_shift: float = 1.0

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
    # AdamW with zero decay is Adam, so the default reproduces both runs on record while
    # leaving the knob reachable. The 96px run's train/val gap widened monotonically from
    # step 40k with nothing in the recipe opposing it.
    weight_decay: float = 0.0

    ema_decay: float = 0.999
    ema_update_every: int = 10
    ema_update_after_step: int = 1000

    # ---- runtime ----------------------------------------------------------
    compile_model: bool = True
    num_workers: int = 4
    # "auto" takes cuda when there is one. Naming cpu explicitly is what lets the whole
    # pipeline be exercised on a machine with no working gpu, which is the difference
    # between verifying a change and asserting that it looks right.
    device: str = "auto"

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
    metric_protocol: str = "v3"

    # Reference clips are real data, so their count no longer has to match the generated
    # count: their features are computed once and reused, and a larger reference tightens
    # every distance plus the nearest-neighbour search behind the memorisation check. 0
    # means every usable mini-scene in the split, which at 96px is a few GB of resident
    # float32 - 512 keeps that in hand while still doubling the old reference.
    metric_ref_samples: int = 512

    # Reuse one frozen set of latents at every checkpoint. The metric difference between two
    # checkpoints is then a paired comparison rather than a difference of two independent
    # draws, which is what made the 96px run's three fvd values indistinguishable.
    metric_fixed_noise: bool = True

    # KVD is an unbiased MMD^2 estimator, but its spread only means something if the subsets
    # actually differ. At subset_size == n every "subset" is the full sample permuted and the
    # reported std is zero by construction.
    kvd_subsets: int = 50
    kvd_subset_size: int = 64

    # Optical flow separates the two things frame differencing conflates: displacement that
    # a flow field explains (motion) and residual that it does not (flicker). Costed at
    # `flow_clips` clips per side rather than the full metric set because RAFT on every clip
    # is the same order of work as the sampling that produced them.
    flow_clips: int = 64
    flow_centre_frac: float = 0.5  # side of the centre window, as a fraction of the frame

    # Conditional runs only: clips generated per behaviour class, scored against real clips
    # of that class. This is the condition-consistency check, and it needs no classifier.
    # 0 disables it.
    metric_class_samples: int = 0

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

    # Metrics tracked into `best-<metric>.pt`, lower being better for all of them. `kvd/val`
    # leads because it is the only unbiased distribution estimator at this sample count;
    # `fvd/val` is still logged but no longer selects, after it ranked the 64px run below
    # the real train-vs-val reference. The held-out loss updates every eval_every and mostly
    # serves as an early warning that something broke.
    best_metrics: tuple = ("kvd/val", "eval/val_eps_mse")

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
        unknown = [s for s in self.species_list if s not in SPECIES_PREFIX]
        if unknown:
            raise SystemExit(f"unknown species {unknown}, expected any of {sorted(SPECIES_PREFIX)}")
        if self.cond_species and len(self.species_list) < 2:
            raise SystemExit("cond_species needs more than one species to condition on")
        if not self.run_name:
            cond = "".join(("-behav" if self.cond_behaviour else "",
                            "-spec" if self.cond_species else ""))
            shift = "" if self.schedule_shift == 1.0 else f"-shift{self.schedule_shift:g}"
            self.run_name = (
                f"{self.species_slug}-{self.image_size}px-{self.num_frames}f"
                f"-d{self.dim}-{self.objective}"
                f"{'-minsnr' + str(int(self.min_snr_gamma)) if self.min_snr_loss_weight else ''}"
                f"{shift}{cond}"
            )

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    # ---- derived paths ----------------------------------------------------
    @property
    def species_list(self) -> tuple[str, ...]:
        return tuple(s.strip() for s in self.species.split(",") if s.strip())

    @property
    def species_slug(self) -> str:
        """Cache and run name stem. A single species keeps its bare name, so the caches and
        runs built before joint training stay addressable."""
        names = self.species_list
        return names[0] if len(names) == 1 else "+".join(sorted(names))

    @property
    def prefixes(self) -> tuple[str, ...]:
        return tuple(SPECIES_PREFIX[s] for s in self.species_list)

    @property
    def image_dir(self) -> Path:
        return Path(self.data_root) / "image"

    @property
    def cache_dir(self) -> Path:
        """Where the decoded uint8 memmap for this (species set, resolution) lives."""
        return Path(self.out_root) / "cache" / f"{self.species_slug}_{self.image_size}px"

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
