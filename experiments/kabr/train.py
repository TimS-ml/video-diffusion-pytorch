"""Training loop for KABR video diffusion, unconditional or behaviour conditioned.

Three logging tiers, sized so evaluation stays a small fraction of training time:

  every step        loss, lr, grad norm, step time
  `eval_every`      deterministic held-out loss on a frozen set of (clip, t, noise)
                    triples, reported in epsilon and x0 space so the curve stays
                    comparable across objectives and loss weightings
  `metric_every`    FVD / KVD / FID / KID, nearest-neighbour memorisation, diversity,
                    motion and optical-flow statistics, from `metric_samples` DDIM samples,
                    plus a per-behaviour block when the run is conditioned

Sampling uses the EMA weights, never the live ones, and reuses one frozen set of latents at
every checkpoint so two checkpoints differ by their weights and nothing else.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F
from ema_pytorch import EMA
from torch.utils.data import DataLoader

import wandb
from kabr import metrics as M
from kabr import wandb_sync
from kabr.config import Config, parse_config
from kabr.data import (KabrClips, class_rows, cycle, deterministic_clips, make_cond_spec,
                       window_conds)
from video_diffusion_pytorch import GaussianDiffusion, Unet3D

# einops rearranges change strides between shapes and blow past the default recompile
# budget; the graphs themselves are small so a higher ceiling is cheap. The knob was called
# cache_size_limit before torch 2.7, and a rented or hosted runtime is not always on the
# version this was written against.
for _name in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _name):
        setattr(torch._dynamo.config, _name, 64)

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def lr_at(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.lr_schedule == "constant":
        return cfg.lr
    if cfg.lr_schedule != "cosine":
        raise SystemExit(f"unknown lr_schedule {cfg.lr_schedule!r}, expected cosine or constant")
    # Cosine from lr down to lr * lr_final_ratio across the post-warmup steps. Past the end
    # of the schedule the floor holds, so overrunning train_steps cannot send the rate back
    # up the curve.
    span = max(1, cfg.train_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / span)
    floor = cfg.lr * cfg.lr_final_ratio
    return floor + (cfg.lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def write_gif(clip: torch.Tensor, path: Path, fps: int, scale: int = 1) -> Path:
    """(c, f, h, w) float in [0, 1] -> animated gif on disk.

    Encoding here rather than handing raw frames to wandb keeps moviepy out of the
    dependency set and leaves the gif on disk for inspection outside the dashboard.
    `scale` upsamples with nearest neighbour, which keeps the pixel grid honest rather
    than inventing detail the model did not produce.
    """
    from PIL import Image

    arr = (clip.clamp(0, 1) * 255).byte().permute(1, 2, 3, 0).cpu().numpy()
    frames = [Image.fromarray(a) for a in arr]
    if scale > 1:
        w, h = frames[0].size
        frames = [f.resize((w * scale, h * scale), Image.NEAREST) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=False)
    return path


def grid_video(clips: torch.Tensor, rows: int, pad: int = 2) -> torch.Tensor:
    """(n, c, f, h, w) -> one (c, f, H, W) tile grid."""
    from einops import rearrange

    n = rows * rows
    clips = F.pad(clips[:n], (pad, pad, pad, pad))
    return rearrange(clips, "(i j) c f h w -> c f (i h) (j w)", i=rows)


class FrozenEval:
    """A fixed set of (clip, t, noise) triples, so the held-out curve has no sampling noise.

    Conditions are the ones the real clips carry, and guidance is off here: this measures
    the denoiser, not the sampler, and a guidance scale would put the two on different
    scales for no gain.
    """

    def __init__(self, clips: torch.Tensor, num_timesteps: int, n_t: int, seed: int,
                 cond: torch.Tensor | None = None, device_type: str = "cuda",
                 amp: dict | None = None):
        self.device_type = device_type
        self.amp = amp or {"dtype": torch.bfloat16, "enabled": True}
        self.clips = clips
        self.cond = cond
        # stay away from both ends: recovering epsilon near t = 0 is ill conditioned
        frac = torch.linspace(0.1, 0.9, n_t)
        self.timesteps = (frac * (num_timesteps - 1)).long()
        g = torch.Generator().manual_seed(seed)
        self.noise = torch.randn(len(clips), *clips.shape[1:], generator=g)

    @torch.no_grad()
    def __call__(self, diffusion: GaussianDiffusion, device, batch_size: int = 4) -> dict:
        eps_se, x0_se, count = 0.0, 0.0, 0
        for t_val in self.timesteps:
            for i in range(0, len(self.clips), batch_size):
                x0 = self.clips[i:i + batch_size].to(device) * 2 - 1
                noise = self.noise[i:i + batch_size].to(device)
                cond = None if self.cond is None else self.cond[i:i + batch_size].to(device)
                t = torch.full((len(x0),), int(t_val), device=device, dtype=torch.long)
                xt = diffusion.q_sample(x0, t, noise)
                with torch.autocast(self.device_type, **self.amp):
                    pred_eps, pred_x0 = diffusion.model_predictions(xt, t, cond=cond)
                eps_se += F.mse_loss(pred_eps.float(), noise, reduction="sum").item()
                x0_se += F.mse_loss(pred_x0.float(), x0, reduction="sum").item()
                count += x0.numel()
        return {"eps_mse": eps_se / count, "x0_mse": x0_se / count}


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = cfg.resolve_device()
        self.device_type = torch.device(self.device).type
        if self.device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        amp_name, amp_on = cfg.resolve_amp()
        self.amp_name = amp_name
        self.amp = {"dtype": AMP_DTYPES[amp_name], "enabled": amp_on}
        # Only fp16 needs the scaler; at bf16 or fp32 it is constructed disabled and every
        # call below becomes a pass-through, so there is one code path rather than two.
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=(amp_name == "fp16"))
        self.fps = round(29.97 / cfg.frame_stride)

        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir = cfg.run_dir / "media"
        self.media_dir.mkdir(exist_ok=True)
        (cfg.run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))

        self.cond_spec = make_cond_spec(cfg)
        cond_dim = self.cond_spec.dim if self.cond_spec else None

        self.unet = Unet3D(
            dim=cfg.dim, dim_mults=tuple(cfg.dim_mults),
            attn_heads=cfg.attn_heads, attn_dim_head=cfg.attn_dim_head,
            cond_dim=cond_dim, use_bert_text_cond=False,
        ).to(self.device)

        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=cfg.image_size, num_frames=cfg.num_frames,
            channels=3, timesteps=cfg.timesteps, loss_type=cfg.loss_type,
            objective=cfg.objective,
            min_snr_loss_weight=cfg.min_snr_loss_weight, min_snr_gamma=cfg.min_snr_gamma,
            sampling_timesteps=cfg.sampling_timesteps, ddim_sampling_eta=cfg.ddim_sampling_eta,
            schedule_shift=cfg.schedule_shift,
        ).to(self.device)

        self.ema = EMA(
            self.unet, beta=cfg.ema_decay,
            update_every=cfg.ema_update_every, update_after_step=cfg.ema_update_after_step,
        ).to(self.device)

        # AdamW at weight_decay 0 is Adam, so the unconditional runs already on record stay
        # reproducible through this call.
        self.opt = torch.optim.AdamW(self.unet.parameters(), lr=cfg.lr,
                                     betas=tuple(cfg.adam_betas), weight_decay=cfg.weight_decay)

        if cfg.compile_model:
            self.diffusion.denoise_fn = torch.compile(self.unet)

        ds = KabrClips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride,
                       split="train", horizontal_flip=cfg.horizontal_flip,
                       cond_spec=self.cond_spec)
        self.dl = cycle(DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                   num_workers=cfg.num_workers, pin_memory=True,
                                   drop_last=True, persistent_workers=cfg.num_workers > 0))
        self.n_train_clips = len(ds)

        # The marginal over conditions, read from the index alone so it is available before
        # the first metric block loads anything onto the gpu.
        self.cond_pool = (window_conds(cfg.cache_dir, cfg.num_frames, cfg.frame_stride,
                                       "train", self.cond_spec)
                          if self.cond_spec is not None else None)

        self.eval_sets = {}
        for split in ("train", "val"):
            clips, _, cond = deterministic_clips(
                cfg.cache_dir, cfg.num_frames, cfg.frame_stride, split,
                limit=cfg.eval_clips, seed=cfg.seed, cond_spec=self.cond_spec)
            self.eval_sets[split] = FrozenEval(clips, cfg.timesteps, cfg.eval_timesteps,
                                               cfg.seed, cond=cond, device_type=self.device_type,
                                               amp=self.amp)

        self.i3d = None
        self.frame_metrics = None
        self.flow = None
        self.ref = {}
        self.class_ref: dict[int, torch.Tensor] = {}
        self.metric_opts = M.MetricOptions(
            batch_size=cfg.metric_batch, kvd_subsets=cfg.kvd_subsets,
            kvd_subset_size=cfg.kvd_subset_size, flow_clips=cfg.flow_clips,
            centre_frac=cfg.flow_centre_frac,
        )
        self.step = 0
        self.best: dict[str, float] = {}

        # A crash during a write leaves one of these behind; it is never the file anything
        # reads from, but there is no reason to keep half a gigabyte of it around.
        for stale in cfg.run_dir.glob("*.pt.tmp"):
            stale.unlink()

        self.params = sum(p.numel() for p in self.unet.parameters())
        cond_note = "unconditional" if not self.cond_spec else \
            f"cond dim {self.cond_spec.dim} ({', '.join(self.cond_spec.names())})"
        print(f"unet {self.params/1e6:.1f}M params | {self.n_train_clips} train mini-scenes "
              f"| {self.fps} fps | {cond_note} | {self.device} {self.amp_name} "
              f"| run dir {cfg.run_dir}")

    # ---------------------------------------------------------------- metrics setup
    def _lazy_metric_setup(self):
        if self.i3d is not None:
            return
        cfg = self.cfg
        self.i3d = M.I3DFeatures(self.device)
        self.frame_metrics = M.FrameMetrics(self.device, kid_subset=min(128, cfg.metric_samples))
        if cfg.flow_clips > 0:
            self.flow = M.FlowMetrics(self.device, centre_frac=cfg.flow_centre_frac)
        limit = cfg.metric_ref_samples or None
        for split in ("train", "val"):
            clips, ids, _ = deterministic_clips(
                cfg.cache_dir, cfg.num_frames, cfg.frame_stride, split,
                limit=limit, seed=cfg.metric_seed)
            self.ref[split] = clips
            self.ref[split + "_ids"] = ids
            # The reference sets never change, so their features are computed once.
            self.ref["f_" + split] = self.i3d(clips, cfg.metric_batch)
        print(f"  metric references: {len(self.ref['train'])} train / {len(self.ref['val'])} val "
              f"clips, I3D features cached")

        if self.cond_spec is not None and cfg.metric_class_samples > 0:
            rows = class_rows(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "train",
                              pure_frac=cfg.cond_pure_frac)
            for k, r in rows.items():
                # A class needs at least a KVD subset worth of real clips before a distance
                # against it says anything.
                if len(r) < cfg.kvd_subset_size:
                    continue
                clips, _, _ = deterministic_clips(cfg.cache_dir, cfg.num_frames,
                                                  cfg.frame_stride, "train", rows=r,
                                                  limit=limit, seed=cfg.metric_seed)
                self.class_ref[k] = self.i3d(clips, cfg.metric_batch)
            names = [self.cond_spec.behaviours[k] for k in self.class_ref]
            print(f"  per-class references: {dict(zip(names, (len(rows[k]) for k in self.class_ref)))}")

    def _sample_conds(self, n: int, seed: int) -> torch.Tensor | None:
        """Conditions for the metric block, drawn from the real training marginal.

        Sampling every class uniformly would compare a flat generated distribution against a
        reference that is 46% Head Up, and the distance would be measuring the mismatch we
        introduced rather than the model.
        """
        if self.cond_spec is None:
            return None
        g = torch.Generator().manual_seed(seed)
        return self.cond_pool[torch.randint(len(self.cond_pool), (n,), generator=g)]

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, n: int, steps: int | None = None, cond: torch.Tensor | None = None,
               cond_scale: float | None = None) -> torch.Tensor:
        """DDIM samples from the EMA weights, on cpu, in [0, 1]."""
        live = self.diffusion.denoise_fn
        live_steps = self.diffusion.sampling_timesteps
        self.diffusion.denoise_fn = self.ema.ema_model
        if steps is not None:
            self.diffusion.sampling_timesteps = steps
        scale = self.cfg.cond_scale if cond_scale is None else cond_scale
        try:
            out = []
            done = 0
            while done < n:
                b = min(self.cfg.metric_batch, n - done)
                chunk = None if cond is None else cond[done:done + b].to(self.device)
                with torch.autocast(self.device_type, **self.amp):
                    out.append(self.diffusion.sample(batch_size=b, cond=chunk,
                                                     cond_scale=scale).float().cpu())
                done += b
            return torch.cat(out)
        finally:
            self.diffusion.denoise_fn = live
            self.diffusion.sampling_timesteps = live_steps

    # ---------------------------------------------------------------- checkpoints
    def _blob(self) -> dict:
        return {
            "step": self.step,
            "unet": self.unet.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.cfg.to_dict(),
            "best": self.best,
        }

    @staticmethod
    def _atomic_save(blob: dict, path):
        """Write via a temporary file so losing power mid-write cannot leave a corrupt
        checkpoint sitting where the newest good one should be."""
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            torch.save(blob, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def save(self, tag: str):
        path = self.cfg.run_dir / f"ckpt-{tag}.pt"
        self._atomic_save(self._blob(), path)
        self._prune_checkpoints()
        return path

    def save_best(self, log: dict) -> dict:
        """Promote the current weights to `best-<metric>.pt` for every tracked metric that
        just reached a new low. Returns the wandb entries recording those lows."""
        out = {}
        for name in self.cfg.best_metrics:
            value = log.get(name)
            if value is None or not math.isfinite(value):
                continue
            if value >= self.best.get(name, math.inf):
                continue
            self.best[name] = value
            slug = name.replace("/", "_")
            blob = self._blob()
            blob["best_metric"] = {"name": name, "value": value, "step": self.step}
            self._atomic_save(blob, self.cfg.run_dir / f"best-{slug}.pt")
            out[f"best/{slug}"] = value
            out[f"best/{slug}_step"] = self.step
            print(f"  new best {name} = {value:.4f} at step {self.step}", flush=True)
        return out

    def _prune_checkpoints(self):
        """Keep the last `ckpt_keep`, plus every `ckpt_milestone_every` forever.

        Milestones survive so a finished run can still be compared against its own earlier
        weights, which a pure rolling window would have deleted.
        """
        numbered = sorted(self.cfg.run_dir.glob("ckpt-[0-9]*.pt"),
                          key=lambda p: int(p.stem.split("-")[-1]))
        recent = set(numbered[-self.cfg.ckpt_keep:])
        for path in numbered:
            step = int(path.stem.split("-")[-1])
            if path in recent or step % self.cfg.ckpt_milestone_every == 0:
                continue
            path.unlink()

    def load(self, path):
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.unet.load_state_dict(blob["unet"])
        self.ema.load_state_dict(blob["ema"])
        self.opt.load_state_dict(blob["opt"])
        # Absent in checkpoints written before fp16 was reachable, and absent in any run
        # that used bf16, where the scaler carries no state worth restoring.
        if blob.get("scaler") and self.scaler.is_enabled():
            self.scaler.load_state_dict(blob["scaler"])
        self.step = blob["step"]
        # Carry the record over, otherwise the first eval after a resume always looks like a
        # new best and overwrites a genuinely better checkpoint.
        self.best = dict(blob.get("best") or {})
        print(f"resumed from {path} at step {self.step}")
        if self.best:
            record = ", ".join(f"{k}={v:.4f}" for k, v in self.best.items())
            print(f"  carrying best so far: {record}")

    # ---------------------------------------------------------------- eval blocks
    def run_eval(self) -> dict:
        self.diffusion.eval()
        live = self.diffusion.denoise_fn
        self.diffusion.denoise_fn = self.ema.ema_model
        try:
            out = {}
            for split, ev in self.eval_sets.items():
                res = ev(self.diffusion, self.device)
                out[f"eval/{split}_eps_mse"] = res["eps_mse"]
                out[f"eval/{split}_x0_mse"] = res["x0_mse"]
            out["eval/gap_eps_mse"] = out["eval/val_eps_mse"] - out["eval/train_eps_mse"]
            out["eval/gap_x0_mse"] = out["eval/val_x0_mse"] - out["eval/train_x0_mse"]
            return out
        finally:
            self.diffusion.denoise_fn = live
            self.diffusion.train()

    def run_metrics(self) -> dict:
        self._lazy_metric_setup()
        cfg = self.cfg
        self.diffusion.eval()
        # With metric_fixed_noise the latents are the same tensor at every checkpoint, so a
        # move in the metric is a move in the weights. It only holds while metric_batch is
        # unchanged: the batching decides how the draws are cut, so a mid-run change to it
        # breaks the pairing and the numbers before and after are independent samples again.
        torch.manual_seed(cfg.metric_seed if cfg.metric_fixed_noise
                          else cfg.metric_seed + self.step)
        t0 = time.time()
        fake = self.sample(cfg.metric_samples, cond=self._sample_conds(cfg.metric_samples,
                                                                      cfg.metric_seed))
        gen_s = time.time() - t0

        scalars, extras = M.evaluate(
            fake, self.ref["train"], self.ref["val"], self.i3d,
            self.frame_metrics, opts=self.metric_opts, flow=self.flow,
            f_train=self.ref["f_train"], f_val=self.ref["f_val"],
        )
        scalars["metric/gen_seconds"] = gen_s
        scalars.update(self.run_class_metrics())

        # the eight generated clips closest to a training clip, paired with that clip:
        # the direct visual read on memorisation
        order = np.argsort(extras["nn_dist"])[:8]
        pairs = []
        for rank, i in enumerate(order):
            j = int(extras["nn_argmin"][i])
            pair = torch.cat([fake[i], self.ref["train"][j]], dim=-1)  # side by side
            path = self.media_dir / f"nn-{self.step:07d}-{rank}.gif"
            write_gif(pair, path, self.fps)
            pairs.append(wandb.Video(
                str(path), format="gif",
                caption=f"generated | nearest train clip {self.ref['train_ids'][j]} "
                        f"(cos dist {extras['nn_dist'][i]:.4f})",
            ))
        scalars["memorisation/closest_pairs"] = pairs
        self.diffusion.train()
        return scalars

    def run_class_metrics(self) -> dict:
        """KVD of clips generated for one behaviour against real clips of that behaviour.

        This is the condition-consistency check. A conditional model can lower the overall
        distance while ignoring the condition entirely; a per-class distance cannot be
        satisfied that way, and it needs no behaviour classifier to compute.
        """
        cfg = self.cfg
        if not self.class_ref or cfg.metric_class_samples <= 0:
            return {}
        out = {}
        for k, ref_features in self.class_ref.items():
            name = self.cond_spec.behaviours[k]
            torch.manual_seed(cfg.metric_seed + k)
            cond = self.cond_spec.one_hot(k).repeat(cfg.metric_class_samples, 1)
            clips = self.sample(cfg.metric_class_samples, cond=cond)
            features = self.i3d(clips, cfg.metric_batch)
            kvd, _ = M.kernel_distance(features, ref_features, subsets=cfg.kvd_subsets,
                                       subset_size=min(cfg.kvd_subset_size, len(clips)))
            out[f"class/{name}_kvd"] = kvd
            path = self.media_dir / f"class-{self.step:07d}-{name.replace(' ', '_')}.gif"
            rows = max(1, int(math.isqrt(min(len(clips), 4))))
            write_gif(grid_video(clips, rows), path, self.fps, scale=self.cfg.preview_scale)
            out[f"class/{name}_samples"] = wandb.Video(str(path), format="gif",
                                                       caption=f"{name} @ cfg {cfg.cond_scale}")
        return out

    def log_samples(self) -> dict:
        n = self.cfg.sample_rows ** 2
        self.diffusion.eval()
        clips = self.sample(n, cond=self._sample_conds(n, self.cfg.metric_seed + 1))
        self.diffusion.train()
        path = self.media_dir / f"samples-{self.step:07d}.gif"
        write_gif(grid_video(clips, self.cfg.sample_rows), path, self.fps)
        return {"samples/grid": wandb.Video(str(path), format="gif")}

    def log_preview(self) -> dict:
        """A slower, upscaled inference for eyeballing, separate from the metric samples."""
        cfg = self.cfg
        n = cfg.preview_rows ** 2
        self.diffusion.eval()
        t0 = time.time()
        clips = self.sample(n, steps=cfg.preview_timesteps,
                            cond=self._sample_conds(n, cfg.metric_seed + 2))
        self.diffusion.train()

        grid = self.media_dir / f"preview-{self.step:07d}-grid.gif"
        write_gif(grid_video(clips, cfg.preview_rows), grid, self.fps, scale=cfg.preview_scale)
        out = {
            "preview/grid": wandb.Video(str(grid), format="gif",
                                        caption=f"step {self.step}, DDIM {cfg.preview_timesteps}"),
            "preview/seconds": time.time() - t0,
        }
        singles = []
        for i in range(n):
            path = self.media_dir / f"preview-{self.step:07d}-{i}.gif"
            write_gif(clips[i], path, self.fps, scale=cfg.preview_scale)
            singles.append(wandb.Video(str(path), format="gif"))
        out["preview/clips"] = singles
        return out

    # ---------------------------------------------------------------- loop
    def train(self):
        cfg = self.cfg
        # Reuse the run id across restarts so a resumed run keeps one continuous set of
        # curves; comparing checkpoints is the whole point of the metric block.
        id_file = cfg.run_dir / "wandb_id.txt"
        if id_file.exists():
            run_id = id_file.read_text().strip()
        else:
            run_id = wandb.util.generate_id()
            id_file.write_text(run_id)
        run = wandb.init(
            project=cfg.wandb_project, name=cfg.run_name, mode=cfg.wandb_mode,
            id=run_id, resume="allow",
            config={**cfg.to_dict(), "params": self.params, "git_sha": git_sha(),
                    "train_clips": self.n_train_clips, "fps": self.fps,
                    "resolved_device": self.device, "resolved_amp": self.amp_name},
            dir=str(cfg.run_dir),
        )
        # A resume restarts from the last checkpoint, which is up to ckpt_every steps
        # behind wherever the crash happened. wandb's own step counter only moves
        # forward, so logging those steps again with step= gets them dropped, and an eval
        # or metric block that lands in the gap never reaches the dashboard even though
        # its checkpoint is on disk. Plot against a step we control instead and let
        # wandb's internal counter run free.
        wandb.define_metric("train/global_step")
        wandb.define_metric("*", step_metric="train/global_step")
        self.diffusion.train()
        t_last = time.time()
        t_start = time.time()
        stop_reason = "train_steps"
        nonfinite = 0

        while self.step < cfg.train_steps:
            lr = lr_at(self.step, cfg)
            for group in self.opt.param_groups:
                group["lr"] = lr

            total = 0.0
            for _ in range(cfg.grad_accum):
                batch = next(self.dl)
                cond = None
                if self.cond_spec is not None:
                    batch, cond = batch
                    cond = cond.to(self.device, non_blocking=True)
                batch = batch.to(self.device, non_blocking=True)
                with torch.autocast(self.device_type, **self.amp):
                    loss = self.diffusion(batch, cond=cond,
                                          null_cond_prob=cfg.null_cond_prob)
                self.scaler.scale(loss / cfg.grad_accum).backward()
                total += loss.item() / cfg.grad_accum

            # Unscale before clipping, or the clip threshold is applied to gradients that
            # are still multiplied by the loss scale and the norm means nothing.
            self.scaler.unscale_(self.opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.unet.parameters(), cfg.max_grad_norm)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.ema.update()
            self.step += 1

            if math.isfinite(total):
                nonfinite = 0
            else:
                nonfinite += 1
                print(f"  non-finite loss at step {self.step} "
                      f"({nonfinite}/{cfg.nonfinite_patience})", flush=True)
                if nonfinite >= cfg.nonfinite_patience:
                    raise RuntimeError(
                        f"loss non-finite for {nonfinite} consecutive steps ending at "
                        f"{self.step}")

            log = {"train/global_step": self.step,
                   "train/loss": total, "train/lr": lr,
                   "train/grad_norm": float(grad_norm),
                   "train/nonfinite_streak": nonfinite,
                   "train/loss_scale": float(self.scaler.get_scale()) if self.scaler.is_enabled() else 1.0,
                   "train/step_seconds": time.time() - t_last,
                   "train/clips_seen": self.step * cfg.batch_size * cfg.grad_accum,
                   "train/epochs": self.step * cfg.batch_size * cfg.grad_accum / self.n_train_clips}
            t_last = time.time()

            if self.step % cfg.eval_every == 0:
                log.update(self.run_eval())
            if self.step % cfg.sample_every == 0:
                log.update(self.log_samples())
            if self.step % cfg.preview_every == 0:
                log.update(self.log_preview())
            if self.step % cfg.metric_every == 0:
                log.update(self.run_metrics())
            # Before save, not after: the checkpoint carries the record of the best metrics
            # seen so far, and a resume reloads it. Saving first writes a checkpoint that
            # has not been told about this step's own eval, so a crash shortly after leaves
            # the next run comparing against a stale record and calling a worse result a
            # new best.
            log.update(self.save_best(log))
            if self.step % cfg.ckpt_every == 0:
                path = self.save(str(self.step))
                if cfg.ckpt_artifact and cfg.ckpt_artifact_every \
                        and self.step % cfg.ckpt_artifact_every == 0:
                    wandb_sync.push_checkpoint(cfg, run, path, self.step,
                                               {"reason": "milestone"})

            wandb.log(log)
            if self.step % 100 == 0:
                print(f"step {self.step} loss {total:.4f} "
                      f"({log['train/step_seconds']*1000:.0f} ms)", flush=True)

            # Checked after the step is fully accounted for - checkpointed, evaluated and
            # logged - so stopping here never costs the work already done.
            if cfg.stop_at_step and self.step >= cfg.stop_at_step:
                stop_reason = "stop_at_step"
                break
            if cfg.stop_after_seconds and time.time() - t_start >= cfg.stop_after_seconds:
                stop_reason = "stop_after_seconds"
                break

        # "final" is reserved for a run that actually reached its horizon. A chunk that ran
        # out of wall clock saves under its step number instead, so the next chunk resumes
        # from it and nothing downstream mistakes a partial run for a finished one.
        done = self.step >= cfg.train_steps
        path = self.save("final" if done else str(self.step))
        # After the save and before finish: this artifact is the only thing a session that
        # is about to be reclaimed leaves behind.
        if cfg.ckpt_artifact:
            try:
                wandb_sync.push_checkpoint(cfg, run, path, self.step,
                                           {"reason": stop_reason, "done": done})
            except Exception as exc:
                print(f"artifact push failed: {type(exc).__name__}: {exc}", flush=True)
        wandb.finish()
        print(f"KABR_STATUS {json.dumps({'step': self.step, 'done': done, 'reason': stop_reason, 'train_steps': cfg.train_steps})}",
              flush=True)
        print("training complete" if done else f"chunk stopped ({stop_reason}) at step {self.step}")


def main():
    cfg = parse_config()
    # Before the Trainer, because "auto" restores wandb_id.txt and the best-*.pt record into
    # the run directory, and both have to be there before training rather than after.
    resume = wandb_sync.resolve_resume(cfg)
    trainer = Trainer(cfg)
    if resume:
        trainer.load(resume)
    trainer.train()


if __name__ == "__main__":
    main()
