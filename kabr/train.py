"""Training loop for unconditional KABR video diffusion.

Three logging tiers, sized so evaluation stays a small fraction of training time:

  every step        loss, lr, grad norm, step time
  `eval_every`      deterministic held-out loss on a frozen set of (clip, t, noise)
                    triples, reported in epsilon and x0 space so the curve stays
                    comparable across objectives and loss weightings
  `metric_every`    FVD / KVD / FID / KID, nearest-neighbour memorisation, diversity and
                    motion statistics, from `metric_samples` DDIM samples

Sampling uses the EMA weights, never the live ones.
"""

from __future__ import annotations

import json
import math
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
from kabr.config import Config, parse_config
from kabr.data import KabrClips, cycle, deterministic_clips
from video_diffusion_pytorch import GaussianDiffusion, Unet3D

# einops rearranges change strides between shapes and blow past the default recompile
# budget; the graphs themselves are small so a higher ceiling is cheap.
torch._dynamo.config.recompile_limit = 64


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def lr_at(step: int, cfg: Config) -> float:
    if step >= cfg.warmup_steps:
        return cfg.lr
    return cfg.lr * (step + 1) / cfg.warmup_steps


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
    """A fixed set of (clip, t, noise) triples, so the held-out curve has no sampling noise."""

    def __init__(self, clips: torch.Tensor, num_timesteps: int, n_t: int, seed: int):
        self.clips = clips
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
                t = torch.full((len(x0),), int(t_val), device=device, dtype=torch.long)
                xt = diffusion.q_sample(x0, t, noise)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_eps, pred_x0 = diffusion.model_predictions(xt, t)
                eps_se += F.mse_loss(pred_eps.float(), noise, reduction="sum").item()
                x0_se += F.mse_loss(pred_x0.float(), x0, reduction="sum").item()
                count += x0.numel()
        return {"eps_mse": eps_se / count, "x0_mse": x0_se / count}


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = "cuda"
        self.fps = round(29.97 / cfg.frame_stride)

        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir = cfg.run_dir / "media"
        self.media_dir.mkdir(exist_ok=True)
        (cfg.run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))

        self.unet = Unet3D(
            dim=cfg.dim, dim_mults=tuple(cfg.dim_mults),
            attn_heads=cfg.attn_heads, attn_dim_head=cfg.attn_dim_head,
            use_bert_text_cond=False,
        ).to(self.device)

        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=cfg.image_size, num_frames=cfg.num_frames,
            channels=3, timesteps=cfg.timesteps, loss_type=cfg.loss_type,
            objective=cfg.objective,
            min_snr_loss_weight=cfg.min_snr_loss_weight, min_snr_gamma=cfg.min_snr_gamma,
            sampling_timesteps=cfg.sampling_timesteps, ddim_sampling_eta=cfg.ddim_sampling_eta,
        ).to(self.device)

        self.ema = EMA(
            self.unet, beta=cfg.ema_decay,
            update_every=cfg.ema_update_every, update_after_step=cfg.ema_update_after_step,
        ).to(self.device)

        self.opt = torch.optim.Adam(self.unet.parameters(), lr=cfg.lr, betas=tuple(cfg.adam_betas))

        if cfg.compile_model:
            self.diffusion.denoise_fn = torch.compile(self.unet)

        ds = KabrClips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride,
                       split="train", horizontal_flip=cfg.horizontal_flip)
        self.dl = cycle(DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                   num_workers=cfg.num_workers, pin_memory=True,
                                   drop_last=True, persistent_workers=cfg.num_workers > 0))
        self.n_train_clips = len(ds)

        self.eval_sets = {}
        for split in ("train", "val"):
            clips, _ = deterministic_clips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride,
                                           split, limit=cfg.eval_clips, seed=cfg.seed)
            self.eval_sets[split] = FrozenEval(clips, cfg.timesteps, cfg.eval_timesteps, cfg.seed)

        self.i3d = None
        self.frame_metrics = None
        self.ref = {}
        self.step = 0

        self.params = sum(p.numel() for p in self.unet.parameters())
        print(f"unet {self.params/1e6:.1f}M params | {self.n_train_clips} train mini-scenes "
              f"| {self.fps} fps | run dir {cfg.run_dir}")

    # ---------------------------------------------------------------- metrics setup
    def _lazy_metric_setup(self):
        if self.i3d is not None:
            return
        cfg = self.cfg
        self.i3d = M.I3DFeatures(self.device)
        self.frame_metrics = M.FrameMetrics(self.device, kid_subset=min(128, cfg.metric_samples))
        for split in ("train", "val"):
            clips, ids = deterministic_clips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride,
                                             split, limit=cfg.metric_samples, seed=cfg.metric_seed)
            self.ref[split] = clips
            self.ref[split + "_ids"] = ids

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, n: int, steps: int | None = None) -> torch.Tensor:
        """DDIM samples from the EMA weights, on cpu, in [0, 1]."""
        live = self.diffusion.denoise_fn
        live_steps = self.diffusion.sampling_timesteps
        self.diffusion.denoise_fn = self.ema.ema_model
        if steps is not None:
            self.diffusion.sampling_timesteps = steps
        try:
            out = []
            done = 0
            while done < n:
                b = min(self.cfg.metric_batch, n - done)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(self.diffusion.sample(batch_size=b).float().cpu())
                done += b
            return torch.cat(out)
        finally:
            self.diffusion.denoise_fn = live
            self.diffusion.sampling_timesteps = live_steps

    # ---------------------------------------------------------------- checkpoints
    def save(self, tag: str):
        path = self.cfg.run_dir / f"ckpt-{tag}.pt"
        torch.save({
            "step": self.step,
            "unet": self.unet.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "config": self.cfg.to_dict(),
        }, path)
        self._prune_checkpoints()
        return path

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
        self.step = blob["step"]
        print(f"resumed from {path} at step {self.step}")

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
        torch.manual_seed(cfg.metric_seed + self.step)
        t0 = time.time()
        fake = self.sample(cfg.metric_samples)
        gen_s = time.time() - t0

        scalars, extras = M.evaluate(
            fake, self.ref["train"], self.ref["val"], self.i3d,
            self.frame_metrics, batch_size=cfg.metric_batch,
        )
        scalars["metric/gen_seconds"] = gen_s

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

    def log_samples(self) -> dict:
        n = self.cfg.sample_rows ** 2
        self.diffusion.eval()
        clips = self.sample(n)
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
        clips = self.sample(n, steps=cfg.preview_timesteps)
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
        wandb.init(
            project=cfg.wandb_project, name=cfg.run_name, mode=cfg.wandb_mode,
            id=run_id, resume="allow",
            config={**cfg.to_dict(), "params": self.params, "git_sha": git_sha(),
                    "train_clips": self.n_train_clips, "fps": self.fps},
            dir=str(cfg.run_dir),
        )
        self.diffusion.train()
        t_last = time.time()

        while self.step < cfg.train_steps:
            lr = lr_at(self.step, cfg)
            for group in self.opt.param_groups:
                group["lr"] = lr

            total = 0.0
            for _ in range(cfg.grad_accum):
                batch = next(self.dl).to(self.device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = self.diffusion(batch)
                (loss / cfg.grad_accum).backward()
                total += loss.item() / cfg.grad_accum

            grad_norm = torch.nn.utils.clip_grad_norm_(self.unet.parameters(), cfg.max_grad_norm)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.ema.update()
            self.step += 1

            if not math.isfinite(total):
                raise RuntimeError(f"loss went non-finite at step {self.step}")

            log = {"train/loss": total, "train/lr": lr,
                   "train/grad_norm": float(grad_norm),
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
            if self.step % cfg.ckpt_every == 0:
                self.save(str(self.step))

            wandb.log(log, step=self.step)
            if self.step % 100 == 0:
                print(f"step {self.step} loss {total:.4f} "
                      f"({log['train/step_seconds']*1000:.0f} ms)", flush=True)

        self.save("final")
        wandb.finish()
        print("training complete")


def main():
    cfg = parse_config()
    trainer = Trainer(cfg)
    if cfg.resume:
        trainer.load(cfg.resume)
    trainer.train()


if __name__ == "__main__":
    main()
