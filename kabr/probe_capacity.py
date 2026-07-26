"""Measure step time and peak VRAM for candidate model sizes, on synthetic clips.

    python -m kabr.probe_capacity --grid 64:64:4:2 128:96:4:1

Each grid entry is `image_size:dim:batch_size:grad_accum`. The point is to pick a config
that fills the card instead of guessing: the first run used a quarter of a 24 GB 4090 and
nobody noticed until the run was already 140k steps deep.

Synthetic input rather than the real cache, so a resolution can be costed before spending
an hour preparing frames for it. Memory and step time do not depend on what the pixels
contain, only on their shape.
"""

from __future__ import annotations

import argparse
import time

import torch

from video_diffusion_pytorch import GaussianDiffusion, Unet3D


def measure(image_size: int, dim: int, batch: int, accum: int, frames: int,
            steps: int, compile_model: bool, sample_batch: int = 0,
            sample_timesteps: int = 50) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    unet = Unet3D(dim=dim, dim_mults=(1, 2, 4, 8), attn_heads=8, attn_dim_head=32,
                  use_bert_text_cond=False).cuda()
    diffusion = GaussianDiffusion(unet, image_size=image_size, num_frames=frames,
                                  channels=3, timesteps=1000, loss_type="l2",
                                  objective="pred_v", min_snr_loss_weight=True,
                                  min_snr_gamma=5.0).cuda()
    if compile_model:
        diffusion.denoise_fn = torch.compile(unet)
    opt = torch.optim.Adam(unet.parameters(), lr=1e-4, betas=(0.9, 0.99))
    params = sum(p.numel() for p in unet.parameters())

    clip = torch.rand(batch, 3, frames, image_size, image_size, device="cuda")

    def one_step():
        for _ in range(accum):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = diffusion(clip)
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    # Warmup covers the compile and the allocator settling; timing it would measure neither.
    for _ in range(3):
        one_step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30

    # The metric block samples long after training has grown the allocator, and an OOM
    # there would surface 10k steps into a run and then loop on every restart. Measure it
    # against the pool training has already claimed, which is the situation it really runs in.
    sample_peak = 0.0
    sample_s = 0.0
    if sample_batch:
        diffusion.sampling_timesteps = sample_timesteps
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            diffusion.sample(batch_size=sample_batch)
        torch.cuda.synchronize()
        sample_s = time.time() - t0
        sample_peak = torch.cuda.max_memory_allocated() / 2**30

    del unet, diffusion, opt, clip
    torch.cuda.empty_cache()
    return {"params": params, "s_per_step": elapsed / steps,
            "peak_gib": peak, "reserved_gib": reserved,
            "clips_per_s": batch * accum * steps / elapsed,
            "sample_peak_gib": sample_peak, "sample_s": sample_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", nargs="+", required=True,
                    help="image_size:dim:batch:grad_accum entries")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--sample-batch", type=int, default=0,
                    help="also sample this many clips after training, to cost the metric block")
    ap.add_argument("--sample-timesteps", type=int, default=50)
    args = ap.parse_args()

    name = torch.cuda.get_device_name()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"{name}, {total:.1f} GiB total, {args.frames} frames\n")
    head = f"{'res':>4} {'dim':>4} {'bs':>3} {'ga':>3} {'params':>8} " \
           f"{'s/step':>7} {'peak GiB':>9} {'resv GiB':>9} {'clips/s':>8}"
    if args.sample_batch:
        head += f" {'smpl GiB':>9} {'smpl s':>7}"
    print(head)
    print("-" * len(head))

    for entry in args.grid:
        image_size, dim, batch, accum = (int(x) for x in entry.split(":"))
        try:
            r = measure(image_size, dim, batch, accum, args.frames, args.steps,
                        not args.no_compile, args.sample_batch, args.sample_timesteps)
        except torch.OutOfMemoryError:
            print(f"{image_size:>4} {dim:>4} {batch:>3} {accum:>3} {'':>8} {'OOM':>7}")
            torch.cuda.empty_cache()
            continue
        line = (f"{image_size:>4} {dim:>4} {batch:>3} {accum:>3} {r['params']/1e6:>7.1f}M "
                f"{r['s_per_step']:>7.3f} {r['peak_gib']:>9.2f} {r['reserved_gib']:>9.2f} "
                f"{r['clips_per_s']:>8.2f}")
        if args.sample_batch:
            line += f" {r['sample_peak_gib']:>9.2f} {r['sample_s']:>7.2f}"
        print(line)


if __name__ == "__main__":
    main()
