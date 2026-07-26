"""End-to-end smoke test: data -> model -> train steps -> eval -> metrics -> ckpt.

Runs a deliberately tiny configuration so every code path executes in a couple of minutes.

    ./experiments/kabr/run.sh python -m kabr.smoke

Deliberately not named test_*.py. It needs a prepared cache and a few GB of VRAM, so it is
a script to run by hand rather than something pytest should collect; under the old name
pytest collected zero tests from it and reported success, which read like a passing suite.
Unit tests that do belong to pytest live in test_diffusion_ext.py.
"""

import os
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from kabr.config import Config
from kabr.data import KabrClips, deterministic_clips

os.environ.setdefault("WANDB_MODE", "disabled")


def check_data(cfg):
    ds = KabrClips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "train")
    x = ds[0]
    assert x.shape == (3, cfg.num_frames, cfg.image_size, cfg.image_size), x.shape
    assert x.dtype == torch.float32 and 0.0 <= x.min() and x.max() <= 1.0
    val = KabrClips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "val")
    print(f"[ok] dataset train={len(ds)} val={len(val)} sample={tuple(x.shape)} "
          f"range=[{x.min():.3f},{x.max():.3f}]")

    idx = ds.index
    assert idx.span == cfg.frame_stride * (cfg.num_frames - 1) + 1
    tr_ids = set(idx.clip_ids[idx.usable("train")])
    va_ids = set(idx.clip_ids[idx.usable("val")])
    assert not (tr_ids & va_ids)
    tr_src = {i.split(".")[0] for i in tr_ids}
    va_src = {i.split(".")[0] for i in va_ids}
    assert not (tr_src & va_src), "train/val share a source video"
    print(f"[ok] split clean: {len(tr_src)} train source videos, {len(va_src)} val, no overlap")

    a, ids_a = deterministic_clips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "val", limit=8, seed=0)
    b, ids_b = deterministic_clips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "val", limit=8, seed=0)
    assert torch.equal(a, b) and list(ids_a) == list(ids_b)
    print(f"[ok] deterministic_clips reproducible, shape={tuple(a.shape)}")
    return ds


def check_metrics(cfg):
    from kabr import metrics as M

    i3d = M.I3DFeatures("cuda")
    half1, _ = deterministic_clips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "train", limit=64, seed=1)
    half2, _ = deterministic_clips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "train", limit=64, seed=2)
    val, _ = deterministic_clips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "val", limit=64, seed=1)
    noise = torch.rand_like(half1)

    f1, f2, fv, fn = (i3d(x) for x in (half1, half2, val, noise))
    d_self = M.frechet_distance(f1, f2)
    d_val = M.frechet_distance(f1, fv)
    d_noise = M.frechet_distance(f1, fn)
    print(f"[ok] FVD  real/real={d_self:8.1f}  real/val={d_val:8.1f}  real/noise={d_noise:8.1f}")
    assert d_self < d_val < d_noise, "FVD ordering is wrong, the feature convention is off"

    k_self, _ = M.kernel_distance(f1, f2, subset_size=32)
    k_noise, _ = M.kernel_distance(f1, fn, subset_size=32)
    print(f"[ok] KVD  real/real={k_self:.5f}  real/noise={k_noise:.5f}")
    assert k_self < k_noise

    nn_val = M.nearest_neighbour_stats(fv, f1)
    nn_self = M.nearest_neighbour_stats(f1, f1)
    print(f"[ok] NN   val->train median={nn_val['median']:.4f}  self->self median={nn_self['median']:.2e}")
    assert nn_self["median"] < 1e-6, "a clip must be its own nearest neighbour"
    assert nn_val["median"] > nn_self["median"]

    still = half1[:, :, :1].repeat(1, 1, cfg.num_frames, 1, 1)
    m_real, m_still = M.motion_stats(half1), M.motion_stats(still)
    print(f"[ok] motion real={m_real['mean']:.5f}  frozen={m_still['mean']:.5f}")
    assert m_still["mean"] < 1e-6 < m_real["mean"]

    fm = M.FrameMetrics("cuda", kid_subset=32)
    res = fm(half1, noise)
    res_self = fm(half1, half2)
    print(f"[ok] FID  real/real={res_self['fid_frame']:8.2f}  real/noise={res['fid_frame']:8.2f}")
    print(f"[ok] KID  real/real={res_self['kid_frame']:.5f}  real/noise={res['kid_frame']:.5f}")
    assert res_self["fid_frame"] < res["fid_frame"]
    assert res_self["kid_frame"] < res["kid_frame"]

    scalars, extras = M.evaluate(half2, half1, val, i3d, fm)
    assert "nn/novelty_ratio" in scalars and len(extras["nn_argmin"]) == len(half2)
    print(f"[ok] evaluate() returns {len(scalars)} scalars, novelty_ratio="
          f"{scalars['nn/novelty_ratio']:.3f} (real data vs itself, expect near 1)")


def check_train(cfg):
    from kabr.train import Trainer

    tr = Trainer(cfg)
    tr.train()
    ck = sorted(cfg.run_dir.glob("ckpt-*.pt"))
    assert ck, "no checkpoint written"
    names = {p.name for p in ck}
    # step 3 is neither one of the last `ckpt_keep` nor a milestone, so it must be gone;
    # step 6 is a milestone and must survive
    assert "ckpt-3.pt" not in names, names
    assert "ckpt-6.pt" in names, names
    print(f"[ok] training loop ran {cfg.train_steps} steps, checkpoints: {sorted(names)}")

    previews = sorted(p.name for p in (cfg.run_dir / "media").glob("preview-*.gif"))
    assert previews, "no preview gif written"
    from PIL import Image
    im = Image.open(cfg.run_dir / "media" / previews[0])
    assert im.size[0] % cfg.preview_scale == 0 and im.size[0] > cfg.image_size
    print(f"[ok] previews written: {previews}, first is {im.size} over {im.n_frames} frames")

    # every tracked metric was logged at least once, so every one of them must have a
    # checkpoint, and the recorded value must be the lowest the run actually saw
    assert tr.best, "no best metric was ever recorded"
    for name, value in tr.best.items():
        slug = name.replace("/", "_")
        path = cfg.run_dir / f"best-{slug}.pt"
        assert path.exists(), f"{name} has a record but no {path.name}"
        blob = torch.load(path, map_location="cpu", weights_only=False)
        assert blob["best_metric"]["name"] == name
        assert blob["best_metric"]["value"] == value
        assert blob["best_metric"]["step"] == blob["step"]
    assert not list(cfg.run_dir.glob("*.pt.tmp")), "atomic save left a temporary file behind"
    print(f"[ok] best checkpoints: { {k: round(v, 4) for k, v in tr.best.items()} }")

    tr2 = Trainer(cfg)
    tr2.load(ck[-1])
    assert tr2.step == tr.step
    assert tr2.best == tr.best, "best record did not survive the resume"
    for (k1, v1), (k2, v2) in zip(tr.unet.state_dict().items(), tr2.unet.state_dict().items()):
        assert k1 == k2 and torch.equal(v1.cpu(), v2.cpu()), k1
    print("[ok] checkpoint round-trips")


if __name__ == "__main__":
    base = dict(
        species="giraffe", image_size=64, num_frames=16, frame_stride=4,
        dim=32, dim_mults=(1, 2, 4), timesteps=100, sampling_timesteps=5,
        batch_size=2, grad_accum=1, warmup_steps=2, train_steps=6,
        num_workers=0, compile_model=False,
        eval_every=3, eval_clips=4, eval_timesteps=2,
        sample_every=6, sample_rows=2,
        preview_every=3, preview_rows=2, preview_timesteps=10, preview_scale=2,
        metric_every=6, metric_samples=16, metric_batch=8,
        ckpt_every=3, ckpt_keep=1, ckpt_milestone_every=6, wandb_mode="disabled",
        run_name="smoke",
    )
    cfg = Config(**base)
    print(f"cache: {cfg.cache_dir}\nrun:   {cfg.run_dir}\n")
    check_data(cfg)
    print()
    check_metrics(Config(**base))
    print()
    check_train(cfg)
    print("\nsmoke test passed")
