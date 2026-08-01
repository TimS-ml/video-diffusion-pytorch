"""End-to-end smoke test: data -> model -> train steps -> eval -> metrics -> ckpt.

Runs a deliberately tiny configuration so every code path executes in a couple of minutes.

    ./utils/run.sh python -m kabr.smoke

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

from kabr.config import BEHAVIOURS, CACHE_VERSION, UNLABELLED, Config
from kabr.data import (CondSpec, KabrClips, class_rows, deterministic_clips, make_cond_spec,
                       window_conds)

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

    a, ids_a, _ = deterministic_clips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "val", limit=8, seed=0)
    b, ids_b, _ = deterministic_clips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "val", limit=8, seed=0)
    assert torch.equal(a, b) and list(ids_a) == list(ids_b)
    print(f"[ok] deterministic_clips reproducible, shape={tuple(a.shape)}")
    return ds


def check_conditioning(cfg):
    """The cache carries labels, and a window turns into a well formed condition vector."""
    idx = KabrClips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "train").index
    assert idx.version == CACHE_VERSION, (
        f"cache is version {idx.version}, rebuild with `python -m kabr.prepare_data --force`")
    idx.require_labels()
    labelled = float((idx.frame_labels != UNLABELLED).mean())
    print(f"[ok] cache v{idx.version}, {labelled*100:.1f}% of frames labelled, "
          f"{len(set(idx.species.tolist()))} species")

    spec = CondSpec(behaviour=True, species=())
    assert spec.dim == len(BEHAVIOURS) + 1
    rows = idx.usable("train")
    vec = spec.vector(idx.labels(int(rows[0]), 0))
    assert abs(vec.sum() - 1.0) < 1e-6, vec
    assert (vec >= 0).all()
    hot = spec.one_hot(0)
    assert hot.sum() == 1.0 and hot[0] == 1.0 and hot[-1] == 0.0
    print(f"[ok] cond vector dim={spec.dim} sums to 1, first window={np.round(vec, 3)}")

    pool = window_conds(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "train", spec)
    assert pool.shape == (len(rows), spec.dim), pool.shape
    marginal = pool.mean(0)
    top = int(marginal.argmax())
    print(f"[ok] condition marginal over {len(pool)} windows, "
          f"dominant={spec.names()[top]} at {marginal[top]:.3f}")

    per_class = class_rows(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "train",
                           pure_frac=cfg.cond_pure_frac)
    counts = {BEHAVIOURS[k]: len(v) for k, v in per_class.items()}
    assert counts, "no mini-scene has a behaviour covering cond_pure_frac of its window"
    # a pure window must really be pure
    k, rows_k = next(iter(per_class.items()))
    labels = KabrClips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "train").index
    frac = (labels.labels(int(rows_k[0]), labels.centre_start(int(rows_k[0]))) == k).mean()
    assert frac >= cfg.cond_pure_frac, frac
    print(f"[ok] pure windows at frac>={cfg.cond_pure_frac}: {counts}")

    ds = KabrClips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "train",
                   cond_spec=make_cond_spec(Config(**{**BASE, "cond_behaviour": True})))
    clip, cond = ds[0]
    assert clip.shape[0] == 3 and cond.shape == (spec.dim,), (clip.shape, cond.shape)
    print(f"[ok] conditional dataset yields (clip, cond) = ({tuple(clip.shape)}, {tuple(cond.shape)})")


def check_metrics(cfg, n=64):
    from kabr import metrics as M

    device = cfg.resolve_device()
    i3d = M.I3DFeatures(device)
    half1, _, _ = deterministic_clips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "train", limit=n, seed=1)
    half2, _, _ = deterministic_clips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "train", limit=n, seed=2)
    val, _, _ = deterministic_clips(cfg.cache_dir, cfg.num_frames, cfg.frame_stride, "val", limit=n, seed=1)
    noise = torch.rand_like(half1)
    subset = max(2, n // 2)

    f1, f2, fv, fn = (i3d(x) for x in (half1, half2, val, noise))
    d_self = M.frechet_distance(f1, f2)
    d_val = M.frechet_distance(f1, fv)
    d_noise = M.frechet_distance(f1, fn)
    print(f"[ok] FVD  real/real={d_self:8.1f}  real/val={d_val:8.1f}  real/noise={d_noise:8.1f}")
    assert d_self < d_val < d_noise, "FVD ordering is wrong, the feature convention is off"

    k_self, s_self = M.kernel_distance(f1, f2, subset_size=subset)
    k_noise, _ = M.kernel_distance(f1, fn, subset_size=subset)
    print(f"[ok] KVD  real/real={k_self:.5f}  real/noise={k_noise:.5f}  subset std={s_self:.5f}")
    assert k_self < k_noise
    # the whole point of a subset smaller than the sample: the spread stops being zero.
    # At subset_size == n the estimator is permutation invariant, so the only spread left is
    # summation order in floating point - orders of magnitude below the real one.
    _, s_full = M.kernel_distance(f1, f2, subset_size=len(f1))
    assert s_self > 0.0 and s_full < 1e-6 * s_self, (s_full, s_self)
    print(f"[ok] subset_size=n reports std {s_full:.1e} (degenerate), "
          f"subset_size={subset} reports {s_self:.5f}")

    nn_val = M.nearest_neighbour_stats(fv, f1)
    nn_self = M.nearest_neighbour_stats(f1, f1)
    print(f"[ok] NN   val->train median={nn_val['median']:.4f}  self->self median={nn_self['median']:.2e}")
    assert nn_self["median"] < 1e-6, "a clip must be its own nearest neighbour"
    assert nn_val["median"] > nn_self["median"]

    still = half1[:, :, :1].repeat(1, 1, cfg.num_frames, 1, 1)
    m_real, m_still = M.motion_stats(half1), M.motion_stats(still)
    print(f"[ok] motion real={m_real['mean']:.5f} (centre {m_real['centre']:.5f} / "
          f"border {m_real['border']:.5f})  frozen={m_still['mean']:.5f}")
    assert m_still["mean"] < 1e-6 < m_real["mean"]

    check_flow(cfg, half1, still)

    fm = M.FrameMetrics(device, kid_subset=min(32, n))
    res = fm(half1, noise)
    res_self = fm(half1, half2)
    print(f"[ok] FID  real/real={res_self['fid_frame']:8.2f}  real/noise={res['fid_frame']:8.2f}")
    print(f"[ok] KID  real/real={res_self['kid_frame']:.5f}  real/noise={res['kid_frame']:.5f}")
    assert res_self["fid_frame"] < res["fid_frame"]
    assert res_self["kid_frame"] < res["kid_frame"]

    opts = M.MetricOptions(batch_size=8, kvd_subset_size=subset, flow_clips=min(8, n))
    flow = M.FlowMetrics(device)
    scalars, extras = M.evaluate(half2, half1, val, i3d, fm, opts=opts, flow=flow, f_train=f1)
    assert "nn/novelty_ratio" in scalars and len(extras["nn_argmin"]) == len(half2)
    assert "flow/gen_mag" in scalars and "kvd/val_std" in scalars
    print(f"[ok] evaluate() returns {len(scalars)} scalars, novelty_ratio="
          f"{scalars['nn/novelty_ratio']:.3f} (real data vs itself, expect near 1)")


def check_flow(cfg, real, still):
    """Flow must see displacement where there is displacement, and nothing where there is none.

    The discriminating case is the flicker clip: a frozen frame with independent noise per
    frame. Frame differencing calls it motion, because the pixels change. Flow calls it what
    it is, because nothing moved.
    """
    from kabr import metrics as M

    flow = M.FlowMetrics(cfg.resolve_device())
    shift = torch.stack([torch.roll(still[:, :, 0], shifts=2 * i, dims=-1)
                         for i in range(cfg.num_frames)], dim=2)
    flicker = (still + torch.randn_like(still) * 0.05).clamp(0, 1)

    n = min(8, len(real))
    f_still, f_shift, f_flick, f_real = (flow(x[:n]) for x in (still, shift, flicker, real))
    print(f"[ok] flow mag   frozen={f_still['mag']:.4f}  translated={f_shift['mag']:.4f}  "
          f"flicker={f_flick['mag']:.4f}  real={f_real['mag']:.4f}")
    assert f_still["mag"] < 0.1 < f_shift["mag"], (f_still["mag"], f_shift["mag"])
    assert f_shift["mag"] > f_flick["mag"], "flicker must not read as displacement"

    print(f"[ok] flow resid ratio  translated={f_shift['residual_ratio']:.3f}  "
          f"flicker={f_flick['residual_ratio']:.3f}  real={f_real['residual_ratio']:.3f}")
    assert f_shift["residual_ratio"] < f_flick["residual_ratio"], (
        "warping must explain a translation better than it explains noise")


def check_train(cfg, label=""):
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
    print(f"[ok] {label}training loop ran {cfg.train_steps} steps, checkpoints: {sorted(names)}")

    previews = sorted(p.name for p in (cfg.run_dir / "media").glob("preview-*.gif"))
    assert previews, "no preview gif written"
    from PIL import Image
    im = Image.open(cfg.run_dir / "media" / previews[0])
    assert im.size[0] % cfg.preview_scale == 0 and im.size[0] > cfg.image_size
    print(f"[ok] {label}previews written: {previews}, first is {im.size} over {im.n_frames} frames")

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
    print(f"[ok] {label}best checkpoints: { {k: round(v, 4) for k, v in tr.best.items()} }")

    tr2 = Trainer(cfg)
    tr2.load(ck[-1])
    assert tr2.step == tr.step
    assert tr2.best == tr.best, "best record did not survive the resume"
    for (k1, v1), (k2, v2) in zip(tr.unet.state_dict().items(), tr2.unet.state_dict().items()):
        assert k1 == k2 and torch.equal(v1.cpu(), v2.cpu()), k1
    print(f"[ok] {label}checkpoint round-trips")
    return tr


def check_conditional_train(cfg):
    """The conditional path end to end, including the per-behaviour metric block."""
    tr = check_train(cfg, label="conditional ")
    assert tr.cond_spec is not None and tr.unet.has_cond
    assert tr.class_ref, "no behaviour had enough pure windows to act as a reference"
    names = [BEHAVIOURS[k] for k in tr.class_ref]
    print(f"[ok] conditional run carried per-class references for {names}")


BASE = dict(
    species="giraffe", image_size=64, num_frames=16, frame_stride=4,
    dim=32, dim_mults=(1, 2, 4), timesteps=100, sampling_timesteps=5,
    batch_size=2, grad_accum=1, warmup_steps=2, train_steps=6,
    num_workers=0, compile_model=False,
    eval_every=3, eval_clips=4, eval_timesteps=2,
    sample_every=6, sample_rows=2,
    preview_every=3, preview_rows=2, preview_timesteps=10, preview_scale=2,
    metric_every=6, metric_samples=16, metric_batch=8, metric_ref_samples=32,
    kvd_subset_size=8, flow_clips=8,
    ckpt_every=3, ckpt_keep=1, ckpt_milestone_every=6, wandb_mode="disabled",
    run_name="smoke",
)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--metric-clips", type=int, default=64,
                    help="clips per side in the metric checks; drop it on cpu")
    ap.add_argument("--skip-train", action="store_true",
                    help="data, conditioning and metrics only")
    ns = ap.parse_args()
    BASE["device"] = ns.device

    cfg = Config(**BASE)
    print(f"cache:  {cfg.cache_dir}\nrun:    {cfg.run_dir}\ndevice: {cfg.resolve_device()}\n")
    check_data(cfg)
    print()
    check_conditioning(Config(**BASE))
    print()
    check_metrics(Config(**BASE), n=ns.metric_clips)
    if ns.skip_train:
        print("\nsmoke test passed (training skipped)")
        raise SystemExit(0)
    print()
    check_train(cfg)
    print()
    check_conditional_train(Config(**{
        **BASE, "run_name": "smoke-cond", "cond_behaviour": True,
        "cond_scale": 2.0, "metric_class_samples": 4, "cond_pure_frac": 0.8,
        "kvd_subset_size": 8,
    }))
    print("\nsmoke test passed")
