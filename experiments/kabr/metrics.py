"""Generative metrics for short video clips.

Five families, all computed from the same set of generated clips:

  distribution  FVD / KVD on I3D features, FID / KID on individual frames.
                FVD alone cannot say whether a bad score comes from appearance or from
                motion; the frame-level pair pins that down, since it is blind to time.
  memorisation  nearest-neighbour distance from generated clips to the training set,
                calibrated against the same distance measured for real held-out clips.
  diversity     mean pairwise distance among generated clips, to catch mode collapse.
  motion        frame-to-frame absolute difference, to catch the standard failure mode of
                a small video model, which is emitting a still image 16 times.
  flow          optical flow magnitude and the residual left after warping one frame onto
                the next with the estimated flow.

Why the last family exists: frame differencing cannot tell displacement from flicker. A
model whose single frames get sharper produces more high frequency texture, and independent
texture between neighbouring frames raises the same pixel difference that real motion does.
Flow separates them - displacement lands in the magnitude, everything flow cannot explain
lands in the warp residual. Both are also reported for a centre window and its complement,
which is the closest cheap proxy available for foreground against background, since a KABR
mini-scene is a tracking crop and its background moves whenever the animal does.

On sample count: at a few hundred samples the Frechet estimators are badly biased. The
kernel versions (KID/KVD) are unbiased and are the ones to trust when comparing runs. Both
are reported. Neither is comparable to published numbers unless the whole protocol - sample
count, sampler, step count, reference split - matches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

I3D_REPO = "flateon/FVD-I3D-torchscript"
I3D_FILE = "i3d_torchscript.pt"
I3D_KWARGS = dict(rescale=False, resize=False, return_features=True)


# ---------------------------------------------------------------- feature backends
class I3DFeatures:
    """The Kinetics-400 I3D used by the reference FVD implementations.

    Expects clips as (b, c, f, h, w) float in [0, 1]; resizing to 224 and the shift to
    [-1, 1] happen here so callers cannot get the convention wrong.
    """

    def __init__(self, device="cuda"):
        from huggingface_hub import hf_hub_download

        # map_location, not a bare load followed by .to(): a TorchScript archive remembers
        # the device its tensors were saved on, and restoring onto an ordinal this machine
        # does not have fails inside the interpreter rather than at the call site.
        path = hf_hub_download(I3D_REPO, I3D_FILE)
        self.model = torch.jit.load(path, map_location=device).eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, clips: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
        out = []
        for i in range(0, len(clips), batch_size):
            x = clips[i:i + batch_size].to(self.device, non_blocking=True).float()
            b, c, f, h, w = x.shape
            if (h, w) != (224, 224):
                x = F.interpolate(
                    x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w),
                    size=(224, 224), mode="bilinear", align_corners=False,
                ).reshape(b, f, c, 224, 224).permute(0, 2, 1, 3, 4)
            x = x.mul(2).sub(1).contiguous()
            out.append(self.model(x, **I3D_KWARGS).float().cpu())
        return torch.cat(out)


# ---------------------------------------------------------------- distances
def frechet_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Frechet distance between two Gaussians fitted to the feature sets."""
    a, b = a.double().numpy(), b.double().numpy()
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def _poly_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d = x.shape[1]
    return (x @ y.T / d + 1.0) ** 3


def kernel_distance(a: torch.Tensor, b: torch.Tensor, subsets: int = 50,
                    subset_size: int | None = 64, seed: int = 0) -> tuple[float, float]:
    """Unbiased MMD^2 with the cubic polynomial kernel, averaged over random subsets.

    This is the KID estimator; applied to I3D features it is the video analogue, KVD.
    Returns (mean, std) over subsets.

    `subset_size` has to stay below the sample count for the std to mean anything. At
    `subset_size == n` every subset is the same full sample in a different order, the
    estimator is permutation invariant, and the reported spread is zero by construction -
    an error bar that says the measurement is exact when it is not.
    """
    a, b = a.double(), b.double()
    n = min(len(a), len(b))
    subset_size = min(subset_size or n, n)
    assert subset_size > 1, f"subset_size must be at least 2, got {subset_size}"
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(subsets):
        x = a[torch.randperm(len(a), generator=g)[:subset_size]]
        y = b[torch.randperm(len(b), generator=g)[:subset_size]]
        m = subset_size
        kxx, kyy, kxy = _poly_kernel(x, x), _poly_kernel(y, y), _poly_kernel(x, y)
        kxx = (kxx.sum() - kxx.diag().sum()) / (m * (m - 1))
        kyy = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
        vals.append(float(kxx + kyy - 2 * kxy.mean()))
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------- frame level
class FrameMetrics:
    """FID and KID over individual frames, via torchmetrics' InceptionV3."""

    def __init__(self, device="cuda", frames_per_clip: int = 4, kid_subset: int = 128):
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance

        self.device = device
        self.frames_per_clip = frames_per_clip
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.kid = KernelInceptionDistance(feature=2048, normalize=True,
                                           subset_size=kid_subset).to(device)

    def _frames(self, clips: torch.Tensor) -> torch.Tensor:
        """Even temporal subsample -> (n * frames_per_clip, 3, h, w) in [0, 1]."""
        f = clips.shape[2]
        idx = torch.linspace(0, f - 1, self.frames_per_clip).long()
        return clips[:, :, idx].permute(0, 2, 1, 3, 4).flatten(0, 1)

    @torch.no_grad()
    def __call__(self, real: torch.Tensor, fake: torch.Tensor, batch_size: int = 64) -> dict:
        self.fid.reset()
        self.kid.reset()
        for tag, clips in (("real", real), ("fake", fake)):
            frames = self._frames(clips)
            for i in range(0, len(frames), batch_size):
                x = frames[i:i + batch_size].to(self.device).clamp(0, 1)
                self.fid.update(x, real=(tag == "real"))
                self.kid.update(x, real=(tag == "real"))
        kid_mean, kid_std = self.kid.compute()
        return {
            "fid_frame": float(self.fid.compute()),
            "kid_frame": float(kid_mean),
            "kid_frame_std": float(kid_std),
        }


# ---------------------------------------------------------------- diagnostics
def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def nearest_neighbour_stats(query: torch.Tensor, reference: torch.Tensor) -> dict:
    """Cosine distance from every query feature to its closest reference feature."""
    q, r = _l2norm(query.double()), _l2norm(reference.double())
    sim = q @ r.T
    best, idx = sim.max(dim=1)
    dist = 1.0 - best
    return {
        "median": float(dist.median()),
        "mean": float(dist.mean()),
        "p10": float(dist.quantile(0.10)),
        "argmin": idx.cpu().numpy(),
        "dist": dist.cpu().numpy(),
    }


def diversity(features: torch.Tensor) -> float:
    """Mean pairwise cosine distance inside one feature set."""
    f = _l2norm(features.double())
    sim = f @ f.T
    n = len(f)
    off = (sim.sum() - sim.diag().sum()) / (n * (n - 1))
    return float(1.0 - off)


def centre_mask(h: int, w: int, centre_frac: float) -> torch.Tensor:
    """(h, w) bool, True inside the central `centre_frac` box."""
    mask = torch.zeros(h, w, dtype=torch.bool)
    ch, cw = int(round(h * centre_frac)), int(round(w * centre_frac))
    top, left = (h - ch) // 2, (w - cw) // 2
    mask[top:top + ch, left:left + cw] = True
    return mask


def motion_stats(clips: torch.Tensor, centre_frac: float = 0.5) -> dict:
    """Mean absolute difference between consecutive frames, per clip.

    Also split by region: the mini-scene tracks the animal, so the centre box is mostly
    animal and the border is mostly the ground sliding past. Whole-frame motion is dominated
    by the border, which is why it moves without telling you anything about the legs.
    """
    d = (clips[:, :, 1:] - clips[:, :, :-1]).abs()
    per_clip = d.flatten(1).mean(1)
    mask = centre_mask(clips.shape[-2], clips.shape[-1], centre_frac).to(d.device)
    centre = d[..., mask].mean()
    border = d[..., ~mask].mean()
    return {
        "mean": float(per_clip.mean()),
        "std": float(per_clip.std()),
        "centre": float(centre),
        "border": float(border),
    }


class FlowMetrics:
    """RAFT-small optical flow, and what the flow fails to explain.

    Two numbers per clip set:

      magnitude       mean |flow| over consecutive frame pairs, in pixels at `size`. This is
                      displacement, and unlike a pixel difference it does not grow when the
                      texture gets sharper.
      warp residual   mean |frame_t - warp(frame_{t+1}, flow)|. Whatever survives being
                      warped is not displacement: appearance change, occlusion, and the
                      per-frame flicker a video model produces when its frames stop agreeing.

    Clips are resized to a fixed `size` first, so a 64px and a 96px run land on the same
    scale and RAFT sees something closer to the resolution it was trained at. That makes the
    numbers comparable across resolutions but not comparable to flow measured natively.
    """

    def __init__(self, device="cuda", size: int = 128, centre_frac: float = 0.5):
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        assert size % 8 == 0, "RAFT downsamples by 8, so the side must be a multiple of 8"
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False)
        self.model = self.model.eval().to(device)
        self.device = device
        self.size = size
        self.centre_frac = centre_frac

    def _resize(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[-2:] == (self.size, self.size):
            return frames
        return F.interpolate(frames, size=(self.size, self.size), mode="bilinear",
                             align_corners=False)

    @staticmethod
    def _warp(frames: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Backward warp: sample `frames` at x + flow(x).

        RAFT's flow runs from the first image to the second, so the frame being sampled is
        the *second* one and the reconstruction it produces is the first. Feeding it the
        first frame instead reconstructs nothing and doubles the apparent displacement,
        which shows up as a residual larger than the plain frame difference.
        """
        b, _, h, w = frames.shape
        ys, xs = torch.meshgrid(torch.arange(h, device=frames.device, dtype=frames.dtype),
                                torch.arange(w, device=frames.device, dtype=frames.dtype),
                                indexing="ij")
        grid = torch.stack((xs, ys)).expand(b, -1, -1, -1) + flow
        grid = torch.stack((grid[:, 0] / max(w - 1, 1) * 2 - 1,
                            grid[:, 1] / max(h - 1, 1) * 2 - 1), dim=-1)
        return F.grid_sample(frames, grid, mode="bilinear", padding_mode="border",
                             align_corners=True)

    @torch.no_grad()
    def __call__(self, clips: torch.Tensor, batch_size: int = 32) -> dict:
        """clips: (n, c, f, h, w) in [0, 1] -> scalar flow statistics."""
        n, _, f = clips.shape[:3]
        assert f > 1, "flow needs at least two frames"
        mask = centre_mask(self.size, self.size, self.centre_frac).to(self.device)
        acc = {k: 0.0 for k in ("mag", "mag_centre", "mag_border", "residual",
                                "residual_centre", "baseline")}
        pairs = 0
        for i in range(0, n, max(1, batch_size // (f - 1))):
            chunk = clips[i:i + max(1, batch_size // (f - 1))].to(self.device).float()
            frames = chunk.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (b*f, c, h, w)
            frames = self._resize(frames).clamp(0, 1)
            frames = frames.reshape(len(chunk), f, *frames.shape[1:])
            a = frames[:, :-1].flatten(0, 1)
            b = frames[:, 1:].flatten(0, 1)
            flow = self.model(a.mul(2).sub(1), b.mul(2).sub(1))[-1]
            mag = flow.square().sum(1).sqrt()
            residual = (a - self._warp(b, flow)).abs().mean(1)
            k = len(a)
            acc["mag"] += float(mag.mean()) * k
            acc["mag_centre"] += float(mag[:, mask].mean()) * k
            acc["mag_border"] += float(mag[:, ~mask].mean()) * k
            acc["residual"] += float(residual.mean()) * k
            acc["residual_centre"] += float(residual[:, mask].mean()) * k
            acc["baseline"] += float((b - a).abs().mean()) * k
            pairs += k
        out = {k: v / max(pairs, 1) for k, v in acc.items()}
        # How much of the raw frame difference the flow field accounts for. Near 1 means the
        # difference is not displacement at all.
        out["residual_ratio"] = out["residual"] / max(out["baseline"], 1e-8)
        return out


# ---------------------------------------------------------------- top level
@dataclass
class MetricOptions:
    """Everything about the metric block that a run is allowed to change.

    Frozen per run and logged with the results: none of these numbers is comparable across
    different settings, so they belong with the results rather than in the call sites.
    """

    batch_size: int = 16
    kvd_subsets: int = 50
    kvd_subset_size: int = 64
    flow_clips: int = 64  # 0 disables the flow block
    centre_frac: float = 0.5


def evaluate(fake: torch.Tensor, real_train: torch.Tensor, real_val: torch.Tensor,
             i3d: I3DFeatures, frame_metrics: FrameMetrics | None = None,
             opts: MetricOptions | None = None, flow: FlowMetrics | None = None,
             f_train: torch.Tensor | None = None,
             f_val: torch.Tensor | None = None) -> tuple[dict, dict]:
    """Compute the full metric block.

    `f_train` / `f_val` accept I3D features already computed for the reference sets. They
    never change during a run, so recomputing them at every checkpoint buys nothing.

    Returns (scalars, extras) where extras carries the nearest-neighbour indices needed to
    render memorisation pairs, and the generated features for any caller that wants them.
    """
    opts = opts or MetricOptions()
    kvd = dict(subsets=opts.kvd_subsets, subset_size=opts.kvd_subset_size)

    f_fake = i3d(fake, opts.batch_size)
    f_train = i3d(real_train, opts.batch_size) if f_train is None else f_train
    f_val = i3d(real_val, opts.batch_size) if f_val is None else f_val

    kvd_train, kvd_train_std = kernel_distance(f_fake, f_train, **kvd)
    kvd_val, kvd_val_std = kernel_distance(f_fake, f_val, **kvd)

    nn_gen = nearest_neighbour_stats(f_fake, f_train)
    nn_val = nearest_neighbour_stats(f_val, f_train)

    m_gen = motion_stats(fake, opts.centre_frac)
    m_train = motion_stats(real_train, opts.centre_frac)
    m_val = motion_stats(real_val, opts.centre_frac)

    out = {
        "fvd/train": frechet_distance(f_fake, f_train),
        "fvd/val": frechet_distance(f_fake, f_val),
        "kvd/train": kvd_train,
        "kvd/train_std": kvd_train_std,
        "kvd/val": kvd_val,
        "kvd/val_std": kvd_val_std,
        "nn/gen_median": nn_gen["median"],
        "nn/gen_p10": nn_gen["p10"],
        "nn/val_median": nn_val["median"],
        # 1.0 means generated clips sit as far from the training set as real held-out
        # clips do; well below 1.0 means the model is reproducing training clips.
        "nn/novelty_ratio": nn_gen["median"] / max(nn_val["median"], 1e-8),
        "diversity/gen": diversity(f_fake),
        "diversity/train": diversity(f_train),
        "diversity/ratio": diversity(f_fake) / max(diversity(f_train), 1e-8),
        "motion/gen_mean": m_gen["mean"],
        "motion/gen_std": m_gen["std"],
        "motion/gen_centre": m_gen["centre"],
        "motion/gen_border": m_gen["border"],
        "motion/train_mean": m_train["mean"],
        "motion/train_centre": m_train["centre"],
        "motion/train_border": m_train["border"],
        "motion/val_mean": m_val["mean"],
    }
    out["motion/ratio"] = out["motion/gen_mean"] / max(out["motion/train_mean"], 1e-8)
    out["motion/centre_ratio"] = m_gen["centre"] / max(m_train["centre"], 1e-8)
    out["motion/border_ratio"] = m_gen["border"] / max(m_train["border"], 1e-8)

    if flow is not None and opts.flow_clips > 0:
        n = opts.flow_clips
        fl_gen, fl_train = flow(fake[:n]), flow(real_train[:n])
        for key, value in fl_gen.items():
            out[f"flow/gen_{key}"] = value
        for key, value in fl_train.items():
            out[f"flow/train_{key}"] = value
        out["flow/mag_ratio"] = fl_gen["mag"] / max(fl_train["mag"], 1e-8)
        out["flow/mag_centre_ratio"] = fl_gen["mag_centre"] / max(fl_train["mag_centre"], 1e-8)
        # Above 1 means more of the frame-to-frame change is unexplained by displacement
        # than in real data, which is the signature of flicker rather than motion.
        out["flow/residual_ratio_vs_train"] = (fl_gen["residual_ratio"]
                                               / max(fl_train["residual_ratio"], 1e-8))

    if frame_metrics is not None:
        out.update(frame_metrics(real_train, fake))

    extras = {"nn_argmin": nn_gen["argmin"], "nn_dist": nn_gen["dist"], "f_fake": f_fake}
    return out, extras
