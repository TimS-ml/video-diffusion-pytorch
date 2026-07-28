"""Generative metrics for short video clips.

Four families, all computed from the same set of generated clips:

  distribution  FVD / KVD on I3D features, FID / KID on individual frames.
                FVD alone cannot say whether a bad score comes from appearance or from
                motion; the frame-level pair pins that down, since it is blind to time.
  memorisation  nearest-neighbour distance from generated clips to the training set,
                calibrated against the same distance measured for real held-out clips.
  diversity     mean pairwise distance among generated clips, to catch mode collapse.
  motion        frame-to-frame absolute difference, to catch the standard failure mode of
                a small video model, which is emitting a still image 16 times.

On sample count: at a few hundred samples the Frechet estimators are badly biased. The
kernel versions (KID/KVD) are unbiased and are the ones to trust when comparing runs. Both
are reported. Neither is comparable to published numbers unless the whole protocol - sample
count, sampler, step count, reference split - matches.
"""

from __future__ import annotations

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

        self.model = torch.jit.load(hf_hub_download(I3D_REPO, I3D_FILE)).eval().to(device)
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
                    subset_size: int | None = None, seed: int = 0) -> tuple[float, float]:
    """Unbiased MMD^2 with the cubic polynomial kernel, averaged over random subsets.

    This is the KID estimator; applied to I3D features it is the video analogue, KVD.
    Returns (mean, std) over subsets.
    """
    a, b = a.double(), b.double()
    n = min(len(a), len(b))
    subset_size = min(subset_size or n, n)
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


def motion_stats(clips: torch.Tensor) -> dict:
    """Mean absolute difference between consecutive frames, per clip."""
    d = (clips[:, :, 1:] - clips[:, :, :-1]).abs()
    per_clip = d.flatten(1).mean(1)
    return {"mean": float(per_clip.mean()), "std": float(per_clip.std())}


# ---------------------------------------------------------------- top level
def evaluate(fake: torch.Tensor, real_train: torch.Tensor, real_val: torch.Tensor,
             i3d: I3DFeatures, frame_metrics: FrameMetrics | None = None,
             batch_size: int = 16) -> tuple[dict, dict]:
    """Compute the full metric block.

    Returns (scalars, extras) where extras carries the nearest-neighbour indices needed to
    render memorisation pairs.
    """
    f_fake = i3d(fake, batch_size)
    f_train = i3d(real_train, batch_size)
    f_val = i3d(real_val, batch_size)

    kvd_train, kvd_train_std = kernel_distance(f_fake, f_train)
    kvd_val, _ = kernel_distance(f_fake, f_val)

    nn_gen = nearest_neighbour_stats(f_fake, f_train)
    nn_val = nearest_neighbour_stats(f_val, f_train)

    out = {
        "fvd/train": frechet_distance(f_fake, f_train),
        "fvd/val": frechet_distance(f_fake, f_val),
        "kvd/train": kvd_train,
        "kvd/train_std": kvd_train_std,
        "kvd/val": kvd_val,
        "nn/gen_median": nn_gen["median"],
        "nn/gen_p10": nn_gen["p10"],
        "nn/val_median": nn_val["median"],
        # 1.0 means generated clips sit as far from the training set as real held-out
        # clips do; well below 1.0 means the model is reproducing training clips.
        "nn/novelty_ratio": nn_gen["median"] / max(nn_val["median"], 1e-8),
        "diversity/gen": diversity(f_fake),
        "diversity/train": diversity(f_train),
        "motion/gen_mean": motion_stats(fake)["mean"],
        "motion/gen_std": motion_stats(fake)["std"],
        "motion/train_mean": motion_stats(real_train)["mean"],
        "motion/val_mean": motion_stats(real_val)["mean"],
    }
    out["motion/ratio"] = out["motion/gen_mean"] / max(out["motion/train_mean"], 1e-8)

    if frame_metrics is not None:
        out.update(frame_metrics(real_train, fake))

    extras = {"nn_argmin": nn_gen["argmin"], "nn_dist": nn_gen["dist"]}
    return out, extras
