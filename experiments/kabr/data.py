"""Clip sampling on top of the prepared uint8 memmap.

A mini-scene is 90 frames at 29.97 fps. A training example is `num_frames` frames taken
with `frame_stride`, from a random start offset, so one mini-scene yields
`90 - stride * (num_frames - 1)` overlapping windows. Augmentation is a horizontal flip
applied to the whole clip - never per frame, which would destroy the motion, and never a
temporal flip, which would destroy the arrow of time the model is supposed to learn.

Deterministic windows (centre of the mini-scene, fixed) are used for evaluation and for
the metric reference sets so those numbers do not move between checkpoints.

Conditioning: `CondSpec` turns a window into the fixed-width float vector the U-Net's FiLM
projection consumes. The behaviour block is the histogram of the window's per-frame labels
plus an explicit "unlabelled" coordinate, so a window the annotation does not cover is
representable rather than silently indistinguishable from a zero histogram.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from kabr.config import BEHAVIOURS, CACHE_VERSION, UNLABELLED


class ClipIndex:
    """The `index.npz` written by prepare_data, plus window arithmetic."""

    def __init__(self, cache_dir: Path, num_frames: int, frame_stride: int):
        idx = np.load(Path(cache_dir) / "index.npz")
        self.clip_ids = idx["clip_ids"]
        self.offsets = idx["offsets"]
        self.lengths = idx["lengths"]
        self.splits = idx["splits"]
        self.version = int(idx["version"]) if "version" in idx else 1
        self.frame_labels = idx["frame_labels"] if "frame_labels" in idx else None
        self.species = idx["species"] if "species" in idx else None
        self.behaviours = tuple(idx["behaviours"]) if "behaviours" in idx else BEHAVIOURS
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.span = frame_stride * (num_frames - 1) + 1

    def usable(self, split: str) -> np.ndarray:
        """Row indices for `split` whose mini-scene is long enough for one window."""
        mask = (self.splits == split) & (self.lengths >= self.span)
        return np.nonzero(mask)[0]

    def window(self, row: int, start: int) -> np.ndarray:
        """Absolute frame indices into the memmap for one window."""
        base = self.offsets[row] + start
        return base + np.arange(self.num_frames) * self.frame_stride

    def max_start(self, row: int) -> int:
        return int(self.lengths[row]) - self.span

    def centre_start(self, row: int) -> int:
        return self.max_start(row) // 2

    def labels(self, row: int, start: int) -> np.ndarray:
        """Per-frame behaviour label for one window, `UNLABELLED` where the csv is silent."""
        if self.frame_labels is None:
            return np.full(self.num_frames, UNLABELLED, dtype=np.uint8)
        return self.frame_labels[self.window(row, start)]

    def require_labels(self) -> None:
        if self.frame_labels is None:
            raise SystemExit(
                f"this cache is version {self.version} and carries no behaviour labels; "
                f"rebuild it with `python -m kabr.prepare_data --force` "
                f"(current format is version {CACHE_VERSION})"
            )


@dataclass(frozen=True)
class CondSpec:
    """Layout of the conditioning vector, and the code that fills it.

    Both blocks are one-hot or histogram valued, so the U-Net's linear projection of this
    vector is an embedding table: no separate embedding module, nothing extra in the
    checkpoint, and a class the run never saw simply has an unused row.
    """

    behaviour: bool
    species: tuple[str, ...]  # empty when species is not conditioned on
    behaviours: tuple[str, ...] = BEHAVIOURS

    @property
    def behaviour_dim(self) -> int:
        # + 1 for the explicit "unlabelled" coordinate
        return len(self.behaviours) + 1 if self.behaviour else 0

    @property
    def dim(self) -> int:
        return self.behaviour_dim + len(self.species)

    def names(self) -> list[str]:
        out = [*self.behaviours, "Unlabelled"] if self.behaviour else []
        return [*out, *self.species]

    def vector(self, labels: np.ndarray, species_index: int = 0) -> np.ndarray:
        """(num_frames,) uint8 labels -> (dim,) float32."""
        out = np.zeros(self.dim, dtype=np.float32)
        if self.behaviour:
            n = len(self.behaviours)
            codes = np.where(labels == UNLABELLED, n, labels).astype(np.int64)
            hist = np.bincount(codes, minlength=n + 1).astype(np.float32)
            out[:n + 1] = hist / max(len(labels), 1)
        if self.species:
            out[self.behaviour_dim + species_index] = 1.0
        return out

    def one_hot(self, behaviour_index: int, species_index: int = 0) -> torch.Tensor:
        """The vector to sample with: one behaviour for every frame, nothing unlabelled."""
        out = np.zeros(self.dim, dtype=np.float32)
        if self.behaviour:
            out[behaviour_index] = 1.0
        if self.species:
            out[self.behaviour_dim + species_index] = 1.0
        return torch.from_numpy(out)


def make_cond_spec(cfg) -> CondSpec | None:
    """Build the spec a config asks for, or None for an unconditional run."""
    if not (cfg.cond_behaviour or cfg.cond_species):
        return None
    return CondSpec(behaviour=cfg.cond_behaviour,
                    species=tuple(sorted(cfg.species_list)) if cfg.cond_species else ())


class KabrClips(Dataset):
    """Random windows with random horizontal flip, for training.

    Yields a clip, or `(clip, cond)` when a `CondSpec` is supplied.
    """

    def __init__(self, cache_dir, num_frames, frame_stride, split="train", horizontal_flip=True,
                 cond_spec: CondSpec | None = None):
        self.cache_dir = Path(cache_dir)
        self.index = ClipIndex(self.cache_dir, num_frames, frame_stride)
        self.rows = self.index.usable(split)
        self.horizontal_flip = horizontal_flip
        self.cond_spec = cond_spec
        self._frames = None
        if len(self.rows) == 0:
            raise SystemExit(f"no usable {split} mini-scenes for {num_frames}x stride {frame_stride}")
        if cond_spec is not None:
            if cond_spec.behaviour:
                self.index.require_labels()
            self.species_index = _species_index(self.index, cond_spec)

    @property
    def frames(self) -> np.ndarray:
        # opened lazily so each dataloader worker gets its own mmap handle
        if self._frames is None:
            self._frames = np.load(self.cache_dir / "frames.u8", mmap_mode="r")
        return self._frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = int(self.rows[i])
        start = int(torch.randint(0, self.index.max_start(row) + 1, (1,)).item())
        clip = np.ascontiguousarray(self.frames[self.index.window(row, start)])
        if self.horizontal_flip and torch.rand(1).item() < 0.5:
            clip = clip[:, :, ::-1]
        clip = torch.from_numpy(np.ascontiguousarray(clip))
        # (f, h, w, c) uint8 -> (c, f, h, w) float in [0, 1]
        clip = clip.permute(3, 0, 1, 2).float().div_(255.0)
        if self.cond_spec is None:
            return clip
        cond = self.cond_spec.vector(self.index.labels(row, start), int(self.species_index[row]))
        return clip, torch.from_numpy(cond)


def _species_index(index: ClipIndex, cond_spec: CondSpec) -> np.ndarray:
    """Per mini-scene position inside `cond_spec.species`."""
    if not cond_spec.species:
        return np.zeros(len(index.clip_ids), dtype=np.int64)
    if index.species is None:
        raise SystemExit("this cache carries no species column; rebuild it with --force")
    lookup = {name: i for i, name in enumerate(cond_spec.species)}
    missing = sorted(set(index.species.tolist()) - set(lookup))
    if missing:
        raise SystemExit(f"cache holds species {missing} that the condition spec does not cover")
    return np.array([lookup[s] for s in index.species], dtype=np.int64)


def deterministic_clips(cache_dir, num_frames, frame_stride, split, limit=None, seed=0,
                        rows=None, cond_spec: CondSpec | None = None):
    """Fixed centre windows for a split, as one (n, c, f, h, w) float tensor in [0, 1].

    Used for the held-out loss and as the real-data reference for FVD/FID. `limit`
    subsamples with a fixed seed so the same clips are used at every checkpoint; `rows`
    selects an explicit subset instead, which is how the per-class references are built.

    Returns (clips, clip_ids, cond) with cond None unless a `CondSpec` is given.
    """
    cache_dir = Path(cache_dir)
    index = ClipIndex(cache_dir, num_frames, frame_stride)
    rows = index.usable(split) if rows is None else np.asarray(rows)
    if limit is not None and limit < len(rows):
        rows = np.random.default_rng(seed).choice(rows, size=limit, replace=False)
        rows = np.sort(rows)

    frames = np.load(cache_dir / "frames.u8", mmap_mode="r")
    out = np.empty((len(rows), num_frames, frames.shape[1], frames.shape[2], 3), dtype=np.uint8)
    for i, row in enumerate(rows):
        out[i] = frames[index.window(int(row), index.centre_start(int(row)))]
    t = torch.from_numpy(out).permute(0, 4, 1, 2, 3).float().div_(255.0)

    cond = None
    if cond_spec is not None:
        species_index = _species_index(index, cond_spec)
        cond = torch.from_numpy(np.stack([
            cond_spec.vector(index.labels(int(r), index.centre_start(int(r))),
                             int(species_index[r]))
            for r in rows
        ])) if len(rows) else torch.zeros(0, cond_spec.dim)
    return t, index.clip_ids[rows], cond


def window_conds(cache_dir, num_frames, frame_stride, split, cond_spec: CondSpec) -> torch.Tensor:
    """Centre-window condition vectors for a whole split, without touching the pixels.

    This is the empirical marginal over conditions. Sampling from it keeps a conditional
    model's generated set comparable to a real reference set, which a uniform draw over
    classes would not be.
    """
    index = ClipIndex(Path(cache_dir), num_frames, frame_stride)
    if cond_spec.behaviour:
        index.require_labels()
    species_index = _species_index(index, cond_spec)
    rows = index.usable(split)
    return torch.from_numpy(np.stack([
        cond_spec.vector(index.labels(int(r), index.centre_start(int(r))), int(species_index[r]))
        for r in rows
    ])) if len(rows) else torch.zeros(0, cond_spec.dim)


def class_rows(cache_dir, num_frames, frame_stride, split, pure_frac=0.8) -> dict[int, np.ndarray]:
    """Rows whose centre window is dominated by a single behaviour.

    A window qualifies for class k when at least `pure_frac` of its frames carry k. Mixed
    windows are dropped rather than assigned to their majority: they are not a clean
    reference for any one class, and the per-class metric is only meaningful against a
    reference that is actually that class.
    """
    index = ClipIndex(Path(cache_dir), num_frames, frame_stride)
    index.require_labels()
    out: dict[int, list[int]] = {}
    for row in index.usable(split):
        labels = index.labels(int(row), index.centre_start(int(row)))
        codes, counts = np.unique(labels, return_counts=True)
        top = int(np.argmax(counts))
        if codes[top] == UNLABELLED or counts[top] < pure_frac * len(labels):
            continue
        out.setdefault(int(codes[top]), []).append(int(row))
    return {k: np.array(v) for k, v in sorted(out.items())}


def cycle(dl):
    while True:
        for batch in dl:
            yield batch
