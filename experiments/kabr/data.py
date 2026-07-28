"""Clip sampling on top of the prepared uint8 memmap.

A mini-scene is 90 frames at 29.97 fps. A training example is `num_frames` frames taken
with `frame_stride`, from a random start offset, so one mini-scene yields
`90 - stride * (num_frames - 1)` overlapping windows. Augmentation is a horizontal flip
applied to the whole clip - never per frame, which would destroy the motion.

Deterministic windows (centre of the mini-scene, fixed) are used for evaluation and for
the metric reference sets so those numbers do not move between checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ClipIndex:
    """The `index.npz` written by prepare_data, plus window arithmetic."""

    def __init__(self, cache_dir: Path, num_frames: int, frame_stride: int):
        idx = np.load(Path(cache_dir) / "index.npz")
        self.clip_ids = idx["clip_ids"]
        self.offsets = idx["offsets"]
        self.lengths = idx["lengths"]
        self.splits = idx["splits"]
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


class KabrClips(Dataset):
    """Random windows with random horizontal flip, for training."""

    def __init__(self, cache_dir, num_frames, frame_stride, split="train", horizontal_flip=True):
        self.cache_dir = Path(cache_dir)
        self.index = ClipIndex(self.cache_dir, num_frames, frame_stride)
        self.rows = self.index.usable(split)
        self.horizontal_flip = horizontal_flip
        self._frames = None
        if len(self.rows) == 0:
            raise SystemExit(f"no usable {split} mini-scenes for {num_frames}x stride {frame_stride}")

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
        return clip.permute(3, 0, 1, 2).float().div_(255.0)


def deterministic_clips(cache_dir, num_frames, frame_stride, split, limit=None, seed=0):
    """Fixed centre windows for a split, as one (n, c, f, h, w) float tensor in [0, 1].

    Used for the held-out loss and as the real-data reference for FVD/FID. `limit`
    subsamples with a fixed seed so the same clips are used at every checkpoint.
    """
    cache_dir = Path(cache_dir)
    index = ClipIndex(cache_dir, num_frames, frame_stride)
    rows = index.usable(split)
    if limit is not None and limit < len(rows):
        rows = np.random.default_rng(seed).choice(rows, size=limit, replace=False)
        rows = np.sort(rows)

    frames = np.load(cache_dir / "frames.u8", mmap_mode="r")
    out = np.empty((len(rows), num_frames, frames.shape[1], frames.shape[2], 3), dtype=np.uint8)
    for i, row in enumerate(rows):
        out[i] = frames[index.window(int(row), index.centre_start(int(row)))]
    t = torch.from_numpy(out).permute(0, 4, 1, 2, 3).float().div_(255.0)
    return t, index.clip_ids[rows]


def cycle(dl):
    while True:
        for batch in dl:
            yield batch
