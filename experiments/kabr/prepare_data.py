"""Decode the KABR mini-scene JPEGs into one uint8 memmap per (species, resolution).

KABR ships each mini-scene as a folder of 400x300 JPEGs, one per frame, in Charades
layout. Decoding those on the fly would put a JPEG decoder in the training loop for a
model that consumes 21 clips a second; instead this runs once and writes:

    frames.u8      (total_frames, size, size, 3) uint8
    index.npz      per mini-scene offset/length/split, plus the manifest

The whole giraffe set at 64px is under 2 GB, so after the first epoch the array is served
from page cache and the dataloader stops being a factor.

Spatial handling: 400x300 is centre cropped to 300x300 and resized to `size`. The mini-
scene is a tracking crop, so the animal already sits near the centre - cropping to square
keeps it and drops background at the left and right edges.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from kabr.config import SPECIES_PREFIX, Config, parse_config

ANNOTATION_REPO = "imageomics/KABR"


def load_split_map(prefix: str) -> dict[str, str]:
    """Map mini-scene id -> 'train' | 'val' using the official KABR annotation csvs.

    The official split is clean at the source-video level (a source video contributes
    mini-scenes to exactly one side), which is what makes held-out metrics meaningful.
    """
    from huggingface_hub import hf_hub_download

    out: dict[str, str] = {}
    for split in ("train", "val"):
        path = hf_hub_download(ANNOTATION_REPO, f"KABR/annotation/{split}.csv", repo_type="dataset")
        df = pd.read_csv(path, sep=" ", usecols=["original_vido_id"])
        ids = df.original_vido_id.astype(str).unique()
        for clip_id in ids:
            out[clip_id] = split
    return {k: v for k, v in out.items() if k.startswith(prefix) and not _is_other_species(k, prefix)}


def _is_other_species(clip_id: str, prefix: str) -> bool:
    """'G' must not swallow 'ZG'; prefixes are checked against the full alphabetic head."""
    head = "".join(c for c in clip_id if c.isalpha())
    return head != prefix


def _decode_clip(args) -> np.ndarray:
    clip_dir, size = args
    files = sorted(os.listdir(clip_dir), key=lambda n: int(n.split(".")[0]))
    frames = np.empty((len(files), size, size, 3), dtype=np.uint8)
    for i, name in enumerate(files):
        im = Image.open(os.path.join(clip_dir, name)).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        frames[i] = np.asarray(im.resize((size, size), Image.BICUBIC))
    return frames


def prepare(cfg: Config, workers: int = 8, force: bool = False) -> Path:
    cache = cfg.cache_dir
    frames_path = cache / "frames.u8"
    index_path = cache / "index.npz"
    if frames_path.exists() and index_path.exists() and not force:
        print(f"cache already built at {cache}, use --force to rebuild")
        return cache

    split_map = load_split_map(cfg.prefix)
    clip_ids = sorted(
        d for d in os.listdir(cfg.image_dir)
        if d in split_map and (cfg.image_dir / d).is_dir()
    )
    if not clip_ids:
        raise SystemExit(f"no {cfg.species} mini-scenes found under {cfg.image_dir}")

    lengths = [len(os.listdir(cfg.image_dir / c)) for c in tqdm(clip_ids, desc="counting frames")]
    total = int(sum(lengths))
    print(f"{len(clip_ids)} mini-scenes, {total} frames -> {total * cfg.image_size ** 2 * 3 / 2**30:.2f} GiB")

    cache.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        frames_path, mode="w+", dtype=np.uint8,
        shape=(total, cfg.image_size, cfg.image_size, 3),
    )

    offsets = np.zeros(len(clip_ids), dtype=np.int64)
    cursor = 0
    tasks = [(str(cfg.image_dir / c), cfg.image_size) for c in clip_ids]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, frames in enumerate(tqdm(pool.map(_decode_clip, tasks, chunksize=4),
                                        total=len(tasks), desc="decoding")):
            offsets[i] = cursor
            arr[cursor:cursor + len(frames)] = frames
            cursor += len(frames)
    arr.flush()
    assert cursor == total, (cursor, total)

    np.savez(
        index_path,
        clip_ids=np.array(clip_ids),
        offsets=offsets,
        lengths=np.array(lengths, dtype=np.int64),
        splits=np.array([split_map[c] for c in clip_ids]),
    )
    (cache / "manifest.json").write_text(json.dumps({
        "species": cfg.species,
        "image_size": cfg.image_size,
        "num_clips": len(clip_ids),
        "num_frames": total,
        "source_fps": 29.97,
        "crop": "centre square crop of 400x300, bicubic resize",
    }, indent=2))
    print(f"wrote {frames_path} and {index_path}")
    return cache


if __name__ == "__main__":
    known, rest = argparse.ArgumentParser(add_help=False), None
    known.add_argument("--workers", type=int, default=8)
    known.add_argument("--force", action="store_true")
    ns, rest = known.parse_known_args()
    prepare(parse_config(rest), workers=ns.workers, force=ns.force)
