"""Decode the KABR mini-scene JPEGs into one uint8 memmap per (species set, resolution).

KABR ships each mini-scene as a folder of 400x300 JPEGs, one per frame, in Charades
layout. Decoding those on the fly would put a JPEG decoder in the training loop for a
model that consumes 21 clips a second; instead this runs once and writes:

    frames.u8      (total_frames, size, size, 3) uint8
    index.npz      per mini-scene offset/length/split/species, the per-frame behaviour
                   label array, and the manifest

The whole giraffe set at 64px is under 2 GB, so after the first epoch the array is served
from page cache and the dataloader stops being a factor.

Spatial handling: 400x300 is centre cropped to 300x300 and resized to `size`. The mini-
scene is a tracking crop, so the animal already sits near the centre - cropping to square
keeps it and drops background at the left and right edges.

Behaviour labels: the annotation csv carries one label per frame, keyed by the frame's
path. They are stored frame-aligned with `frames.u8` so a training window can be turned
into a conditioning vector without a second pass over the csv. A frame the csv does not
cover gets `UNLABELLED`, which the conditioning code treats as "no condition" rather than
as a ninth behaviour.
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

from kabr.config import BEHAVIOURS, CACHE_VERSION, SPECIES_PREFIX, UNLABELLED, Config, parse_config

ANNOTATION_REPO = "imageomics/KABR"


def _annotation_frames(prefixes: tuple[str, ...]) -> tuple[dict[str, str], dict[str, dict[int, int]]]:
    """Read the official csvs into (mini-scene -> split) and (mini-scene -> {frame: label}).

    The official split is clean at the source-video level (a source video contributes
    mini-scenes to exactly one side), which is what makes held-out metrics meaningful.
    """
    from huggingface_hub import hf_hub_download

    splits: dict[str, str] = {}
    labels: dict[str, dict[int, int]] = {}
    for split in ("train", "val"):
        path = hf_hub_download(ANNOTATION_REPO, f"KABR/annotation/{split}.csv", repo_type="dataset")
        df = pd.read_csv(path, sep=" ", usecols=["original_vido_id", "path", "labels"])
        df["original_vido_id"] = df.original_vido_id.astype(str)
        keep = df.original_vido_id.map(lambda c: _species_of(c, prefixes) is not None)
        df = df[keep]
        for clip_id, group in df.groupby("original_vido_id", sort=False):
            splits[clip_id] = split
            # `path` is '<clip_id>/<frame>.jpg'; the stem is what the decoder sorts on, so
            # keying on it survives any renumbering of the csv's own frame_id column.
            stems = group.path.astype(str).map(lambda p: int(Path(p).stem))
            labels[clip_id] = dict(zip(stems, group.labels.astype(int)))
    return splits, labels


def _species_of(clip_id: str, prefixes: tuple[str, ...]) -> str | None:
    """Which requested species a mini-scene id belongs to, or None.

    Matching is on the full alphabetic head so 'G' cannot swallow 'ZG'.
    """
    head = "".join(c for c in clip_id if c.isalpha())
    for species, prefix in SPECIES_PREFIX.items():
        if head == prefix and prefix in prefixes:
            return species
    return None


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


def _frame_labels(clip_dir: Path, per_frame: dict[int, int]) -> np.ndarray:
    """Label per decoded frame, in the same order `_decode_clip` writes them."""
    stems = sorted((int(n.split(".")[0]) for n in os.listdir(clip_dir)))
    return np.array([per_frame.get(s, UNLABELLED) for s in stems], dtype=np.uint8)


def prepare(cfg: Config, workers: int = 8, force: bool = False) -> Path:
    cache = cfg.cache_dir
    frames_path = cache / "frames.u8"
    index_path = cache / "index.npz"
    if frames_path.exists() and index_path.exists() and not force:
        print(f"cache already built at {cache}, use --force to rebuild")
        return cache

    prefixes = tuple(cfg.prefixes)
    split_map, label_map = _annotation_frames(prefixes)
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

    frame_labels = np.full(total, UNLABELLED, dtype=np.uint8)
    offsets = np.zeros(len(clip_ids), dtype=np.int64)
    cursor = 0
    tasks = [(str(cfg.image_dir / c), cfg.image_size) for c in clip_ids]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, frames in enumerate(tqdm(pool.map(_decode_clip, tasks, chunksize=4),
                                        total=len(tasks), desc="decoding")):
            offsets[i] = cursor
            arr[cursor:cursor + len(frames)] = frames
            labels = _frame_labels(cfg.image_dir / clip_ids[i], label_map.get(clip_ids[i], {}))
            assert len(labels) == len(frames), (clip_ids[i], len(labels), len(frames))
            frame_labels[cursor:cursor + len(frames)] = labels
            cursor += len(frames)
    arr.flush()
    assert cursor == total, (cursor, total)

    species = np.array([_species_of(c, prefixes) for c in clip_ids])
    covered = float((frame_labels != UNLABELLED).mean())
    np.savez(
        index_path,
        clip_ids=np.array(clip_ids),
        offsets=offsets,
        lengths=np.array(lengths, dtype=np.int64),
        splits=np.array([split_map[c] for c in clip_ids]),
        species=species,
        frame_labels=frame_labels,
        behaviours=np.array(BEHAVIOURS),
        version=np.array(CACHE_VERSION),
    )
    (cache / "manifest.json").write_text(json.dumps({
        "version": CACHE_VERSION,
        "species": sorted(set(species.tolist())),
        "image_size": cfg.image_size,
        "num_clips": len(clip_ids),
        "num_frames": total,
        "source_fps": 29.97,
        "crop": "centre square crop of 400x300, bicubic resize",
        "behaviours": list(BEHAVIOURS),
        "labelled_frame_fraction": round(covered, 4),
    }, indent=2))
    print(f"wrote {frames_path} and {index_path}")
    print(f"labelled frames: {covered * 100:.2f}%  species: {sorted(set(species.tolist()))}")
    return cache


if __name__ == "__main__":
    known, rest = argparse.ArgumentParser(add_help=False), None
    known.add_argument("--workers", type=int, default=8)
    known.add_argument("--force", action="store_true")
    ns, rest = known.parse_known_args()
    prepare(parse_config(rest), workers=ns.workers, force=ns.force)
