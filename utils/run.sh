#!/usr/bin/env bash
# Wrapper that pins the environment this repo needs, then execs whatever it is given.
#
#   ./utils/run.sh python -m kabr.prepare_data
#   ./utils/run.sh python -m kabr.train --train-steps 300000
#
# Required in the environment (never hardcoded in the repo):
#   KABR_DATA_ROOT  dataset root, the directory holding image/
#   KABR_OUT_ROOT   writable output root for cache/, runs/
# Optional:
#   KABR_GPU        PCI bus index of the GPU to use (default 0)
set -euo pipefail

# Experiment packages live under experiments/, so `kabr` is importable from there while
# `video_diffusion_pytorch` is importable from the repository root.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exp_root="${repo_root}/experiments"

: "${KABR_DATA_ROOT:?set KABR_DATA_ROOT to the KABR dataset root}"
: "${KABR_OUT_ROOT:?set KABR_OUT_ROOT to a writable output directory}"

# The conda-provided libstdc++ has to win over the system one, or importing torchvision
# dies on a missing GLIBCXX symbol pulled in by optree.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# Address GPUs the way nvidia-smi does. The default ordering is by capability, which on a
# mixed-GPU box silently selects the wrong card.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${KABR_GPU:-0}"

# Activation memory here is fragmentation-prone; expandable segments buy real headroom.
# torch 2.10 reads PYTORCH_ALLOC_CONF and only warns about the older CUDA-specific name,
# so set both and stay correct on either side of that rename.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PYTHONPATH="${repo_root}:${exp_root}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${repo_root}"
exec "$@"
