#!/usr/bin/env bash
# Wrapper that pins the environment this repo needs, then execs whatever it is given.
#
#   ./kabr/run.sh python -m kabr.prepare_data
#   ./kabr/run.sh python -m kabr.train --train-steps 300000
#
# Required in the environment (never hardcoded in the repo):
#   KABR_DATA_ROOT  dataset root, the directory holding image/
#   KABR_OUT_ROOT   writable output root for cache/, runs/
# Optional:
#   KABR_GPU        PCI bus index of the GPU to use (default 0)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${repo_root}"
exec "$@"
