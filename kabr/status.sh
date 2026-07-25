#!/usr/bin/env bash
# Non-blocking snapshot of a training run: latest step, newest media, checkpoints, GPU.
#
#   KABR_OUT_ROOT=... ./kabr/status.sh [run-name]
#
# Defaults to the only run under $KABR_OUT_ROOT/runs if there is exactly one.
set -euo pipefail

: "${KABR_OUT_ROOT:?set KABR_OUT_ROOT}"
runs_dir="${KABR_OUT_ROOT}/runs"

name="${1:-}"
if [[ -z "${name}" ]]; then
  mapfile -t found < <(ls -1 "${runs_dir}" 2>/dev/null)
  if [[ ${#found[@]} -ne 1 ]]; then
    echo "several runs under ${runs_dir}, pass one:" >&2
    printf '  %s\n' "${found[@]}" >&2
    exit 1
  fi
  name="${found[0]}"
fi
run="${runs_dir}/${name}"
[[ -d "${run}" ]] || { echo "no such run: ${run}" >&2; exit 1; }

echo "run       ${run}"
echo "media     ${run}/media"
[[ -f "${run}/wandb_id.txt" ]] && echo "wandb id  $(cat "${run}/wandb_id.txt")"

echo
echo "-- checkpoints (newest 5 of $(ls -1 "${run}"/ckpt-*.pt 2>/dev/null | wc -l)) --"
ls -1sht "${run}"/ckpt-*.pt 2>/dev/null | head -5 || echo "   none yet"

echo
echo "-- best --"
ls -1sh "${run}"/best-*.pt 2>/dev/null || echo "   none yet"

echo
echo "-- newest media --"
ls -1t "${run}/media" 2>/dev/null | head -6 || echo "   none yet"

echo
echo "-- latest samples / preview --"
ls -1t "${run}/media"/samples-*.gif 2>/dev/null | head -1 || true
ls -1t "${run}/media"/preview-*-grid.gif 2>/dev/null | head -1 || true

echo
echo "-- gpu --"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu \
           --format=csv,noheader 2>/dev/null || echo "   nvidia-smi unavailable"
