#!/usr/bin/env bash
# Keep a training run alive on hardware that drops off the bus mid-run.
#
#   ./kabr/supervise.sh --train-steps 300000
#
# Relaunches from the newest checkpoint whenever training exits non-zero, waiting for the
# GPU to come back first. Exits for good when training finishes cleanly, when the GPU stays
# missing, or when the run dies fast enough times in a row to look like a code bug rather
# than flaky hardware.
#
# Environment: everything run.sh needs, plus
#   KABR_RUN_NAME       run directory under $KABR_OUT_ROOT/runs to resume from
#   KABR_MAX_RESTARTS   give up after this many restarts (default 100)
#   KABR_GPU_WAIT       seconds to wait for the GPU to reappear (default 1800)
#   KABR_GPU_NAME       substring the GPU at KABR_GPU must match, guarding against a
#                       re-enumeration that shifts every index
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

: "${KABR_OUT_ROOT:?set KABR_OUT_ROOT}"
: "${KABR_RUN_NAME:?set KABR_RUN_NAME to the run directory to supervise}"
run_dir="${KABR_OUT_ROOT}/runs/${KABR_RUN_NAME}"
max_restarts="${KABR_MAX_RESTARTS:-100}"
gpu_wait="${KABR_GPU_WAIT:-1800}"
gpu_index="${KABR_GPU:-0}"
log="${repo_root}/logs/supervise_${KABR_RUN_NAME}.log"
mkdir -p "${repo_root}/logs" "${run_dir}"

say() { printf '[supervise %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "${log}"; }

# One supervisor per run. Two of them would fight over the GPU and, worse, interleave
# writes into the same checkpoint filenames.
exec 9>"${run_dir}/supervise.lock"
if ! flock -n 9; then
  say "another supervisor already holds ${run_dir}/supervise.lock, nothing to do"
  exit 0
fi

newest_ckpt() {
  # Highest step wins; ls -t would pick whichever file the filesystem touched last, which
  # after a crash is not necessarily the furthest along.
  ls -1 "${run_dir}"/ckpt-[0-9]*.pt 2>/dev/null \
    | sed -E 's/.*ckpt-([0-9]+)\.pt/\1 &/' | sort -n -k1,1 | tail -1 | cut -d' ' -f2-
}

gpu_present() {
  local name
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${gpu_index}" 2>/dev/null)" || return 1
  # On a mixed-GPU box a device that re-enumerates at a different bus address shifts every
  # index, and training would silently land on the wrong card.
  if [[ -n "${KABR_GPU_NAME:-}" && "${name}" != *"${KABR_GPU_NAME}"* ]]; then
    say "gpu ${gpu_index} is '${name}', expected something matching '${KABR_GPU_NAME}'"
    return 1
  fi
  return 0
}

wait_for_gpu() {
  local waited=0
  while ! gpu_present; do
    if (( waited >= gpu_wait )); then
      say "gpu ${gpu_index} still missing after ${gpu_wait}s, giving up"
      say "an external gpu in this state usually needs the cable replugged"
      return 1
    fi
    (( waited == 0 )) && say "gpu ${gpu_index} is gone, waiting for it to come back"
    sleep 30
    waited=$(( waited + 30 ))
  done
  (( waited > 0 )) && say "gpu ${gpu_index} is back after ${waited}s"
  return 0
}

restarts=0
fast_failures=0
while :; do
  wait_for_gpu || exit 1

  ckpt="$(newest_ckpt)"
  args=("$@")
  if [[ -n "${ckpt}" ]]; then
    say "starting from ${ckpt##*/}"
    args+=(--resume "${ckpt}")
  else
    say "no checkpoint yet, starting from scratch"
  fi

  started=${SECONDS}
  ./kabr/run.sh python -m kabr.train "${args[@]}"
  code=$?
  ran=$(( SECONDS - started ))

  if (( code == 0 )); then
    say "training finished cleanly after ${restarts} restart(s)"
    exit 0
  fi

  # A run that dies before it has trained anything is almost never the hardware.
  if (( ran < 120 )); then
    fast_failures=$(( fast_failures + 1 ))
    if (( fast_failures >= 3 )); then
      say "exit ${code} after only ${ran}s, three times running - this looks like a bug, stopping"
      exit 1
    fi
  else
    fast_failures=0
  fi

  restarts=$(( restarts + 1 ))
  if (( restarts > max_restarts )); then
    say "exit ${code}, hit the ${max_restarts} restart limit, stopping"
    exit 1
  fi
  say "exit ${code} after ${ran}s, restart ${restarts}/${max_restarts} in 60s"
  sleep 60
done
