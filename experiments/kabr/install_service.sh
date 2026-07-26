#!/usr/bin/env bash
# Install a systemd user service that brings the supervisor back up after a reboot.
#
# Run it with the environment you would normally train in (conda env active, KABR_* set):
#
#   KABR_RUN_NAME=my-run ./experiments/kabr/install_service.sh --train-steps 300000
#
# The unit and its environment file are written under $HOME, so no machine specific path
# ever lands in the repository.
#
# The service exists for unattended recovery after a reboot, not for day to day use: it
# runs the supervisor directly and its output goes to the journal. Running the supervisor
# by hand in tmux still works and is nicer to watch. Only one of the two can ever be live,
# because the supervisor takes a lock on the run directory.
#
#   journalctl --user -u kabr-train -f
#
# Without lingering enabled the service starts at login rather than at boot. Enabling it
# needs root, which this script deliberately does not ask for:
#
#   sudo loginctl enable-linger "$USER"
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${KABR_DATA_ROOT:?set KABR_DATA_ROOT}"
: "${KABR_OUT_ROOT:?set KABR_OUT_ROOT}"
: "${KABR_RUN_NAME:?set KABR_RUN_NAME to the run directory to supervise}"

command -v python >/dev/null || { echo "no python on PATH - activate the environment first" >&2; exit 1; }

conf_dir="${HOME}/.config/kabr"
unit_dir="${HOME}/.config/systemd/user"
mkdir -p "${conf_dir}" "${unit_dir}"

{
  echo "KABR_DATA_ROOT=${KABR_DATA_ROOT}"
  echo "KABR_OUT_ROOT=${KABR_OUT_ROOT}"
  echo "KABR_RUN_NAME=${KABR_RUN_NAME}"
  echo "KABR_GPU=${KABR_GPU:-0}"
  echo "KABR_GPU_NAME=${KABR_GPU_NAME:-}"
  echo "KABR_GPU_WAIT=${KABR_GPU_WAIT:-86400}"
  # Set this to 0 to make a start a single attempt, which is what you want while you are
  # working out whether a failure is the hardware or the code.
  echo "KABR_MAX_RESTARTS=${KABR_MAX_RESTARTS:-100}"
  # Both of these are easy to set for a shell you launched by hand and then lose on the
  # next boot, which is exactly when unattended recovery has to work.
  echo "KABR_PCI_RESET=${KABR_PCI_RESET:-}"
  echo "KABR_POWER_LIMIT=${KABR_POWER_LIMIT:-}"
  # The supervisor execs `python`, so the environment it needs has to be on PATH already;
  # a login shell started by systemd will not have run conda activate.
  echo "PATH=$(dirname "$(command -v python)"):/usr/local/bin:/usr/bin:/bin"
  [[ -n "${CONDA_PREFIX:-}" ]] && echo "CONDA_PREFIX=${CONDA_PREFIX}"
} > "${conf_dir}/env"

cat > "${unit_dir}/kabr-train.service" <<UNIT
[Unit]
Description=KABR video diffusion training supervisor

[Service]
Type=simple
EnvironmentFile=${conf_dir}/env
WorkingDirectory=${repo_root}
ExecStart=${repo_root}/experiments/kabr/supervise.sh $*
# The supervisor handles its own retries; systemd restarting it would only fight the
# three-fast-failures guard that stops a real bug from looping forever.
Restart=no
TimeoutStopSec=120

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
systemctl --user enable kabr-train.service

echo "installed ${unit_dir}/kabr-train.service"
echo "  env      ${conf_dir}/env"
echo "  start    systemctl --user start kabr-train"
echo "  watch    journalctl --user -u kabr-train -f"
if ! loginctl show-user "${USER}" -p Linger 2>/dev/null | grep -q 'Linger=yes'; then
  echo "note: lingering is off, so this starts at login rather than at boot"
  echo "      sudo loginctl enable-linger ${USER}"
fi
if [[ -n "${KABR_PCI_RESET:-}${KABR_POWER_LIMIT:-}" ]] && ! sudo -n true 2>/dev/null; then
  echo "note: bus reset and the power cap both need a sudoers rule to work unattended"
  echo "      ./experiments/kabr/print_sudoers.sh | sudo tee /etc/sudoers.d/kabr-gpu"
fi
