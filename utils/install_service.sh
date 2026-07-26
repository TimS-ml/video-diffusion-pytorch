#!/usr/bin/env bash
# Install a systemd user service that runs the training supervisor on demand.
#
# Run it with the environment you would normally train in (conda env active, KABR_* set):
#
#   KABR_RUN_NAME=my-run ./utils/install_service.sh --train-steps 300000
#
# The unit and its environment file are written under $HOME, so no machine specific path
# ever lands in the repository.
#
# Training NEVER starts on its own. The unit deliberately has no [Install] section, so it
# cannot be enabled and systemd will not pull it in at login or at boot; `systemctl --user
# enable kabr-train` fails outright. Starting a long GPU job behind the user's back is the
# one thing this must not do. A run begins only when someone types:
#
#   systemctl --user start kabr-train
#   journalctl --user -u kabr-train -f
#
# The unit buys journal logging and survives the terminal closing. Running the supervisor
# by hand in tmux works too and is nicer to watch. Only one of the two can ever be live,
# because the supervisor takes a lock on the run directory.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
ExecStart=${repo_root}/utils/supervise.sh $*
# The supervisor handles its own retries; systemd restarting it would only fight the
# three-fast-failures guard that stops a real bug from looping forever.
Restart=no
TimeoutStopSec=120

# No [Install] section on purpose. Without one the unit cannot be enabled and systemd has
# no target pulling it in, so a reboot or a login never starts training by itself.
UNIT

systemctl --user daemon-reload
# Deliberately no `systemctl --user enable`. See the [Install] note above.

# An enable symlink from an older install would still auto-start this at login, and the
# unit no longer has the [Install] section that `disable` reads to clean up after itself.
legacy_link="${unit_dir}/default.target.wants/kabr-train.service"
if [[ -e "${legacy_link}" || -L "${legacy_link}" ]]; then
  rm -f "${legacy_link}"
  systemctl --user daemon-reload
  echo "removed a leftover autostart symlink from an earlier install"
fi

echo "installed ${unit_dir}/kabr-train.service"
echo "  env      ${conf_dir}/env"
echo "  start    systemctl --user start kabr-train    (nothing starts until you run this)"
echo "  watch    journalctl --user -u kabr-train -f"
if [[ -n "${KABR_PCI_RESET:-}${KABR_POWER_LIMIT:-}" ]] && ! sudo -n true 2>/dev/null; then
  echo "note: bus reset and the power cap both need a sudoers rule to work unattended"
  echo "      ./utils/print_sudoers.sh | sudo tee /etc/sudoers.d/kabr-gpu"
fi
