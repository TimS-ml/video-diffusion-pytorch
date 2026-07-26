#!/usr/bin/env bash
# Print the sudoers rule the supervisor needs to recover an external GPU on its own.
#
#   ./utils/print_sudoers.sh | sudo tee /etc/sudoers.d/kabr-gpu
#   sudo chmod 0440 /etc/sudoers.d/kabr-gpu
#
# Two capabilities, no more: re-enumerate a PCI device that has fallen off the bus, and
# set a power cap that the card forgets every time it disappears. Without this rule both
# KABR_PCI_RESET and KABR_POWER_LIMIT degrade to a log line, and every dropout waits for
# someone to walk over and replug the cable.
#
# This script deliberately does not write to /etc itself, so nothing here ever runs as
# root on your behalf. Read the output before you pipe it into tee.
set -euo pipefail

user="${SUDO_USER:-${USER}}"
nvidia_smi="$(command -v nvidia-smi || echo /usr/bin/nvidia-smi)"
tee_bin="$(command -v tee || echo /usr/bin/tee)"

cat <<RULE
# Installed by utils/print_sudoers.sh - lets the training supervisor recover a GPU that
# drops off the bus, without granting a general root shell.
#
# The device path is a wildcard rather than one bus address because a card that
# re-enumerates can come back somewhere else. Removing a PCI device can knock hardware
# offline until the next rescan or reboot, but it cannot escalate privileges, so the
# blast radius stays a denial of service on this machine.
${user} ALL=(root) NOPASSWD: ${tee_bin} /sys/bus/pci/devices/*/remove
${user} ALL=(root) NOPASSWD: ${tee_bin} /sys/bus/pci/rescan
${user} ALL=(root) NOPASSWD: ${nvidia_smi} -i [0-9] -pl [0-9]*
RULE
