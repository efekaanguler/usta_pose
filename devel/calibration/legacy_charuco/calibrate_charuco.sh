#!/usr/bin/env bash
# Preserved entry point for the former pairwise ChArUco daily calibration flow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../calibrate.sh" --method charuco "$@"
