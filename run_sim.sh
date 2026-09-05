#!/bin/bash
# One-command launcher for SO-101 MuJoCo Simulation

PYTHON_BIN="/Users/nam/miniforge3/envs/lerobot/bin/mjpython"
FALLBACK_BIN="/Users/nam/miniforge3/envs/lerobot/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/sim_so101_pick.py"

if [ -f "$PYTHON_BIN" ]; then
    exec "$PYTHON_BIN" "$TARGET_SCRIPT" "$@"
else
    exec "$FALLBACK_BIN" "$TARGET_SCRIPT" "$@"
fi
