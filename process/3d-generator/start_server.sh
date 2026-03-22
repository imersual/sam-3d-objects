#!/bin/bash
# Start the SAM3D persistent inference server inside the correct conda env.
# Run this ONCE before starting your task poller.
#
# Usage:
#   ./start_server.sh [port]   (default port: 8000)
#
# To run in background:
#   nohup ./start_server.sh > /var/log/sam3d-server.log 2>&1 &

set -e

export PATH="/opt/miniforge3/condabin:/opt/miniforge3/bin:/usr/local/bin:/usr/bin:/bin"

PORT="${1:-8000}"
# Which GPU to use. Default=1 to avoid GPU 0 shared with other processes.
# Set to 0,1 to use both GPUs, or 0 if you want GPU 0.
GPU="${2:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ── conda init ────────────────────────────────────────────────────────────────
if [ -f "${HOME}/conda/etc/profile.d/conda.sh" ]; then
    source "${HOME}/conda/etc/profile.d/conda.sh"
elif [ -f "/opt/miniforge3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniforge3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
    else
        echo "Error: Could not initialize conda"
        exit 1
    fi
fi

conda activate sam3d-objects

cd "$SCRIPT_DIR"
echo "Starting SAM3D server on port $PORT (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) ..."
python server.py --host 0.0.0.0 --port "$PORT"
