#!/bin/bash
set -e

export PATH="/opt/miniforge3/condabin:/opt/miniforge3/bin:/usr/local/bin:/usr/bin:/bin"

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <image_path> <mask_path1> [mask_path2 ...] <output_path>"
    exit 1
fi

IMAGE_PATH="$1"
OUTPUT_PATH="${@: -1}"
MASK_PATHS=("${@:2:$(($#-2))}")

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Initialize conda (works regardless of whether mamba is installed)
if [ -f "${HOME}/conda/etc/profile.d/conda.sh" ]; then
    source "${HOME}/conda/etc/profile.d/conda.sh"
elif [ -f "/opt/miniforge3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniforge3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Warning: conda.sh not found in expected locations, trying conda hook..."
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
    else
        echo "Error: Could not initialize conda"
        exit 1
    fi
fi

# Activate the correct environment
conda activate sam3d-objects

cd "$SCRIPT_DIR"
python run_inference.py "$IMAGE_PATH" "${MASK_PATHS[@]}" "$OUTPUT_PATH"