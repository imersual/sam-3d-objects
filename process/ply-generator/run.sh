#!/bin/bash
set -e

export PATH="/opt/miniforge3/condabin:/opt/miniforge3/bin:/usr/local/bin:/usr/bin:/bin"

# Usage: ./run_sam3d.sh <image_path> <mask_path> <output_path>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <image_path> <mask_path> <output_path>"
    exit 1
fi

IMAGE_PATH="$1"
MASK_PATH="$2"
OUTPUT_PATH="$3"

# Directory of this script
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

# Run inference
cd "$SCRIPT_DIR"
python run_inference.py "$IMAGE_PATH" "$MASK_PATH" "$OUTPUT_PATH"
