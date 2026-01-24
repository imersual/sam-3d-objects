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

# Initialize mamba
eval "$(mamba shell hook --shell bash)"

# Activate the correct environment
mamba activate sam3d-objects

# Run inference
cd "$SCRIPT_DIR"
python run_inference.py "$IMAGE_PATH" "$MASK_PATH" "$OUTPUT_PATH"
