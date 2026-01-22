#!/bin/bash
set -e

# Usage: ./run_sam3d.sh <task_dir> <image_path> <output_path>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <task_dir> <image_path> <output_path>"
    exit 1
fi

TASK_DIR="$1"
IMAGE_PATH="$2"
OUTPUT_PATH="$3"

# Directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Initialize mamba
eval "$(mamba shell hook --shell bash)"

# Activate the correct environment
mamba activate sam3d-objects

# Run inference
cd "$SCRIPT_DIR"
python run_inference.py "$TASK_DIR" "$IMAGE_PATH" "$OUTPUT_PATH"
