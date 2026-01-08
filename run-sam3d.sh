#!/bin/bash
set -e

# Activate mamba environment and run inference
# Usage: ./run_sam3d.sh <task_dir> <image_path> <output_path>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <task_dir> <image_path> <output_path>"
    exit 1
fi

TASK_DIR="$1"
IMAGE_PATH="$2"
OUTPUT_PATH="$3"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate mamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$SCRIPT_DIR"

# Run inference script
cd "$SCRIPT_DIR"
python run_inference.py "$TASK_DIR" "$IMAGE_PATH" "$OUTPUT_PATH"
