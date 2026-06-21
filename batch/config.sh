#!/bin/bash
# config.sh - paths and conda env names for the batch pipeline.
# Sourced by run_batch.sh. Override any value by exporting it before running,
# e.g.  LOTUS_DIR=/data/Lotus-2 ./run_batch.sh

# --- repos (on the remote /workspace box) -----------------------------------
: "${SAM3D_DIR:=/workspace/sam-3d-objects}"
: "${LOTUS_DIR:=/workspace/Lotus-2}"
: "${DA3_DIR:=/workspace/Depth-Anything-3}"
: "${DEPTHPRO_DIR:=/workspace/ml-depth-pro}"

# --- conda/mamba env names ---------------------------------------------------
: "${ENV_SAM3D:=sam3d-objects}"
: "${ENV_LOTUS:=lotus2}"
: "${ENV_DA3:=da3}"
: "${ENV_DEPTHPRO:=depthpro}"

# --- input / output ----------------------------------------------------------
: "${INPUT_DIR:=${SAM3D_DIR}/input/images}"
: "${OUTPUT_DIR:=${SAM3D_DIR}/output/images}"

# --- HuggingFace cache (large models live on the workspace disk) -------------
: "${HF_HOME:=/workspace/hf_cache}"
: "${TRANSFORMERS_CACHE:=/workspace/hf_cache}"
: "${HF_HUB_CACHE:=/workspace/hf_cache/hub}"
export HF_HOME TRANSFORMERS_CACHE HF_HUB_CACHE

# --- which backends to run (set to 0 to disable) -----------------------------
: "${RUN_LOTUS:=1}"
: "${RUN_DA3:=1}"
: "${RUN_DEPTHPRO:=1}"
: "${RUN_SAM3D:=1}"

# --- conversion parameters ---------------------------------------------------
# Lotus-2 has no scale/intrinsics, so these must be tuned to your scene.
: "${LOTUS_FOV_H:=24}"     # 24=85mm FF (product/food), 70=phone, 90=wide
: "${LOTUS_NEAR:=0.2}"     # closest depth in metres
: "${LOTUS_FAR:=1.5}"      # farthest depth in metres
: "${DA3_CONF_PERCENTILE:=0}"  # discard bottom N% confidence pixels (0 = keep all)
: "${SAM3D_SEED:=1}"
: "${SAM3D_SKIP_EXISTING:=0}"  # 1 = keep existing splat_*.glb, only render missing ones
: "${GPU:=0}"              # CUDA_VISIBLE_DEVICES
