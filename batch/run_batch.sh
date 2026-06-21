#!/bin/bash
# =============================================================================
# run_batch.sh - master orchestrator for the SAM3D depth-pipeline.
#
# For every image under  $INPUT_DIR/<name>/{image.*, mask.png}  it:
#   1. runs Lotus-2        (env: lotus2)   -> output/images/<name>/lotus2/pointmap.pt
#   2. runs Depth Anything 3 (env: da3)    -> output/images/<name>/da3/pointmap.pt
#   3. runs Depth Pro      (env: depthpro) -> output/images/<name>/depthpro/pointmap.pt
#   4. runs MoGe-2         (env: moge2)    -> output/images/<name>/moge2/pointmap.pt
#   5. runs SAM3D          (env: sam3d)    -> output/images/<name>/splat_*.glb
#                                             (one GLB per backend + MoGe v1 default)
#
# Each depth model lives in its own conda env; we switch envs per stage.
# A failure in one stage/image is logged and skipped; the batch continues.
#
# Usage:
#   ./run_batch.sh                # process every folder under $INPUT_DIR
#   ./run_batch.sh foo bar        # process only input/images/foo and input/images/bar
# Override config via env vars, e.g.  RUN_DEPTHPRO=0 GPU=1 ./run_batch.sh
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/config.sh"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- conda init (mirror the project's run.sh) -------------------------------
init_conda() {
    if [ -f "${HOME}/conda/etc/profile.d/conda.sh" ]; then
        source "${HOME}/conda/etc/profile.d/conda.sh"
    elif [ -f "/opt/miniforge3/etc/profile.d/conda.sh" ]; then
        source "/opt/miniforge3/etc/profile.d/conda.sh"
    elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    elif command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
    else
        echo "ERROR: could not initialize conda" >&2
        exit 1
    fi
}
init_conda

# --- helpers ----------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Find the input image inside a sample folder (image.* but not mask.png).
find_image() {
    local dir="$1"
    for ext in jpg jpeg png JPG JPEG PNG; do
        if [ -f "${dir}/image.${ext}" ]; then echo "${dir}/image.${ext}"; return 0; fi
    done
    # fallback: first non-mask image in the folder
    for f in "${dir}"/*.{jpg,jpeg,png,JPG,JPEG,PNG}; do
        [ -f "$f" ] || continue
        case "$(basename "$f")" in mask.*) continue;; esac
        echo "$f"; return 0
    done
    return 1
}

# --- collect samples --------------------------------------------------------
if [ "$#" -gt 0 ]; then
    SAMPLES=("$@")
else
    SAMPLES=()
    for d in "${INPUT_DIR}"/*/; do
        [ -d "$d" ] && SAMPLES+=("$(basename "$d")")
    done
fi

if [ "${#SAMPLES[@]}" -eq 0 ]; then
    log "No samples found under ${INPUT_DIR}"
    exit 0
fi

log "Batch start: ${#SAMPLES[@]} sample(s) | GPU=${CUDA_VISIBLE_DEVICES}"
log "  input : ${INPUT_DIR}"
log "  output: ${OUTPUT_DIR}"

OK=0; FAIL=0
for NAME in "${SAMPLES[@]}"; do
    IN_DIR="${INPUT_DIR}/${NAME}"
    OUT="${OUTPUT_DIR}/${NAME}"
    log "==================== ${NAME} ===================="

    if [ ! -d "${IN_DIR}" ]; then log "  missing input dir ${IN_DIR} -> skip"; FAIL=$((FAIL+1)); continue; fi
    IMG="$(find_image "${IN_DIR}")" || { log "  no image in ${IN_DIR} -> skip"; FAIL=$((FAIL+1)); continue; }
    MASK="${IN_DIR}/mask.png"
    if [ ! -f "${MASK}" ]; then log "  no mask.png in ${IN_DIR} -> skip"; FAIL=$((FAIL+1)); continue; fi
    IMG_STEM="$(basename "${IMG}")"; IMG_STEM="${IMG_STEM%.*}"
    mkdir -p "${OUT}"
    log "  image=${IMG}  mask=${MASK}"

    # ---- 1. Lotus-2 -------------------------------------------------------
    if [ "${RUN_LOTUS}" = "1" ]; then
        log "  [lotus2] depth inference"
        (
            set -e
            conda activate "${ENV_LOTUS}"
            mkdir -p "${OUT}/lotus2/_input"
            cp "${IMG}" "${OUT}/lotus2/_input/"
            cd "${LOTUS_DIR}"
            python infer.py \
                --input_dir="${OUT}/lotus2/_input" \
                --output_dir="${OUT}/lotus2" \
                --seed="0" --task_name=depth
        ) && (
            set -e
            conda activate "${ENV_SAM3D}"
            python "${SCRIPT_DIR}/lotus2_to_pointmap.py" \
                --depth "${OUT}/lotus2/depth_npy/${IMG_STEM}.npy" \
                --out-dir "${OUT}/lotus2" \
                --fov-h "${LOTUS_FOV_H}" --near "${LOTUS_NEAR}" --far "${LOTUS_FAR}"
        ) || log "  [lotus2] FAILED (continuing)"
    fi

    # ---- 2. Depth Anything 3 ---------------------------------------------
    if [ "${RUN_DA3}" = "1" ]; then
        log "  [da3] depth inference"
        (
            set -e
            conda activate "${ENV_DA3}"
            python "${SCRIPT_DIR}/run_da3.py" --image "${IMG}" --out-dir "${OUT}/da3"
        ) && (
            set -e
            conda activate "${ENV_SAM3D}"
            python "${SCRIPT_DIR}/depth_anything_to_pointmap.py" \
                --depth      "${OUT}/da3/depth.npy" \
                --confidence "${OUT}/da3/confidence.npy" \
                --intrinsics "${OUT}/da3/intrinsics.npy" \
                --extrinsics "${OUT}/da3/extrinsics.npy" \
                --out-dir    "${OUT}/da3" \
                --conf-percentile "${DA3_CONF_PERCENTILE}"
        ) || log "  [da3] FAILED (continuing)"
    fi

    # ---- 3. Depth Pro -----------------------------------------------------
    if [ "${RUN_DEPTHPRO}" = "1" ]; then
        log "  [depthpro] depth inference + pointmap"
        (
            set -e
            conda activate "${ENV_DEPTHPRO}"
            cd "${DEPTHPRO_DIR}"
            python "${SCRIPT_DIR}/run_depthpro.py" --image "${IMG}" --out-dir "${OUT}/depthpro"
        ) || log "  [depthpro] FAILED (continuing)"
    fi

    # ---- 4. MoGe-2 (own env; outputs metric pointmap directly) -----------
    if [ "${RUN_MOGE2}" = "1" ]; then
        log "  [moge2] depth inference + pointmap"
        (
            set -e
            conda activate "${ENV_MOGE2}"
            python "${SCRIPT_DIR}/run_moge2.py" \
                --image "${IMG}" --out-dir "${OUT}/moge2" --model "${MOGE2_MODEL}"
        ) || log "  [moge2] FAILED (continuing)"
    fi

    # ---- 5. SAM3D (one GLB per backend + MoGe default) -------------------
    if [ "${RUN_SAM3D}" = "1" ]; then
        log "  [sam3d] reconstruction"
        (
            set -e
            conda activate "${ENV_SAM3D}"
            cd "${SAM3D_DIR}"
            SKIP_FLAG=""
            [ "${SAM3D_SKIP_EXISTING}" = "1" ] && SKIP_FLAG="--skip-existing"
            python "${SCRIPT_DIR}/batch_sam3d.py" \
                --image "${IMG}" --mask "${MASK}" --out-dir "${OUT}" --seed "${SAM3D_SEED}" ${SKIP_FLAG}
        ) && { OK=$((OK+1)); log "  DONE ${NAME}"; } \
          || { FAIL=$((FAIL+1)); log "  [sam3d] FAILED ${NAME}"; }
    else
        OK=$((OK+1))
    fi
done

log "Batch finished: ${OK} ok, ${FAIL} failed."
