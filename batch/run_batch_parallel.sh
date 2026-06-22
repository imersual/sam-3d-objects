#!/bin/bash
# =============================================================================
# run_batch_parallel.sh - run the SAM3D batch across multiple GPUs at once.
#
# Shards the sample list round-robin across the GPUs in $GPUS and launches one
# run_batch.sh per GPU concurrently. Each worker is pinned to its own GPU (via
# the GPU env var -> CUDA_VISIBLE_DEVICES), so there is no cross-GPU contention;
# you get ~Nx throughput for N GPUs. Output dirs are per-sample, so the workers
# never collide.
#
# Usage:
#   ./run_batch_parallel.sh                 # every folder under $INPUT_DIR, 2 GPUs
#   ./run_batch_parallel.sh foo bar baz     # only these samples
#   GPUS="0 1"   ./run_batch_parallel.sh    # choose which GPUs (default "0 1")
#   GPUS="0 1 2 3" ./run_batch_parallel.sh  # scales to any number of GPUs
# Any RUN_*/SAM3D_* override from config.sh still applies, e.g.
#   RUN_DEPTHPRO=0 GPUS="0 1" ./run_batch_parallel.sh
#
# Per-GPU logs are written to $OUTPUT_DIR/_logs/gpu<N>.log . Follow them with:
#   tail -f output/images/_logs/gpu0.log
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/config.sh"

# GPUs to spread work across (space-separated list of CUDA device ids).
GPUS="${GPUS:-0 1}"
read -ra GPU_ARR <<< "${GPUS}"
N="${#GPU_ARR[@]}"
if [ "${N}" -eq 0 ]; then echo "ERROR: GPUS is empty" >&2; exit 1; fi

# --- collect samples (args, else every folder under INPUT_DIR) --------------
if [ "$#" -gt 0 ]; then
    SAMPLES=("$@")
else
    SAMPLES=()
    for d in "${INPUT_DIR}"/*/; do
        [ -d "$d" ] && SAMPLES+=("$(basename "$d")")
    done
fi
if [ "${#SAMPLES[@]}" -eq 0 ]; then
    echo "No samples found under ${INPUT_DIR}"; exit 0
fi

LOG_DIR="${OUTPUT_DIR}/_logs"
mkdir -p "${LOG_DIR}"

echo "[parallel] ${#SAMPLES[@]} sample(s) across ${N} GPU(s): ${GPUS}"

# --- shard round-robin and launch one worker per GPU ------------------------
pids=()
gpus_used=()
for i in "${!GPU_ARR[@]}"; do
    shard=()
    for j in "${!SAMPLES[@]}"; do
        if [ "$(( j % N ))" -eq "$i" ]; then shard+=("${SAMPLES[$j]}"); fi
    done
    [ "${#shard[@]}" -eq 0 ] && continue

    gpu="${GPU_ARR[$i]}"
    log="${LOG_DIR}/gpu${gpu}.log"
    echo "[parallel] GPU ${gpu}: ${#shard[@]} sample(s) -> ${log}"
    GPU="${gpu}" "${SCRIPT_DIR}/run_batch.sh" "${shard[@]}" >"${log}" 2>&1 &
    pids+=("$!")
    gpus_used+=("${gpu}")
done

# --- wait for all workers, report per-GPU exit status -----------------------
rc=0
for k in "${!pids[@]}"; do
    if wait "${pids[$k]}"; then
        echo "[parallel] GPU ${gpus_used[$k]} finished OK"
    else
        echo "[parallel] GPU ${gpus_used[$k]} FAILED (see ${LOG_DIR}/gpu${gpus_used[$k]}.log)"
        rc=1
    fi
done

echo "[parallel] all workers done (rc=${rc}). Logs in ${LOG_DIR}"
exit "${rc}"
