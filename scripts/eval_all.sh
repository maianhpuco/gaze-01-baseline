#!/usr/bin/env bash
# Evaluate GRADIA, Temporal, and UNet models on the test split using the best checkpoints.

set -euo pipefail

CONFIG=${1:-"configs/train_gazegnn.yaml"}
GPU=${2:-"0"}

# Get absolute path to script directory, then go to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"
RUN_DIR="${ROOT_DIR}/run"
CHECKPOINTS_DIR="${ROOT_DIR}/checkpoints"
CACHE_DIR=${CACHE_DIR:-${ROOT_DIR}/cache/heatmaps}

# Change to root directory to ensure relative paths work correctly
cd "${ROOT_DIR}"

echo "=================================================="
echo "Running evaluations with config: ${CONFIG}"
echo "GPU: ${GPU}"
echo "Root: ${ROOT_DIR}"
echo "=================================================="

# ------------- Helper functions -------------
fail() {
    echo "ERROR: $1" >&2
    exit 1
}

latest_run_with_best() {
    local pattern="$1"
    local best_file="$2"
    local latest
    latest=$(find "${pattern}" -maxdepth 1 -mindepth 1 -type d -print 2>/dev/null | sort | while read -r dir; do
        if [[ -f "${dir}/${best_file}" ]]; then
            echo "${dir}"
        fi
    done | tail -n 1)
    echo "${latest}"
}

# ------------- GRADIA -------------
echo ""
echo ">>> Evaluating GRADIA"
# Try checkpoints directory first, then final_checkpoint, then run directory, then results directory
if [[ -f "${CHECKPOINTS_DIR}/gradia_best_weights.pth" ]]; then
    echo "Using checkpoint from checkpoints directory"
    GRADIA_TEMP_DIR="${CHECKPOINTS_DIR}/gradia_eval_temp"
    mkdir -p "${GRADIA_TEMP_DIR}"
    cp "${CHECKPOINTS_DIR}/gradia_best_weights.pth" "${GRADIA_TEMP_DIR}/model_phase1_best.pth"
    GRADIA_DIR="${GRADIA_TEMP_DIR}"
elif [[ -f "${ROOT_DIR}/final_checkpoint/gradia_best_weights.pth" ]]; then
    echo "Using checkpoint from final_checkpoint directory"
    GRADIA_TEMP_DIR="${ROOT_DIR}/final_checkpoint/gradia_eval_temp"
    mkdir -p "${GRADIA_TEMP_DIR}"
    cp "${ROOT_DIR}/final_checkpoint/gradia_best_weights.pth" "${GRADIA_TEMP_DIR}/model_phase1_best.pth"
    GRADIA_DIR="${GRADIA_TEMP_DIR}"
elif [[ -d "${RUN_DIR}/gradia" ]]; then
    # Check for best model or use latest epoch
    if [[ -f "${RUN_DIR}/gradia/model_phase1_best.pth" ]]; then
        echo "Using checkpoint from run directory (best model)"
        GRADIA_DIR="${RUN_DIR}/gradia"
    elif [[ -f "${RUN_DIR}/gradia/model_phase2_best.pth" ]]; then
        echo "Using checkpoint from run directory (phase2 best model)"
        GRADIA_DIR="${RUN_DIR}/gradia"
    else
        # Find latest epoch file
        LATEST_EPOCH=$(ls -t "${RUN_DIR}/gradia"/model_phase1_epoch*.pth 2>/dev/null | head -1)
        if [[ -n "${LATEST_EPOCH}" ]]; then
            echo "Using latest epoch from run directory: $(basename ${LATEST_EPOCH})"
            GRADIA_DIR="${RUN_DIR}/gradia"
        else
            fail "No GRADIA checkpoint found in run directory"
        fi
    fi
elif [[ -d "${RESULTS_DIR}/gradia" ]] && [[ -f "${RESULTS_DIR}/gradia/model_phase1_best.pth" ]]; then
    GRADIA_DIR="${RESULTS_DIR}/gradia"
else
    fail "GRADIA checkpoint not found in checkpoints, final_checkpoint, run, or results directory"
fi

python "${ROOT_DIR}/main_train_gradia.py" \
    --config "${CONFIG}" \
    --model_dir "${GRADIA_DIR}" \
    --gpus "${GPU}" \
    --test_only \
    --cache_dir "${CACHE_DIR}" || echo "Warning: GRADIA evaluation failed or no checkpoint found"

# Cleanup temp directory if created
[[ -d "${GRADIA_TEMP_DIR}" ]] && rm -rf "${GRADIA_TEMP_DIR}" 2>/dev/null || true

# ------------- Temporal model -------------
echo ""
echo ">>> Evaluating Temporal model"
# Try checkpoints directory first, then final_checkpoint, then run directory, then results directory
if [[ -f "${CHECKPOINTS_DIR}/temporal_best_weights.pth" ]]; then
    echo "Using checkpoint from checkpoints directory"
    TEMP_TEMP_DIR="${CHECKPOINTS_DIR}/temporal_eval_temp"
    mkdir -p "${TEMP_TEMP_DIR}"
    cp "${CHECKPOINTS_DIR}/temporal_best_weights.pth" "${TEMP_TEMP_DIR}/best_weights.pth"
    LATEST_TEMP="${TEMP_TEMP_DIR}"
elif [[ -f "${ROOT_DIR}/final_checkpoint/temporal_best_weights.pth" ]]; then
    echo "Using checkpoint from final_checkpoint directory"
    TEMP_TEMP_DIR="${ROOT_DIR}/final_checkpoint/temporal_eval_temp"
    mkdir -p "${TEMP_TEMP_DIR}"
    cp "${ROOT_DIR}/final_checkpoint/temporal_best_weights.pth" "${TEMP_TEMP_DIR}/best_weights.pth"
    LATEST_TEMP="${TEMP_TEMP_DIR}"
else
    # Try run directory first
    TEMP_PATTERN="${RUN_DIR}/temporal"
    LATEST_TEMP=$(latest_run_with_best "${TEMP_PATTERN}" "best_weights.pth")
    if [[ -z "${LATEST_TEMP}" ]]; then
        # Fall back to results directory
        TEMP_PATTERN="${RESULTS_DIR}/temporal"
        LATEST_TEMP=$(latest_run_with_best "${TEMP_PATTERN}" "best_weights.pth")
    fi
    [[ -n "${LATEST_TEMP}" ]] || fail "No temporal run with best_weights.pth found in checkpoints, final_checkpoint, run, or results directory"
    echo "Using checkpoint from: ${LATEST_TEMP}"
fi

python "${ROOT_DIR}/main_train_temporal.py" \
    --config "${CONFIG}" \
    --output_dir "${RESULTS_DIR}/temporal" \
    --gpus "${GPU}" \
    --test \
    --testdir "${LATEST_TEMP}" \
    --cache_dir "${CACHE_DIR}" || echo "Warning: Temporal evaluation failed"

# Cleanup temp directory if created
[[ -d "${TEMP_TEMP_DIR}" ]] && rm -rf "${TEMP_TEMP_DIR}" 2>/dev/null || true

# ------------- UNet model -------------
echo ""
echo ">>> Evaluating UNet model"
# Try checkpoints directory first, then final_checkpoint, then run directory, then results directory
if [[ -f "${CHECKPOINTS_DIR}/unet_best_weights.pth" ]]; then
    echo "Using checkpoint from checkpoints directory"
    UNET_TEMP_DIR="${CHECKPOINTS_DIR}/unet_eval_temp"
    mkdir -p "${UNET_TEMP_DIR}"
    cp "${CHECKPOINTS_DIR}/unet_best_weights.pth" "${UNET_TEMP_DIR}/best_weights.pth"
    LATEST_UNET="${UNET_TEMP_DIR}"
elif [[ -f "${ROOT_DIR}/final_checkpoint/unet_best_weights.pth" ]]; then
    echo "Using checkpoint from final_checkpoint directory"
    UNET_TEMP_DIR="${ROOT_DIR}/final_checkpoint/unet_eval_temp"
    mkdir -p "${UNET_TEMP_DIR}"
    cp "${ROOT_DIR}/final_checkpoint/unet_best_weights.pth" "${UNET_TEMP_DIR}/best_weights.pth"
    LATEST_UNET="${UNET_TEMP_DIR}"
else
    # Try run directory first
    UNET_PATTERN="${RUN_DIR}/unet"
    LATEST_UNET=$(latest_run_with_best "${UNET_PATTERN}" "best_weights.pth")
    if [[ -z "${LATEST_UNET}" ]]; then
        # Fall back to results directory
        UNET_PATTERN="${RESULTS_DIR}/unet"
        LATEST_UNET=$(latest_run_with_best "${UNET_PATTERN}" "best_weights.pth")
    fi
    [[ -n "${LATEST_UNET}" ]] || fail "No UNet run with best_weights.pth found in checkpoints, final_checkpoint, run, or results directory"
    echo "Using checkpoint from: ${LATEST_UNET}"
fi

python "${ROOT_DIR}/main_train_unet.py" \
    --config "${CONFIG}" \
    --output_dir "${RESULTS_DIR}/unet" \
    --gpus "${GPU}" \
    --test \
    --testdir "${LATEST_UNET}" \
    --cache_dir "${CACHE_DIR}" || echo "Warning: UNet evaluation failed"

# Cleanup temp directory if created
[[ -d "${UNET_TEMP_DIR}" ]] && rm -rf "${UNET_TEMP_DIR}" 2>/dev/null || true

echo ""
echo "All evaluations finished."

