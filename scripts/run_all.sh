#!/bin/bash
# Submit preprocessing (if needed) plus all three training jobs to SLURM

set -euo pipefail

REPO_DIR="/home/qtnguy50/gaze-01-baseline"
cd "${REPO_DIR}"

# Create output directory
mkdir -p runs

echo "=========================================="
echo "Checking precomputed heatmaps..."
echo "=========================================="

CSV_DIR="${REPO_DIR}/datasets/filtered_csvs/fold1"
HEATMAP_DIR="${REPO_DIR}/datasets/heatmaps/fixation_heatmaps"

if [[ ! -d "${CSV_DIR}" ]]; then
    echo "ERROR: CSV directory not found at ${CSV_DIR}"
    exit 1
fi

HEATMAPS_READY=1
if ! CSV_DIR="${CSV_DIR}" HEATMAP_DIR="${HEATMAP_DIR}" python - <<'PY'
import os
from pathlib import Path
from main_train_temporal import ensure_heatmaps_exist

csv_dir = Path(os.environ["CSV_DIR"])
heat_dir = Path(os.environ["HEATMAP_DIR"])
ensure_heatmaps_exist(csv_dir, heat_dir, require_temporal=True, sample_limit=5)
print("✓ All required heatmaps already exist.")
PY
then
    HEATMAPS_READY=0
fi

PREPROCESS_JOBID=""
if [[ "${HEATMAPS_READY}" -eq 0 ]]; then
    echo "Heatmaps missing — submitting preprocessing job first..."
    PREPROCESS_JOBID=$(sbatch scripts/preprocess_heatmaps.sh | awk '{print $4}')
    echo "Submitted heatmap preprocessing job: ${PREPROCESS_JOBID}"
    echo "Training jobs will wait for preprocessing to finish (afterok dependency)."
else
    echo "No preprocessing job needed."
fi

submit_job() {
    local label="$1"
    local script_path="$2"
    local needs_heatmaps="${3:-1}"
    local job_output
    echo ""
    echo "Submitting ${label}..."
    if [[ -n "${PREPROCESS_JOBID}" && "${needs_heatmaps}" -eq 1 ]]; then
        job_output=$(sbatch --dependency=afterok:${PREPROCESS_JOBID} "${script_path}")
    else
        job_output=$(sbatch "${script_path}")
    fi
    echo "${job_output}"
}

submit_job "UNet training" scripts/run_unet.sh 1
submit_job "Temporal training" scripts/run_temporal.sh 1
submit_job "GRADIA training" scripts/run_gradia.sh 0

echo ""
echo "All jobs submitted!"
echo "Check status with: squeue -u ${USER}"
echo "Monitor logs with: tail -f runs/*.log"
