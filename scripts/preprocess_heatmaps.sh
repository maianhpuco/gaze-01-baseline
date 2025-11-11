#!/usr/bin/env bash
#SBATCH --job-name=preprocess_heatmaps
#SBATCH --output=runs/preprocess_heatmaps_%j.log
#SBATCH --error=runs/preprocess_heatmaps_%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate quynhng
export PYTHONUNBUFFERED=1

REPO_DIR="/home/qtnguy50/gaze-01-baseline"
cd "${REPO_DIR}"
mkdir -p runs

echo "=========================================="
echo "Running heatmap preprocessing pipeline..."
echo "Output logs: runs/preprocess_heatmaps_${SLURM_JOB_ID:-local}.log/.err"
echo "=========================================="

python -u preprocess_heatmaps.py

echo ""
echo "✓ Heatmap preprocessing finished."
