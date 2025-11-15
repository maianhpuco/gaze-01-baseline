#!/bin/bash
# Master Script to Submit All Training Models as Separate Parallel Jobs
# Submits 3 independent sbatch jobs:
#   1. Temporal Model (run_temporal.sh)
#   2. UNet Classifier (run_unet.sh)
#   3. GRADIA Two-Phase (run_gradia.sh)

echo "=========================================================="
echo "Master Training Script - Submit All Models in Parallel"
echo "Start time: $(date)"
echo "=========================================================="

# Create necessary directories
mkdir -p logs
mkdir -p results/temporal
mkdir -p results/unet
mkdir -p results/gradia

CACHE_DIR=${CACHE_DIR:-cache/heatmaps}
CONFIG_PATH=${CONFIG_PATH:-configs/train_gazegnn.yaml}

echo ""
echo "Checking cached heatmaps at ${CACHE_DIR}..."
if [ ! -f "${CACHE_DIR}/meta.json" ] || [ "${PREPROCESS_OVERWRITE:-0}" -eq 1 ]; then
    echo "Cache missing or overwrite requested. Precomputing heatmaps before submissions."
    source /opt/anaconda3/etc/profile.d/conda.sh
    conda activate quynhng
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to activate quynhng environment for preprocessing."
        exit 1
    fi
    python scripts/precompute_heatmaps.py --config "$CONFIG_PATH" --cache_dir "$CACHE_DIR"
    PREPROC_STATUS=$?
    conda deactivate
    if [ $PREPROC_STATUS -ne 0 ]; then
        echo "ERROR: Heatmap preprocessing failed (exit code $PREPROC_STATUS). Aborting submissions."
        exit $PREPROC_STATUS
    fi
else
    echo "Heatmap cache already prepared."
fi

echo ""
echo "Submitting models as separate parallel sbatch jobs..."
echo ""

# Submit Temporal Model
echo "1. Submitting Temporal Model..."
TEMPORAL_JOB=$(sbatch --parsable --export=ALL,CACHE_DIR=$CACHE_DIR scripts/run_temporal.sh)
if [ $? -eq 0 ]; then
    echo "   ✓ Temporal job submitted: Job ID $TEMPORAL_JOB"
    echo "     Log: logs/temporal_${TEMPORAL_JOB}.out"
else
    echo "   ✗ Failed to submit Temporal job"
    TEMPORAL_JOB="FAILED"
fi

# Submit UNet Model
echo "2. Submitting UNet Model..."
UNET_JOB=$(sbatch --parsable --export=ALL,CACHE_DIR=$CACHE_DIR scripts/run_unet.sh)
if [ $? -eq 0 ]; then
    echo "   ✓ UNet job submitted: Job ID $UNET_JOB"
    echo "     Log: logs/unet_${UNET_JOB}.out"
else
    echo "   ✗ Failed to submit UNet job"
    UNET_JOB="FAILED"
fi

# Submit GRADIA Model
echo "3. Submitting GRADIA Model..."
GRADIA_JOB=$(sbatch --parsable scripts/run_gradia.sh)
if [ $? -eq 0 ]; then
    echo "   ✓ GRADIA job submitted: Job ID $GRADIA_JOB"
    echo "     Log: logs/gradia_${GRADIA_JOB}.out"
else
    echo "   ✗ Failed to submit GRADIA job"
    GRADIA_JOB="FAILED"
fi

echo ""
echo "=========================================================="
echo "All jobs submitted!"
echo "=========================================================="
echo ""
echo "Job Summary:"
echo "  Temporal: Job $TEMPORAL_JOB"
echo "  UNet:     Job $UNET_JOB"
# echo "  GRADIA:   Job $GRADIA_JOB"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in: logs/"
echo ""
echo "Cancel jobs with:"
if [ "$TEMPORAL_JOB" != "FAILED" ]; then
    echo "  scancel $TEMPORAL_JOB  # Temporal"
fi
if [ "$UNET_JOB" != "FAILED" ]; then
    echo "  scancel $UNET_JOB  # UNet"
fi
if [ "$GRADIA_JOB" != "FAILED" ]; then
    echo "  scancel $GRADIA_JOB  # GRADIA"
fi
echo ""
echo "Submission complete at: $(date)"
echo "=========================================================="
