#!/bin/bash
# Submit all three training jobs to SLURM

cd /home/qtnguy50/gaze-01-baseline

# Create output directory
mkdir -p runs

# Submit jobs
echo "Submitting UNet training..."
sbatch scripts/run_unet.sh

echo "Submitting Temporal training..."
sbatch scripts/run_temporal.sh

echo "Submitting GRADIA training..."
sbatch scripts/run_gradia.sh

echo ""
echo "All jobs submitted!"
echo "Check status with: squeue -u $USER"
echo "Monitor logs with: tail -f runs/*.log"
