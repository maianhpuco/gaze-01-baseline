#!/bin/bash
#SBATCH --job-name=unet_train
#SBATCH --output=runs/unet_training.log
#SBATCH --error=runs/unet_training.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate quynhng

# Force unbuffered output
export PYTHONUNBUFFERED=1

cd /home/qtnguy50/gaze-01-baseline

# Step 1: Preprocess heatmaps (run and wait for completion)
echo "=========================================="
echo "Step 1: Preprocessing heatmaps..."
echo "=========================================="

# Get total number of unique DICOM IDs in dataset
TOTAL_IMAGES=$(cat datasets/filtered_csvs/fold1/*.csv | tail -n +2 | cut -d',' -f1 | sort -u | wc -l)
echo "Total images in dataset: $TOTAL_IMAGES"

# Check if preprocessing is already complete
EXISTING_HEATMAPS=$(ls -1 datasets/heatmaps/fixation_heatmaps 2>/dev/null | wc -l)
echo "Existing heatmap directories: $EXISTING_HEATMAPS"

if [ "$EXISTING_HEATMAPS" -lt "$TOTAL_IMAGES" ]; then
    echo ""
    echo "Generating heatmap PNG files from fixations..."
    echo "This will take a while (processing $TOTAL_IMAGES images with 38 parallel sessions)..."
    echo ""
    
    # Run preprocessing
    stdbuf -oL -eL python -u preprocess_heatmaps.py
    
    # Verify completion
    FINAL_COUNT=$(ls -1 datasets/heatmaps/fixation_heatmaps 2>/dev/null | wc -l)
    echo ""
    echo "✓ Preprocessing complete! Generated $FINAL_COUNT/$TOTAL_IMAGES heatmap directories"
else
    echo "✓ All heatmaps already exist ($EXISTING_HEATMAPS/$TOTAL_IMAGES), skipping preprocessing"
fi

# Step 2: Train UNet model
echo ""
echo "=========================================="
echo "Step 2: Training UNet model..."
echo "=========================================="
# Set to use only 2 GPUs
export CUDA_VISIBLE_DEVICES=0,1
stdbuf -oL -eL python -u main_train_unet.py --config configs/train_unet.yaml

echo ""
echo "✓ UNet training completed!"

