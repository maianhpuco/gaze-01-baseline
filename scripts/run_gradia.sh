#!/bin/bash
#SBATCH --job-name=gradia_train
#SBATCH --output=runs/gradia_training.log
#SBATCH --error=runs/gradia_training.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate quynhng

# Set to use only 2 GPUs
export CUDA_VISIBLE_DEVICES=4,5

# Force unbuffered output
export PYTHONUNBUFFERED=1

# Run training
cd /home/qtnguy50/gaze-01-baseline
stdbuf -oL -eL python -u main_train_gradia.py --config configs/data_egd-cxr.yaml \
    --attention_weight 1.0 --iou_samples 50

echo "GRADIA training completed!"

