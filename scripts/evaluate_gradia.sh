#!/bin/bash
#SBATCH --job-name=eval_gradia
#SBATCH --output=runs/gradia_eval.log
#SBATCH --error=runs/gradia_eval.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate quynhng

# Set GPU
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Run evaluation
cd /home/qtnguy50/gaze-01-baseline

echo "============================================================"
echo "GRADIA TEST SET EVALUATION"
echo "============================================================"
echo "Checkpoint: runs/checkpoints/gradia_best.pt"
echo "Expected: Epoch 34, Val macroAUC = 0.860"
echo "============================================================"
echo

stdbuf -oL -eL python -u main_train_gradia.py \
    --config configs/data_egd-cxr.yaml \
    --attention_weight 0.0 \
    --iou_samples 50 \
    --evaluate \
    --checkpoint runs/checkpoints/gradia_best.pt

echo
echo "============================================================"
echo "GRADIA EVALUATION COMPLETE"
echo "============================================================"

