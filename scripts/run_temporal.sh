#!/bin/bash
#SBATCH --job-name=temporal_train
#SBATCH --output=logs/temporal_%j.out
#SBATCH --error=logs/temporal_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=farsightQ
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Temporal Model Training Script
# Reproduces exact steps from eye-gaze-dataset/Experiments/main.py

echo "=================================================="
echo "Temporal Model Training"
echo "Start time: $(date)"
echo "=================================================="

# Create log directory
mkdir -p logs
mkdir -p results/temporal

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src/externals/eye-gaze-dataset/Experiments"

# GPU setup
export CUDA_VISIBLE_DEVICES=0,1
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Configuration
# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate quynhng

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate quynhng environment"
    exit 1
fi

echo "Environment activated: $(which python)"

CONFIG="configs/train_gazegnn.yaml"
OUTPUT_DIR="results/temporal"
BATCH_SIZE=32
EPOCHS=500
LR=3e-4
MODEL_TYPE="temporal"
NUM_WORKERS=4
CACHE_DIR=${CACHE_DIR:-cache/heatmaps}
MAX_FIXATIONS=${MAX_FIXATIONS:-10}

echo ""
echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Model Type: $MODEL_TYPE"
echo ""

echo "=================================================="
echo "TRAINING PHASE"
echo "=================================================="

echo ""
echo "Preparing cached heatmaps in ${CACHE_DIR} (if needed)..."
if [ ! -f "${CACHE_DIR}/meta.json" ] || [ "${PREPROCESS_OVERWRITE:-0}" -eq 1 ]; then
    python scripts/precompute_heatmaps.py --config "$CONFIG" --cache_dir "$CACHE_DIR" --max_fixations "$MAX_FIXATIONS"
else
    echo "Cache metadata found at ${CACHE_DIR}/meta.json. Skipping preprocessing."
fi

python main_train_temporal.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay 1e-4 \
    --model_type $MODEL_TYPE \
    --num_workers $NUM_WORKERS \
    --gpus 0,1 \
    --rseed 42 \
    --dropout 0.5 \
    --hidden_dim 64 \
    --emb_dim 64 \
    --hidden_hm 256 128 \
    --num_layers_hm 1 \
    --cell lstm \
    --brnn_hm \
    --attention \
    --max_fixations $MAX_FIXATIONS \
    --cache_dir "$CACHE_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Training completed successfully!"
    echo "=================================================="
    
    # Find the latest model directory
    LATEST_MODEL=$(ls -td $OUTPUT_DIR/*/ | head -1)
    echo "Model saved at: $LATEST_MODEL"
    
    # Display metrics summary
    echo ""
    echo "=================================================="
    echo "METRICS SUMMARY"
    echo "=================================================="
    
    if [ -d "$LATEST_MODEL/plots" ]; then
        echo "Test Results:"
        for log_file in "$LATEST_MODEL/plots"/*.log; do
            if [ -f "$log_file" ]; then
                echo "--- $(basename $log_file) ---"
                cat "$log_file"
                echo ""
            fi
        done
    fi
    
    echo ""
    echo "=================================================="
    echo "Training and Testing Complete"
    echo "End time: $(date)"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "ERROR: Training failed with exit code $EXIT_CODE"
    echo "End time: $(date)"
    echo "=================================================="
    exit $EXIT_CODE
fi

# Display checkpoint information
echo ""
echo "Saved checkpoints:"
ls -lh $OUTPUT_DIR/*/Epoch_*.pth 2>/dev/null || echo "No checkpoints found"

echo ""
echo "Temporal training script completed!"

