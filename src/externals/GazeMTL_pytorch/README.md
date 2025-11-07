# GazeMTL PyTorch - Modern Implementation

A modern PyTorch-only implementation of the Gaze Multi-Task Learning (GazeMTL) framework for medical image classification. This version removes the dependency on Emmental and uses the latest PyTorch libraries and best practices.

## Features

- **Pure PyTorch**: No Emmental dependency, uses only standard PyTorch
- **Modern Libraries**: Uses latest versions of PyTorch, torchvision, and torchmetrics
- **Multi-Task Learning**: Supports multiple helper tasks (diffusivity, location, time)
- **ResNet50 Backbone**: Pretrained ResNet50 encoder with task-specific heads
- **Comprehensive Metrics**: Accuracy, AUC, precision, recall, F1 score
- **Checkpointing**: Automatic best model saving and resume capability
- **Flexible Configuration**: Easy to modify and extend

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python train.py \
    --source cxr2 \
    --data_dir /path/to/data \
    --gaze_mtl_task diffusivity \
    --pretrained \
    --task_weights "1.0" \
    --log_path ./logs/cxr2/diffusivity/seed_0 \
    --n_epochs 15 \
    --lr 0.0001 \
    --l2 0.01 \
    --batch_size 32
```

### Using the Training Script

```bash
bash train.sh
```

### Resume Training

```bash
python train.py \
    --resume ./logs/cxr2/diffusivity/seed_0/checkpoint_latest.pth \
    # ... other arguments
```

## Architecture

The model consists of:
- **Shared Backbone**: ResNet50 encoder (pretrained on ImageNet)
- **Task-Specific Heads**: 
  - Target task head: Binary/multi-class classification
  - Helper task heads: Task-specific output dimensions

## Data Format

The dataset expects:
- DICOM images in the specified data directory
- File markers in `../GazeMTL/file_markers/{source}/`
- Gaze data in `../GazeMTL/gaze_data/`

## Output

Training produces:
- `checkpoint_latest.pth`: Latest model checkpoint
- `checkpoint_best.pth`: Best model based on validation metric
- `history.json`: Training history with all metrics
- TensorBoard logs: All metrics logged for visualization
- Console output with progress bars and metrics

## TensorBoard Visualization

The training script automatically logs all metrics to TensorBoard. To visualize:

```bash
# Start TensorBoard (from the log directory or parent directory)
tensorboard --logdir=logs/gaze_mtl/cxr2/diffusivity/seed_0

# Or from the project root
tensorboard --logdir=src/externals/GazeMTL_pytorch/logs

# Then open http://localhost:6006 in your browser
```

TensorBoard will show:
- **Train/** - Training metrics (loss, accuracy, AUC for all tasks)
- **Val/** - Validation metrics
- **Test/** - Test metrics (after training completes)
- **Learning_Rate** - Learning rate schedule over epochs

## Differences from Emmental Version

1. **No Emmental**: Pure PyTorch implementation
2. **Modern Metrics**: Uses torchmetrics instead of custom metrics
3. **Simpler API**: More straightforward training loop
4. **Better Debugging**: Easier to debug and modify
5. **Latest Libraries**: Uses most recent PyTorch features

## Configuration

Key parameters:
- `--source`: Dataset source (cxr, cxr2, mets)
- `--gaze_mtl_task`: Helper task(s), e.g., "diffusivity", "loc", "time", or "loc_time"
- `--task_weights`: Comma-separated weights for tasks (target,helper1,helper2,...)
- `--lr_scheduler`: Learning rate scheduler (cosine_annealing, step, plateau, none)

## Requirements

- Python 3.8+
- PyTorch 2.1+
- CUDA-capable GPU (recommended)

## License

Same as the original GazeMTL project.

