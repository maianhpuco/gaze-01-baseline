# gaze-01-baseline


Add external code: git add submodule: 
```bash  
git submodule add https://github.com/ukaukaaaa/GazeGNN externals/GazeGNN
```
```bash 
git submodule add https://github.com/cxr-eye-gaze/eye-gaze-dataset.git datasets/eye-gaze-dataset
```

To see example of the dataset :

```
make sanity_check_dataset 
```

EGDCXRDataset ready: 780 cases | classes=['CHF', 'pneumonia', 'Normal'] | class_counts=[251.0, 266.0, 263.0] | regions=4 | bbox_classes=17
EGDCXRDataset ready: 195 cases | classes=['CHF', 'pneumonia', 'Normal'] | class_counts=[72.0, 61.0, 62.0] | regions=4 | bbox_classes=17
EGDCXRDataset ready: 108 cases | classes=['CHF', 'pneumonia', 'Normal'] | class_counts=[40.0, 33.0, 35.0] | regions=4 | bbox_classes=17

## Training Models

Three training scripts are available, all following the same structure and using the same dataset splits:

```bash
# GazeGNN model (Graph Neural Network with gaze)
python main_train_gnn.py --config configs/train_gazegnn.yaml

# UNet model (Segmentation + Classification)
python main_train_unet.py --config configs/train_unet.yaml

# Temporal RNN model (Sequential gaze processing)
python main_train_temporal.py --config configs/train_temporal.yaml
```

What's included
- `main_train_gnn.py`
  - Loads splits from `split_files.dir`.
  - Builds `EGDCXRDataset` for train/val/test with your classes and `max_fixations`.
  - Adapts each sample to GazeGNN format:
    - image: `[3,224,224]` with ImageNet normalization (from dataset’s grayscale).
    - label: integer class index (uses `labels.single_index`; ambiguous set to 0).
    - gaze: `[56,56]` heatmap derived from fixations+dwell like GazeGNN’s `read_data.py`.
  - Trains `pvig_ti_224_gelu` with AdamW.
  - Prints logs in your requested format:
    - TRAIN/VAL per epoch:
      - `[Epoch N] TRAIN  loss=...  acc=...  (cls=..., txt=0.0000) gate(μ=0.000,σ=0.000)`
      - `[Epoch N] VAL    loss=...  acc=...  macroAUC=...  perClassAUC=[...]`
    - TEST at end:
      - `[Test] LOSS=...  ACC=...  macroAUC=...  perClassAUC=[...]`
  - Saves best checkpoint to `output_path.checkpoint_dir/best_ckpt_name`.

- `configs/train_gazegnn.yaml`
  - Matches your example structure:
    - `input_path.*`, `split_files.dir`
    - `train` block (classes, batch_size, epochs, max_fixations, workers, lr, weight_decay, label_smoothing)
    - `output_path` for checkpoints
    - `options` hints for model/backbone
  - Update paths to your environment if needed.

Notes
- The “(cls=..., txt=..., gate(...))” fields are formatted as requested. Since the core GazeGNN model doesn’t expose those internal components via the public API, `cls` mirrors CE loss, `txt` and `gate(μ,σ)` are placeholders (0.0). If you want real gate stats, we can instrument the model forward to return them.
- The adapter reproduces GazeGNN’s gaze preprocessing: accumulate dwell per pixel on a 224×224 grid, log+normalize, apply the same image transforms, then downsample to 56×56 via sliding window sum.

- `main_train_unet.py`
  - UNet encoder-decoder with dual outputs: segmentation masks (from fixation heatmaps) + classification
  - Uses same splits and logging format as GazeGNN
  - Configurable gamma for balancing segmentation vs classification loss
  - Supports both Dice loss and BCE for segmentation
  - Multi-label classification with BCEWithLogitsLoss

- `main_train_temporal.py`
  - Temporal RNN model processing sequential gaze heatmaps
  - Combines image CNN features with temporal gaze sequence features
  - Supports LSTM/GRU with attention mechanism
  - Handles variable-length fixation sequences with padding
  - Multi-label classification output

Key files created
- Training scripts: `main_train_gnn.py`, `main_train_unet.py`, `main_train_temporal.py`
- Configs: `configs/train_gazegnn.yaml`, `configs/train_unet.yaml`, `configs/train_temporal.yaml`

All three models:
- Use the same train/val/test splits from `split_files.dir`
- Output consistent logging format: `[Epoch N] TRAIN/VAL/TEST loss/acc/macroAUC/perClassAUC`
- Save best checkpoint based on validation macroAUC
- Support multi-GPU training with DataParallel 
