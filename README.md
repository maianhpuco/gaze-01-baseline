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

```
python main_train_gnn.py --config configs/train_gazegnn.yaml
```

What’s included
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

Key files created or updated
- New: `main_train_gnn.py`
- New: `configs/train_gazegnn.yaml`

If you want me to add weighted sampler support, early stopping by `patience`, or to filter out ambiguous labels ahead of time (instead of remapping to 0), I can extend `main_train_gnn.py` accordingly. 