#!/usr/bin/env python3
"""
Evaluate Temporal model on test set using saved checkpoint.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from main_train import (
    Cfg, build_splits_ids, EGDCXRDataset, TemporalAdapter, 
    collate_temporal, compute_metrics
)
from torch.utils.data import DataLoader
from src.externals.eye_gaze_dataset.Experiments.models.eyegaze_model import EyegazeModel


@torch.no_grad()
def evaluate(model, loader, device, criterion, num_classes: int) -> Tuple[float, float, float, List[float]]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_probs = []
    all_labels = []
    
    for img, heatmaps, labels, lengths in loader:
        img = img.to(device)
        heatmaps = heatmaps.to(device)
        labels = labels.to(device)
        
        logits = model((img, heatmaps))
        loss = criterion(logits, labels)
        
        total_loss += float(loss.item()) * labels.size(0)
        total_count += int(labels.size(0))
        
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    acc, macro_auc, per_auc = compute_metrics(all_probs, all_labels, num_classes)
    
    return total_loss / total_count, acc, macro_auc, per_auc


def main():
    print("=" * 80)
    print("TEMPORAL MODEL TEST SET EVALUATION")
    print("=" * 80)
    
    # Load config
    cfg = Cfg(ROOT / "configs/train_temporal.yaml")
    
    # Paths
    root = Path(cfg.get("input_path", "gaze_raw"))
    seg = Path(cfg.get("input_path", "segmentation_dir"))
    transcripts = Path(cfg.get("input_path", "transcripts_dir", default=seg))
    dicom_root_cfg = cfg.get("input_path", "dicom_raw")
    dicom_root = Path(dicom_root_cfg) if dicom_root_cfg else None
    
    classes = cfg.get("train", "classes", default=["CHF", "pneumonia", "Normal"])
    batch_size = int(cfg.get("train", "batch_size", default=32))
    max_fix = int(cfg.get("train", "max_fixations", default=30))
    num_workers = 0  # Use 0 for evaluation
    
    # Model params
    model_type = cfg.get("train", "model_type", default="temporal")
    dropout = float(cfg.get("train", "dropout", default=0.5))
    hidden_dim = int(cfg.get("train", "hidden_dim", default=64))
    emb_dim = int(cfg.get("train", "emb_dim", default=64))
    hidden_hm = cfg.get("train", "hidden_hm", default=[256, 128])
    num_layers_hm = int(cfg.get("train", "num_layers_hm", default=1))
    cell = cfg.get("train", "cell", default="lstm")
    brnn_hm = bool(cfg.get("train", "brnn_hm", default=True))
    attention = bool(cfg.get("train", "attention", default=True))
    
    # Checkpoint
    checkpoint_path = ROOT / "runs/checkpoints/temporal/temporal_best.pt"
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Expected: Epoch 2, Val macroAUC = 0.767")
    print("=" * 80)
    print()
    
    # Get test split
    _, _, test_ids = build_splits_ids(cfg)
    
    # Build test dataset
    dtest_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts,
                                dicom_root=dicom_root, max_fixations=max_fix,
                                case_ids=test_ids, classes=classes)
    test_ds = TemporalAdapter(dtest_base, mode="test", max_temporal_frames=max_fix)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             collate_fn=collate_temporal, drop_last=False)
    
    print(f"Test set: {len(test_ds)} samples")
    print()
    
    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(classes)
    
    model = EyegazeModel(
        model_type=model_type,
        num_classes=n_classes,
        dropout=dropout,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        hidden_hm=hidden_hm,
        attention=attention,
        cell=cell,
        brnn_hm=brnn_hm,
        num_layers_hm=num_layers_hm
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✓ Checkpoint loaded successfully")
    print()
    
    # Evaluate
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    print("Evaluating on test set...")
    test_loss, test_acc, test_macroauc, test_perauc = evaluate(
        model, test_loader, device, criterion, n_classes
    )
    
    print("=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    print(f"Loss:              {test_loss:.4f}")
    print(f"Accuracy:          {test_acc:.3f}")
    print(f"Macro AUC:         {test_macroauc:.3f}")
    print(f"Per-Class AUC:     {[round(x, 3) for x in test_perauc]}")
    print(f"  - CHF:           {test_perauc[0]:.3f}")
    print(f"  - Pneumonia:     {test_perauc[1]:.3f}")
    print(f"  - Normal:        {test_perauc[2]:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

