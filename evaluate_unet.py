#!/usr/bin/env python3
"""
Evaluate UNet model on test set using saved checkpoint.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
import segmentation_models_pytorch as smp

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from main_train_unet import (
    Cfg, build_splits_ids, EGDCXRDataset, UNetAdapter, compute_metrics,
    UnetClassifier
)
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model, loader, device, classifier_criterion, segment_criterion, 
             gamma: float, num_classes: int) -> Tuple[float, float, float, List[float]]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_probs = []
    all_labels = []
    
    for img, labels, heatmap in loader:
        img = img.to(device)
        labels = labels.to(device)
        heatmap = heatmap.to(device)
        
        masks_pred, logits = model(img)
        
        loss_classifier = classifier_criterion(logits, labels)
        loss_segment = segment_criterion(masks_pred, heatmap)
        loss = (gamma * loss_classifier) + ((1 - gamma) * loss_segment)
        
        total_loss += float(loss.item()) * labels.size(0)
        total += int(labels.size(0))
        
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    acc, macro_auc, per_auc = compute_metrics(all_probs, all_labels, num_classes)
    
    return total_loss / total, acc, macro_auc, per_auc


def main():
    print("=" * 80)
    print("UNET MODEL TEST SET EVALUATION")
    print("=" * 80)
    
    # Load config
    cfg = Cfg(ROOT / "configs/train_unet.yaml")
    
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
    model_type = cfg.get("train", "model_type", default="unet")
    encoder_name = cfg.get("train", "encoder_name", default="resnet34")
    dropout = float(cfg.get("train", "dropout", default=0.5))
    gamma = float(cfg.get("train", "gamma", default=0.5))
    second_loss = cfg.get("train", "second_loss", default="dice")
    
    # Checkpoint
    checkpoint_path = ROOT / "runs/checkpoints/unet/unet_best.pt"
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Expected: Epoch 24, Val macroAUC = 0.858")
    print("=" * 80)
    print()
    
    # Get test split
    _, _, test_ids = build_splits_ids(cfg)
    
    # Build test dataset
    dtest_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts,
                                dicom_root=dicom_root, max_fixations=max_fix,
                                case_ids=test_ids, classes=classes)
    test_ds = UNetAdapter(dtest_base, mode="test")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    
    print(f"Test set: {len(test_ds)} samples")
    print()
    
    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(classes)
    n_segments = 1  # heatmap segmentation
    aux_params = dict(
        pooling='avg',
        dropout=dropout,
        activation=None,
        classes=n_classes,
    )
    
    if model_type == 'baseline':
        model = UnetClassifier(encoder_name=encoder_name, classes=n_segments,
                              encoder_weights='noisy-student', aux_params=aux_params)
    else:
        model = smp.Unet(encoder_name, classes=n_segments, encoder_weights='noisy-student',
                        aux_params=aux_params)
    
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✓ Checkpoint loaded successfully")
    print()
    
    # Evaluate
    classifier_criterion = nn.BCEWithLogitsLoss().to(device)
    if second_loss == "dice":
        segment_criterion = smp.losses.DiceLoss(mode='binary', from_logits=True).to(device)
    else:
        segment_criterion = nn.BCEWithLogitsLoss().to(device)
    
    print("Evaluating on test set...")
    test_loss, test_acc, test_macroauc, test_perauc = evaluate(
        model, test_loader, device, classifier_criterion, 
        segment_criterion, gamma, n_classes
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

