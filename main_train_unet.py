#!/usr/bin/env python3
"""
UNet Classifier Training - Adapted from eye-gaze-dataset/Experiments/main_Unet.py
Reproduces the exact training steps with EGDCXRDataset
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from pathlib import Path
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from imgaug import augmenters as iaa
from PIL import Image
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import gaze_utils after sys.path is set
from gaze_utils.heatmap_utils import create_static_heatmap

# Import dataset
try:
    from egd_cxr_dataset.datasets.edg_cxr import EGDCXRDataset
except ModuleNotFoundError:
    DATASETS_DIR = SRC / "egd_cxr_dataset" / "datasets"
    if str(DATASETS_DIR) not in sys.path:
        sys.path.insert(0, str(DATASETS_DIR))
    from edg_cxr import EGDCXRDataset

# Import UNet utilities from externals
sys.path.insert(0, str(SRC / "externals/eye-gaze-dataset/Experiments"))
from models.classifier import UnetClassifier
from utils.utils import cyclical_lr
from utils.dice_loss import GeneralizedDiceLoss
from utils.visualization import plot_roc_curve

plt.rcParams['figure.figsize'] = [25, 10]

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger('unet')
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('PIL').setLevel(logging.INFO)


class UnetDatasetAdapter(Dataset):
    """Adapt EGDCXRDataset to UNet format with static heatmaps"""

    def __init__(
        self,
        base: EGDCXRDataset,
        image_transform,
        heatmap_static_transform,
        heatmaps_threshold=None,
        *,
        cache_dir: Path | None = None,
        split: str = "train",
    ):
        self.base = base
        self.image_transform = image_transform
        self.heatmap_static_transform = heatmap_static_transform
        self.heatmaps_threshold = heatmaps_threshold
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]

        img1 = sample["image"]
        img3 = img1.repeat(3, 1, 1)
        pil_img = transforms.ToPILImage()(img3)
        img_tensor = self.image_transform(pil_img)

        fx = sample["fixations"]
        xy = fx["xy"]
        dwell = fx["dwell"]
        dicom_id = sample["dicom_id"]

        static_np = None
        if self.cache_dir:
            static_path = self.cache_dir / "static" / self.split / f"{dicom_id}.npy"
            if static_path.exists():
                static_np = np.load(static_path)

        if static_np is None:
            static_np = create_static_heatmap(xy, dwell)

        if self.heatmaps_threshold is not None:
            static_np = (static_np >= self.heatmaps_threshold).astype(np.float32)

        static_heatmap_pil = Image.fromarray((static_np * 255).astype(np.uint8))
        static_tensor = self.heatmap_static_transform(static_heatmap_pil)  # [1, 224, 224]

        y_idx = int(sample["labels"]["single_index"].item())
        if y_idx < 0:
            y_idx = 0

        # Return class index (for CrossEntropyLoss) instead of one-hot (for BCEWithLogitsLoss)
        label = torch.tensor(y_idx, dtype=torch.long)

        return img_tensor, label, idx, static_tensor, static_tensor  # X_hm, y_hm


def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch Eye Gaze UNet')
    
    # Data
    parser.add_argument('--config', type=str, default='configs/train_gazegnn.yaml', help='Config path')
    parser.add_argument('--output_dir', type=str, default='results/unet', help='Output directory')
    parser.add_argument('--class_names', type=list, default=['CHF', 'pneumonia', 'Normal'], help='Label names')
    parser.add_argument('--num_workers', type=int, default=None, help='number of workers')
    parser.add_argument('--resize', type=int, default=None, help='Resizing images')
    parser.add_argument('--cache_dir', type=str, default='cache/heatmaps', help='Directory to read/write cached heatmaps')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate (default: from config or 3e-4)')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay (default: from config or 1e-4)')
    parser.add_argument('--scheduler', action='store_true', help='[USE] scheduler (cyclical LR, default: True)')
    parser.add_argument('--no_scheduler', dest='scheduler', action='store_false', help='Disable scheduler (use ReduceLROnPlateau instead)')
    parser.add_argument('--step_size', type=int, default=5, help='scheduler step size')
    
    # UNET Specific
    parser.add_argument('--model_type', default='unet', choices=['baseline', 'unet'], help='baseline or unet')
    parser.add_argument('--heatmaps_threshold', type=float, default=None, help='threshold for heatmap')
    parser.add_argument('--gamma', type=float, default=1.0, help='weight between classifier and segmentation')
    parser.add_argument('--model_teacher', type=str, default='timm-efficientnet-b0', help='encoder model')
    parser.add_argument('--pretrained_name', type=str, default='noisy-student', help='pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--second_loss', type=str, default='ce', choices=['dice', 'ce'], help='Segmentation loss')
    
    # Misc
    parser.add_argument('--gpus', type=str, default='0,1', help='Which gpus to use')
    parser.add_argument('--viz', default=False, action='store_true', help='[USE] Tensorboard')
    parser.add_argument('--test', default=False, action='store_true', help='[USE] flag for testing only')
    parser.add_argument('--testdir', type=str, default=None, help='model to test')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    return parser


def load_data(args):
    """Load data using EGDCXRDataset"""
    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Paths
    root = Path(cfg['input_path']['gaze_raw'])
    seg = Path(cfg['input_path']['segmentation_dir'])
    transcripts = Path(cfg['input_path'].get('transcripts_dir', seg))
    dicom_root = Path(cfg['input_path']['dicom_raw'])
    
    # Training configuration overrides
    train_cfg = cfg.get('train', {}) if isinstance(cfg, dict) else {}
    if args.batch_size is None:
        args.batch_size = int(train_cfg.get('batch_size', 32))
    if args.num_workers is None:
        args.num_workers = int(train_cfg.get('num_workers', 16))
    if args.epochs is None:
        args.epochs = int(train_cfg.get('epochs', 10))
    if args.weight_decay is None:
        args.weight_decay = float(train_cfg.get('weight_decay', 1e-4))
    if args.resize is None:
        args.resize = int(train_cfg.get('resize', 224))
    
    args.gamma = float(train_cfg.get('gamma', args.gamma))
    if 'scheduler' in train_cfg:
        args.scheduler = bool(train_cfg.get('scheduler'))
    else:
        args.scheduler = True
    if args.heatmaps_threshold is None:
        args.heatmaps_threshold = train_cfg.get('heatmaps_threshold', args.heatmaps_threshold)
    if args.heatmaps_threshold is not None:
        args.heatmaps_threshold = float(args.heatmaps_threshold)
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    
    # Splits
    split_cfg = cfg.get('split_files', {}) if cfg else {}
    default_split_dir = ROOT / "configs" / "splits" / "fold1"
    split_dir_path = split_cfg.get('dir', default_split_dir)
    split_dir = Path(split_dir_path)
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    if not split_dir.exists():
        logger.warning(f"Split directory {split_dir} not found. Falling back to {default_split_dir}.")
        split_dir = default_split_dir
    
    def read_ids(name):
        file_path = split_dir / f"{name}_ids.txt"
        if not file_path.exists():
            raise FileNotFoundError(f"Expected split file not found: {file_path}")
        return [
            line.strip()
            for line in file_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    
    train_ids = read_ids("train")
    val_ids = read_ids("val")
    test_ids = read_ids("test")
    
    # Classes
    class_names = train_cfg.get('classes', args.class_names)
    if not class_names:
        class_names = args.class_names
    args.class_names = class_names
    
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    seq = iaa.Sequential([iaa.Resize((args.resize, args.resize))])

    def _apply_seq(img):
        arr = np.array(img)
        arr = seq(image=arr)
        return Image.fromarray(arr)

    image_transform = transforms.Compose([
        transforms.Lambda(_apply_seq),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    heatmap_static_transform = transforms.Compose([
        transforms.Resize([args.resize, args.resize]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    
    # Create base datasets
    max_fix = int(train_cfg.get('max_fixations', 100))
    
    train_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts,
                               dicom_root=dicom_root, max_fixations=max_fix,
                               case_ids=train_ids, classes=args.class_names)
    valid_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts,
                               dicom_root=dicom_root, max_fixations=max_fix,
                               case_ids=val_ids, classes=args.class_names)
    test_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts,
                              dicom_root=dicom_root, max_fixations=max_fix,
                              case_ids=test_ids, classes=args.class_names)
    
    # Wrap with adapter
    train_dataset = UnetDatasetAdapter(train_base, image_transform, heatmap_static_transform,
                                       args.heatmaps_threshold, cache_dir=cache_dir, split="train")
    valid_dataset = UnetDatasetAdapter(valid_base, image_transform, heatmap_static_transform,
                                       args.heatmaps_threshold, cache_dir=cache_dir, split="val")
    test_dataset = UnetDatasetAdapter(test_base, image_transform, heatmap_static_transform,
                                      args.heatmaps_threshold, cache_dir=cache_dir, split="test")
    
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True, pin_memory=True)
    valid_dl = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, drop_last=True, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, drop_last=False, pin_memory=True)
    
    return train_dl, valid_dl, test_dl


def eval_net(model, loader, classifier_criterion, segment_criterion, model_type, gamma=0.5):
    """Evaluation function"""
    model.eval()
    tot, seg, clas = 0, 0, 0
    correct = 0
    total = 0
    counter = 0
    for images, labels, idx, X_hm, y_hm in loader:
        images = images.cuda()
        labels = labels.cuda()
        if not model_type == 'baseline':
            y_hm = y_hm.cuda()
        l_segment = 0.
        with torch.no_grad():
            masks_pred, y_pred = model(images)
            if not model_type == 'baseline':
                l_segment = segment_criterion(masks_pred, y_hm)
            # Convert one-hot to class indices if needed (for backward compatibility)
            labels_cls = labels
            if labels.dim() > 1 and labels.size(1) > 1:
                labels_cls = labels.argmax(dim=1)
            l_classifier = classifier_criterion(y_pred, labels_cls)
            tot += ((gamma * l_classifier) + ((1 - gamma) * l_segment)).item()
            seg += l_segment if isinstance(l_segment, (int, float)) else l_segment.item()
            clas += l_classifier.item()
            # Calculate accuracy
            _, predicted = torch.max(y_pred.data, 1)
            total += labels_cls.size(0)
            correct += (predicted == labels_cls).sum().item()
            counter += 1
    model.train()
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return tot/counter, seg/counter, clas/counter, accuracy


def train_unet(args, model, train_dl, valid_dl, output_model_path, comment):
    """Training loop"""
    if args.viz:
        writer = SummaryWriter(comment=comment)
    
    global_step = 0
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler:
        clr = cyclical_lr(step_sz=args.step_size, min_lr=args.lr, max_lr=1, mode='triangular2')
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, [clr])
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    classifier_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    segment_criterion = GeneralizedDiceLoss() if args.second_loss == 'dice' else nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 10  # Early stopping patience (reduced to prevent overfitting)

    for epoch in range(args.epochs):
        model.train()
        counter = 0
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_seg_loss = 0
        
        for images, labels, idx, X_hm, y_hm in train_dl:
            images = images.cuda()
            labels = labels.cuda()
            if not args.model_type == 'baseline':
                y_hm = y_hm.cuda()
            
            masks_pred, y_pred = model(images)
            
            if not args.model_type == 'baseline':
                loss_segment = segment_criterion(masks_pred, y_hm)
            else:
                loss_segment = 0
            
            # Convert one-hot to class indices if needed (for backward compatibility)
            if labels.dim() > 1 and labels.size(1) > 1:
                labels = labels.argmax(dim=1)
            loss_classifier = classifier_criterion(y_pred, labels)
            total_loss = (args.gamma * loss_classifier) + ((1 - args.gamma) * loss_segment)
            epoch_loss += total_loss.item()
            epoch_cls_loss += loss_classifier.item()
            if isinstance(loss_segment, torch.Tensor):
                epoch_seg_loss += loss_segment.item()
            
            if args.viz:
                writer.add_scalar('Classifier_Loss', loss_classifier.item(), global_step)
                writer.add_scalar('Loss/Train', total_loss.item(), global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                if not args.model_type == 'baseline':
                    writer.add_scalar('Mask_Loss', loss_segment.item(), global_step)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
            optimizer.step()
            global_step += 1
            counter += 1
        
        avg_loss = epoch_loss / counter
        avg_cls = epoch_cls_loss / counter
        avg_seg = epoch_seg_loss / counter if epoch_seg_loss > 0 else 0
        logger.info(f'[Epoch {epoch+1}] TRAIN loss={avg_loss:.4f} cls_loss={avg_cls:.4f} seg_loss={avg_seg:.4f}')
        
        with torch.no_grad():
            val_loss, val_segment, val_classifier, val_acc = eval_net(model, valid_dl, classifier_criterion,
                                                                       segment_criterion, args.model_type, args.gamma)
        
        logger.info(f'[Epoch {epoch+1}] VAL loss={val_loss:.4f} cls_loss={val_classifier:.4f} seg_loss={val_segment:.4f} acc={val_acc:.2f}%')
        
        if args.viz:
            writer.add_scalar('Validation_Loss', val_loss, global_step)
            writer.add_scalar('Validation_Accuracy', val_acc, global_step)
            if not args.model_type == 'baseline':
                writer.add_scalar('Validation_Mask_Loss', val_segment, global_step)
            writer.add_scalar('Validation_Classifier_Loss', val_classifier, global_step)
            writer.add_images('images', images, global_step)
            if args.model_type == 'unet':
                writer.add_images('masks/true', y_hm, global_step)
                writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        
        model.train()
        if args.scheduler and isinstance(scheduler, optim.lr_scheduler.LambdaLR):
            scheduler.step()
        elif not args.scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        
        try:
            os.makedirs(output_model_path)
            logger.info('Created Checkpoint directory')
        except OSError:
            pass
        
        ckpt_path = os.path.join(output_model_path, f"Epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Checkpoint {epoch + 1} saved!")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_path = os.path.join(output_model_path, "best_weights.pth")
            torch.save(model.state_dict(), best_path)
            logger.info(f"Best model updated at epoch {best_epoch} with val_acc={best_val_acc:.2f}% val_loss={best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
            logger.info(f'Best model: epoch {best_epoch}, val_acc={best_val_acc:.2f}% val_loss={best_val_loss:.4f}')
            break
    
    if args.viz:
        writer.close()
    if best_epoch > 0 and best_val_acc > 0:
        logger.info(f"Best validation accuracy {best_val_acc:.2f}% (loss={best_val_loss:.4f}) achieved at epoch {best_epoch}")
    else:
        logger.info("No improvement recorded during training (no epochs run).")


def eval_eyegaze_network(model, test_dl, class_names, model_dir, model_name, model_type,
                         gamma=1.0, second_loss='ce', plot_data=True):
    """Evaluation with extended metrics for classifier and segmentation outputs."""
    model.eval()
    if len(test_dl) == 0:
        logger.warning("Received empty dataloader in eval_eyegaze_network.")
        return {}

    classifier_criterion = nn.CrossEntropyLoss(reduction='mean')
    if model_type == 'baseline':
        segment_criterion = None
    else:
        segment_criterion = GeneralizedDiceLoss() if second_loss == 'dice' else nn.BCEWithLogitsLoss(reduction='mean')

    labels_all = []
    probs_all = []
    losses = []
    seg_ious = []

    with torch.no_grad():
        for images, y_batch, idx, X_hm, y_hm in test_dl:
            images = images.cuda()
            y_batch = y_batch.cuda()
            masks_pred, logits = model(images)

            # Convert class indices to one-hot for loss computation if needed
            y_batch_cls = y_batch
            if y_batch.dim() == 1:  # class indices
                y_batch_onehot = torch.zeros(y_batch.size(0), logits.size(1), device=y_batch.device)
                y_batch_onehot.scatter_(1, y_batch.unsqueeze(1), 1.0)
                cls_loss = classifier_criterion(logits, y_batch_cls)
                # Use softmax for multi-class (not sigmoid)
                probs = torch.softmax(logits, dim=1)
                labels_all.append(y_batch_onehot.cpu().numpy())
            else:  # one-hot format (backward compatibility)
                y_batch_cls = y_batch.argmax(dim=1)
                cls_loss = classifier_criterion(logits, y_batch_cls)
                probs = torch.sigmoid(logits)
                labels_all.append(y_batch.cpu().numpy())
            
            if model_type == 'baseline':
                seg_loss_val = torch.tensor(0.0, device=logits.device)
            else:
                y_hm = y_hm.cuda()
                seg_loss_val = segment_criterion(masks_pred, y_hm.float())

            total_loss = gamma * cls_loss + (1 - gamma) * seg_loss_val
            losses.append(float(total_loss.item()))
            probs_all.append(probs.cpu().numpy())

            if model_type != 'baseline':
                mask_prob = torch.sigmoid(masks_pred)
                pred_mask = (mask_prob > 0.5).cpu().numpy()
                true_mask = (y_hm > 0.5).float().cpu().numpy()
                for pmask, tmask in zip(pred_mask, true_mask):
                    intersection = np.logical_and(pmask > 0.5, tmask > 0.5).sum()
                    union = np.logical_or(pmask > 0.5, tmask > 0.5).sum()
                    if union == 0:
                        seg_ious.append(1.0)
                    else:
                        seg_ious.append(intersection / union)

    y_true = np.vstack(labels_all)
    y_prob = np.vstack(probs_all)
    y_pred = (y_prob >= 0.5).astype(np.float32)

    logger.info('--' * 60)

    write_dir = os.path.join(model_dir, 'plots')
    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)
    test_log_path = os.path.join(write_dir, os.path.splitext(model_name)[0] + ".log")
    logger.info(f"** write log to {test_log_path} **")

    aurocs = []
    fpr = dict()
    tpr = dict()
    per_class_precision = []
    per_class_recall = []
    per_class_iou = []

    with open(test_log_path, "w") as f:
        for idx_cls, cls_name in enumerate(class_names):
            y_cls = y_true[:, idx_cls]
            y_prob_cls = y_prob[:, idx_cls]
            y_pred_cls = y_pred[:, idx_cls]
            try:
                auc_score = roc_auc_score(y_cls, y_prob_cls)
            except ValueError:
                auc_score = float("nan")
            aurocs.append(auc_score)
            try:
                fpr[idx_cls], tpr[idx_cls], _ = roc_curve(y_cls, y_prob_cls)
            except ValueError:
                fpr[idx_cls], tpr[idx_cls] = None, None

            tp = np.sum((y_cls == 1) & (y_pred_cls == 1))
            fp = np.sum((y_cls == 0) & (y_pred_cls == 1))
            fn = np.sum((y_cls == 1) & (y_pred_cls == 0))
            precision_cls = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            recall_cls = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            iou_cls = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")

            per_class_precision.append(precision_cls)
            per_class_recall.append(recall_cls)
            per_class_iou.append(iou_cls)

            f.write(f"{cls_name} AUC: {auc_score}\n")
            f.write(f"{cls_name} Precision: {precision_cls}\n")
            f.write(f"{cls_name} Recall: {recall_cls}\n")
            f.write(f"{cls_name} IoU: {iou_cls}\n")

            logger.info(f"{cls_name} AUC: {auc_score}")
            logger.info(f"{cls_name} Precision: {precision_cls}")
            logger.info(f"{cls_name} Recall: {recall_cls}")
            logger.info(f"{cls_name} IoU: {iou_cls}")

        mean_auroc = float(np.nanmean(aurocs))
        mean_precision = float(np.nanmean(per_class_precision))
        mean_recall = float(np.nanmean(per_class_recall))
        mean_iou = float(np.nanmean(per_class_iou))
        subset_acc = float(np.mean(np.all(y_pred == y_true, axis=1)))
        
        # Add multi-class accuracy using argmax (for single-label classification)
        y_true_class = np.argmax(y_true, axis=1)  # Get true class index
        y_pred_class = np.argmax(y_prob, axis=1)  # Get predicted class index from probabilities
        multiclass_acc = float(np.mean(y_pred_class == y_true_class))
        
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        seg_iou_mean = float(np.mean(seg_ious)) if seg_ious else float("nan")

        f.write("-------------------------\n")
        f.write(f"mean auroc: {mean_auroc}\n")
        f.write(f"macro precision: {mean_precision}\n")
        f.write(f"macro recall: {mean_recall}\n")
        f.write(f"macro IoU (classification): {mean_iou}\n")
        f.write(f"subset accuracy: {subset_acc}\n")
        f.write(f"multiclass accuracy: {multiclass_acc}\n")
        f.write(f"loss: {avg_loss}\n")
        f.write(f"segmentation IoU: {seg_iou_mean}\n")

        logger.info(f"mean auroc: {mean_auroc}")
        logger.info(f"macro precision: {mean_precision}")
        logger.info(f"macro recall: {mean_recall}")
        logger.info(f"macro IoU (classification): {mean_iou}")
        logger.info(f"subset accuracy: {subset_acc}")
        logger.info(f"multiclass accuracy: {multiclass_acc}")
        logger.info(f"loss: {avg_loss}")
        logger.info(f"segmentation IoU: {seg_iou_mean}")

    if plot_data:
        valid_fpr = {k: v for k, v in fpr.items() if v is not None}
        valid_tpr = {k: v for k, v in tpr.items() if v is not None}
        if valid_fpr and valid_tpr:
            logger.info("** plot and save ROC curves **")
            name_variable = os.path.splitext(model_name)[0] + ".png"
            plot_roc_curve(valid_tpr, valid_fpr, class_names, aurocs, write_dir, name_variable)

    return {
        "macro_auc": mean_auroc,
        "per_class_auc": aurocs,
        "macro_precision": mean_precision,
        "per_class_precision": per_class_precision,
        "macro_recall": mean_recall,
        "per_class_recall": per_class_recall,
        "macro_iou": mean_iou,
        "per_class_iou": per_class_iou,
        "subset_accuracy": subset_acc,
        "multiclass_accuracy": multiclass_acc,
        "loss": avg_loss,
        "segmentation_iou": seg_iou_mean,
    }


if __name__ == '__main__':
    args = make_parser().parse_args()
    
    # Set seeds
    random.seed(args.rseed)
    np.random.seed(args.rseed)
    torch.manual_seed(args.rseed)
    
    cuda = torch.cuda.is_available() and args.gpus != '-1'
    
    if cuda:
        torch.cuda.manual_seed(args.rseed)
        torch.cuda.manual_seed_all(args.rseed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.cuda.set_device("cuda:" + args.gpus.split(',')[0])
    args.device = torch.device("cuda:" + args.gpus.split(',')[0]) if cuda else torch.device('cpu')
    
    logger.info(f"Using device: {args.device}")
    if cuda:
        logger.info(torch.cuda.get_device_name(args.device))
    
    if args.second_loss == 'ce':
        args.heatmaps_threshold = None
    
    # Create output path
    timestamp = str(datetime.now()).replace(" ", "").split('.')[0]
    comment_variable = f'{args.model_type}_bs{args.batch_size}_ep{args.epochs}_gamma{args.gamma}_{timestamp}'
    output_model_path = os.path.join(args.output_dir, comment_variable)
    
    # Load data (this sets scheduler default and other config values)
    train_dl, valid_dl, test_dl = load_data(args)
    
    # Log arguments AFTER load_data so scheduler default is applied
    logger.info(f"[Arguments]: {args}")
    
    n_classes = len(args.class_names)
    n_segments = 1
    aux_params = dict(
        pooling='avg',
        dropout=args.dropout,
        activation=None,
        classes=n_classes,
    )
    
    if not args.test:  # Training
        logger.info('---- TRAINING ----')
        if args.model_type == 'baseline':
            model = UnetClassifier(encoder_name=args.model_teacher, classes=n_segments,
                                   encoder_weights=args.pretrained_name, aux_params=aux_params).to(device=args.device)
        else:
            model = smp.Unet(args.model_teacher, classes=n_segments, encoder_weights=args.pretrained_name,
                           aux_params=aux_params).to(device=args.device)
        
        total_params_net = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        logger.info(f'Number of parameters: {total_params_net}')
        
        if len(args.gpus.split(',')) > 1:
            logger.info(f"Using {len(args.gpus.split(','))} GPUs!")
            device_ids = [int(i) for i in args.gpus.split(',')]
            model = nn.DataParallel(model, device_ids=device_ids)
        
        train_unet(args, model, train_dl, valid_dl, output_model_path, comment_variable)
    
    # Testing
    logger.info('---- TESTING ----')
    best_metrics = None
    best_model_name = ''
    model_dir = args.testdir if args.testdir else output_model_path

    candidate_models = []
    best_weights_path = os.path.join(model_dir, "best_weights.pth")
    if os.path.exists(best_weights_path):
        candidate_models.append("best_weights.pth")
    else:
        candidate_models.extend([f"Epoch_{i+1}.pth" for i in range(args.epochs)])

    for model_name in candidate_models:
        if args.model_type == 'baseline':
            model = UnetClassifier(encoder_name=args.model_teacher, classes=n_segments,
                                   encoder_weights=args.pretrained_name, aux_params=aux_params).to(device=args.device)
        else:
            model = smp.Unet(args.model_teacher, classes=n_segments, encoder_weights=args.pretrained_name,
                           aux_params=aux_params).to(device=args.device)

        output_weights_name = os.path.join(model_dir, model_name)
        if not os.path.exists(output_weights_name):
            logger.info(f'Skipping missing model: {output_weights_name}')
            continue

        logger.info(f'Loading Model: {output_weights_name}')
        
        if len(args.gpus.split(',')) > 1:
            device_ids = [int(i) for i in args.gpus.split(',')]
            model = nn.DataParallel(model, device_ids=device_ids)

        state_dict = torch.load(
            output_weights_name,
            map_location=args.device
        )
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

        metrics = eval_eyegaze_network(
            model, test_dl, args.class_names, model_dir, model_name,
            args.model_type, args.gamma, args.second_loss, plot_data=True
        )

        if metrics:
            logger.info(
                f"{model_name} metrics: AUC={metrics.get('macro_auc')}, "
                f"Precision={metrics.get('macro_precision')}, "
                f"Recall={metrics.get('macro_recall')}, "
                f"Accuracy={metrics.get('subset_accuracy')}, "
                f"Loss={metrics.get('loss')}, "
                f"Cls IoU={metrics.get('macro_iou')}, "
                f"Seg IoU={metrics.get('segmentation_iou')}"
            )

        macro_auc = metrics.get('macro_auc', 0.0) if metrics else 0.0
        if best_metrics is None or macro_auc >= best_metrics.get('macro_auc', -np.inf):
            best_model_name = model_name
            best_metrics = metrics

    if best_metrics:
        logger.info(
            f"[TEST] Best model: {best_model_name} | "
            f"AUC={best_metrics.get('macro_auc')}, "
            f"Precision={best_metrics.get('macro_precision')}, "
            f"Recall={best_metrics.get('macro_recall')}, "
            f"Accuracy={best_metrics.get('subset_accuracy')}, "
            f"Loss={best_metrics.get('loss')}, "
            f"Cls IoU={best_metrics.get('macro_iou')}, "
            f"Seg IoU={best_metrics.get('segmentation_iou')}"
        )
        
        logger.info("[FINAL TEST] Per-class metrics:")
        per_class_auc = best_metrics.get("per_class_auc", [])
        per_class_precision = best_metrics.get("per_class_precision", [])
        per_class_recall = best_metrics.get("per_class_recall", [])
        per_class_iou = best_metrics.get("per_class_iou", [])
        
        for i, cls_name in enumerate(args.class_names):
            if i < len(per_class_auc) and i < len(per_class_precision) and i < len(per_class_recall) and i < len(per_class_iou):
                logger.info(f"  {cls_name}: AUC={per_class_auc[i]:.4f}, Precision={per_class_precision[i]:.4f}, "
                           f"Recall={per_class_recall[i]:.4f}, IoU={per_class_iou[i]:.4f}")
    else:
        logger.info("[TEST] No valid UNet checkpoints evaluated.")

