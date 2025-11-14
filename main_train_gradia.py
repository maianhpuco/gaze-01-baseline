#!/usr/bin/env python3
"""
GRADIA Training - Adapted from externals/GRADIA/GRADIA.py
Reproduces the exact training steps with EGDCXRDataset
Two-phase training: Phase 1 baseline, Phase 2 with attention supervision
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
import cv2
import json
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from imgaug import augmenters as iaa
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import gaze_utils after sys.path is set
from gaze_utils.heatmap_utils import create_gradia_heatmap

# Import dataset
try:
    from egd_cxr_dataset.datasets.edg_cxr import EGDCXRDataset
except ModuleNotFoundError:
    DATASETS_DIR = SRC / "egd_cxr_dataset" / "datasets"
    if str(DATASETS_DIR) not in sys.path:
        sys.path.insert(0, str(DATASETS_DIR))
    from edg_cxr import EGDCXRDataset

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger('gradia')


# ======== GradCAM Implementation ========
class FeatureExtractor:
    """Extract activations and register gradients from target layers"""
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
    
    def save_gradient(self, grad):
        self.gradients.append(grad)
    
    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs:
    """Get network output, activations, and gradients"""
    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)
    
    def get_gradients(self):
        return self.feature_extractor.gradients
    
    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)
        return target_activations, x


class GradCam:
    """GradCAM implementation"""
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)
    
    def forward(self, input):
        return self.model(input)
    
    def get_attention_map(self, input, index=None):
        """Get attention map as tensor (for training)"""
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        
        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        grads_val = self.extractor.get_gradients()[-1]
        target = features[-1].squeeze()
        weights = torch.mean(grads_val, axis=(2, 3)).squeeze()
        
        if self.cuda:
            cam = torch.zeros(target.shape[1:]).cuda()
        else:
            cam = torch.zeros(target.shape[1:])
        
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        return cam, output


# ======== Dataset Adapter ========
class GRADIADatasetAdapter(Dataset):
    """Adapt EGDCXRDataset for GRADIA training"""

    def __init__(
        self,
        base: EGDCXRDataset,
        transform,
        return_heatmap=False,
        *,
        cache_dir: Path | None = None,
        split: str = "train",
    ):
        self.base = base
        self.transform = transform
        self.return_heatmap = return_heatmap
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]

        img1 = sample["image"]
        img3 = img1.repeat(3, 1, 1)
        pil_img = transforms.ToPILImage()(img3)
        img_tensor = self.transform(pil_img)

        y_idx = int(sample["labels"]["single_index"].item())
        if y_idx < 0:
            y_idx = 0

        if self.return_heatmap:
            fx = sample["fixations"]
            xy = fx["xy"]
            dwell = fx["dwell"]
            dicom_id = sample["dicom_id"]

            heatmap = None
            if self.cache_dir:
                cache_path = self.cache_dir / "gradia" / self.split / f"{dicom_id}.npy"
                if cache_path.exists():
                    heatmap = np.load(cache_path)

            if heatmap is None:
                heatmap = create_gradia_heatmap(xy, dwell)
                att_weight = torch.tensor(1.0 if xy.numel() > 0 else 0.0, dtype=torch.float32)
            else:
                att_weight = torch.tensor(1.0 if heatmap.max() > 0 else 0.0, dtype=torch.float32)

            pred_weight = torch.tensor(1.0, dtype=torch.float32)
            return (
                img_tensor,
                torch.tensor(y_idx, dtype=torch.long),
                torch.from_numpy(heatmap.astype(np.float32)),
                pred_weight,
                att_weight,
            )

        return img_tensor, torch.tensor(y_idx, dtype=torch.long)


# ======== Utility Functions ========
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes accuracy"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_iou(x, y):
    """Compute IoU between two binary masks"""
    intersection = np.bitwise_and(x, y)
    union = np.bitwise_or(x, y)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou


# ======== Training Functions ========
def model_train(model, train_loader, val_loader, n_epoch, use_cuda, model_dir, lr=3e-4, weight_decay=1e-4):
    """Phase 1: Standard training"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    best_val_acc = 0
    patience_counter = 0
    patience = 10
    
    for epoch in np.arange(n_epoch) + 1:
        model.train()
        st = time.time()
        train_losses = []
        
        if use_cuda:
            torch.cuda.empty_cache()
        
        outputs_all = []
        targets_all = []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
            optimizer.step()
            
            train_losses.append(loss.cpu().detach().item())
            outputs_all.append(outputs)
            targets_all.append(targets)
        
        et = time.time()
        train_time = et - st
        train_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach().item()
        
        # Validation
        logger.info('Start validation')
        model.eval()
        outputs_all = []
        targets_all = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                outputs = model(inputs)
                outputs_all.append(outputs)
                targets_all.append(targets)
        
        val_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach().item()
        
        # Update scheduler
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model, os.path.join(model_dir, 'model_phase1_best.pth'))
            logger.info('UPDATE BEST MODEL!')
        else:
            patience_counter += 1
        
        torch.save(model, os.path.join(model_dir, f'model_phase1_epoch{epoch}.pth'))
        logger.info(f'[Phase1 Epoch {epoch}] TRAIN time={train_time:.1f}s loss={np.mean(train_losses):.4f} acc={train_acc:.2f}% VAL acc={val_acc:.2f}% LR={optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f'Early stopping at epoch {epoch} (no improvement for {patience} epochs)')
            logger.info(f'Best validation accuracy: {best_val_acc:.2f}%')
            break
    
    return best_val_acc


def model_train_with_map(model, train_loader, val_loader, n_epoch, use_cuda, model_dir, attention_weight=1.0, lr=3e-4, weight_decay=1e-4):
    """Phase 2: Training with attention supervision"""
    task_criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.0)
    attention_criterion = nn.L1Loss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_val_iou = 0
    best_val_acc = 0
    patience_counter = 0
    patience = 10
    
    grad_cam = GradCam(model=model, feature_module=model.layer4,
                      target_layer_names=["2"], use_cuda=use_cuda)
    
    for epoch in np.arange(n_epoch) + 1:
        model.train()
        st = time.time()
        train_losses = []
        
        if use_cuda:
            torch.cuda.empty_cache()
        
        outputs_all = []
        targets_all = []
        
        for batch_idx, (inputs, targets, target_maps, pred_weight, att_weight) in enumerate(train_loader):
            attention_loss = 0
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda(non_blocking=True)
                target_maps = target_maps.cuda()
                pred_weight = pred_weight.cuda()
                att_weight = att_weight.cuda()
            
            att_maps = []
            att_map_labels = []
            att_weights = []
            outputs = model(inputs)
            
            for input, target, target_map, valid_weight in zip(inputs, targets, target_maps, att_weight):
                if valid_weight > 0.0:
                    att_map, _ = grad_cam.get_attention_map(torch.unsqueeze(input, 0), target)
                    att_maps.append(att_map)
                    att_map_labels.append(target_map)
                    att_weights.append(valid_weight)
            
            task_loss = task_criterion(outputs, targets)
            task_loss = torch.mean(pred_weight * task_loss)
            
            if att_maps:
                att_maps = torch.stack(att_maps)
                att_map_labels = torch.stack(att_map_labels)
                att_weights = torch.stack(att_weights)
                attention_loss = attention_criterion(att_maps, att_map_labels)
                attention_loss = torch.mean(att_weights * torch.mean(torch.mean(attention_loss, dim=-1), dim=-1))
                loss = task_loss + attention_weight * attention_loss
            else:
                loss = task_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)
            optimizer.step()
            
            train_losses.append(loss.cpu().detach().item())
            outputs_all.append(outputs)
            targets_all.append(targets)
        
        et = time.time()
        train_time = et - st
        train_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach().item()
        
        # Validation with IoU
        logger.info('Start validation')
        model.eval()
        outputs_all = []
        targets_all = []
        iou = AverageMeter()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, target_maps, pred_weight, att_weight) in enumerate(val_loader):
                if use_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda(non_blocking=True)
                outputs = model(inputs)
                outputs_all.append(outputs)
                targets_all.append(targets)
        
        val_acc = accuracy(torch.cat(outputs_all), torch.cat(targets_all))[0].cpu().detach().item()
        val_iou = iou.avg
        
        # Update scheduler
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_iou = val_iou
            patience_counter = 0
            torch.save(model, os.path.join(model_dir, 'model_phase2_best.pth'))
            logger.info('UPDATE BEST MODEL!')
        else:
            patience_counter += 1
        
        torch.save(model, os.path.join(model_dir, f'model_phase2_epoch{epoch}.pth'))
        logger.info(f'[Phase2 Epoch {epoch}] TRAIN time={train_time:.1f}s loss={np.mean(train_losses):.4f} acc={train_acc:.2f}% VAL acc={val_acc:.2f}% IoU={val_iou:.4f} LR={optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f'Early stopping at epoch {epoch} (no improvement for {patience} epochs)')
            logger.info(f'Best validation accuracy: {best_val_acc:.2f}%, IoU: {best_val_iou:.4f}')
            break
    
    return best_val_iou


def model_test(model, test_loader, use_cuda, class_names=None):
    """Test the model"""
    logger.info('Start testing')
    model.eval()
    st = time.time()
    criterion = nn.CrossEntropyLoss()
    
    if class_names is None:
        class_names = ['CHF', 'pneumonia', 'Normal']

    loss_values = []
    prob_list = []
    target_list = []
    pred_list = []

    logger.info('Starting test iteration...')

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if batch_idx % 10 == 0:
                logger.info(f'Processing batch {batch_idx + 1}...')
            
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda(non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            loss_values.append(float(loss.item()))
            prob_list.append(probs.cpu().numpy())
            target_list.append(targets.cpu().numpy())
            pred_list.append(preds.cpu().numpy())
    
    et = time.time()
    test_time = et - st
    
    y_true = np.concatenate(target_list)
    y_pred = np.concatenate(pred_list)
    y_prob = np.concatenate(prob_list)

    avg_loss = float(np.mean(loss_values)) if loss_values else float("nan")
    accuracy_score = float((y_pred == y_true).mean())

    precision_per, recall_per, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(y_prob.shape[1])), zero_division=0
    )
    macro_precision = float(np.mean(precision_per))
    macro_recall = float(np.mean(recall_per))

    try:
        one_hot = np.zeros_like(y_prob)
        one_hot[np.arange(len(y_true)), y_true] = 1
        macro_auc = float(roc_auc_score(one_hot, y_prob, average='macro', multi_class='ovr'))
        per_class_auc = roc_auc_score(one_hot, y_prob, average=None, multi_class='ovr')
    except ValueError:
        macro_auc = float("nan")
        per_class_auc = [float("nan")] * y_prob.shape[1]

    # IoU per class
    num_classes = y_prob.shape[1]
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf_mat[t, p] += 1
    iou_per = []
    for c in range(num_classes):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float("nan")
        iou_per.append(iou)
    macro_iou = float(np.nanmean(iou_per))

    if class_names is None or len(class_names) != num_classes:
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    logger.info('Per-class metrics:')
    for i, cls_name in enumerate(class_names):
        if i < len(per_class_auc) and i < len(precision_per) and i < len(recall_per) and i < len(iou_per):
            logger.info(f'  {cls_name} AUC: {per_class_auc[i]:.4f}, Precision: {precision_per[i]:.4f}, Recall: {recall_per[i]:.4f}, IoU: {iou_per[i]:.4f}')
    
    logger.info(f'[TEST] time={test_time:.1f}s loss={avg_loss:.4f} acc={accuracy_score*100:.2f}% '
                f'AUC={macro_auc:.4f} precision={macro_precision:.4f} recall={macro_recall:.4f} IoU={macro_iou:.4f}')

    return {
        "loss": avg_loss,
        "accuracy": accuracy_score,
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "macro_precision": macro_precision,
        "per_class_precision": precision_per,
        "macro_recall": macro_recall,
        "per_class_recall": recall_per,
        "macro_iou": macro_iou,
        "per_class_iou": iou_per,
        "test_time": test_time,
    }


def make_parser():
    parser = argparse.ArgumentParser(description='GRADIA Training')
    
    # Data
    parser.add_argument('--config', type=str, default='configs/train_gazegnn.yaml', help='Config path')
    parser.add_argument('--model_dir', type=str, default='results/gradia', help='Output directory')
    parser.add_argument('--class_names', type=list, default=['CHF', 'pneumonia', 'Normal'], help='Label names')
    
    # Training
    parser.add_argument('--train_batch', type=int, default=None, help='train batchsize')
    parser.add_argument('--test_batch', type=int, default=None, help='test batchsize')
    parser.add_argument('--workers', type=int, default=None, help='number of data loading workers')
    parser.add_argument('--n_epoch_phase1', type=int, default=None, help='Number of epochs for phase 1')
    parser.add_argument('--n_epoch_phase2', type=int, default=None, help='Number of epochs for phase 2')
    parser.add_argument('--attention_weight', type=float, default=None, help='Attention loss weight')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (default: from config or 3e-4)')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay (default: 1e-4)')
    parser.add_argument('--cache_dir', type=str, default='cache/heatmaps', help='Directory for cached heatmaps')
    
    # Misc
    parser.add_argument('--gpus', type=str, default='0,1', help='Which gpus to use')
    parser.add_argument('--phase1_only', action='store_true', help='Run phase 1 only')
    parser.add_argument('--phase2_only', action='store_true', help='Run phase 2 only (requires phase1 model)')
    parser.add_argument('--test_only', action='store_true', help='Test only')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    
    return parser


def load_data(args):
    """Load datasets"""
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
    if args.train_batch is None:
        args.train_batch = int(train_cfg.get('batch_size', 32))
    if args.test_batch is None:
        args.test_batch = int(train_cfg.get('test_batch', train_cfg.get('batch_size', args.train_batch)))
    if args.workers is None:
        args.workers = int(train_cfg.get('num_workers', 4))
    if args.n_epoch_phase1 is None:
        args.n_epoch_phase1 = int(train_cfg.get('epochs_phase1', train_cfg.get('epochs', 20)))
    if args.n_epoch_phase2 is None:
        args.n_epoch_phase2 = int(train_cfg.get('epochs_phase2', train_cfg.get('epochs', 20)))
    if args.attention_weight is None:
        args.attention_weight = float(train_cfg.get('attention_weight', 1.0))
    if args.lr is None:
        args.lr = float(train_cfg.get('lr', 3e-4))
    if args.weight_decay is None:
        args.weight_decay = float(train_cfg.get('weight_decay', 1e-4))
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    
    # Splits from config (align with main_train_gnn)
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
    
    # Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    
    max_fix = int(train_cfg.get('max_fixations', 100))
    
    # Create base datasets
    train_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts,
                               dicom_root=dicom_root, max_fixations=max_fix,
                               case_ids=train_ids, classes=args.class_names)
    val_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts,
                             dicom_root=dicom_root, max_fixations=max_fix,
                             case_ids=val_ids, classes=args.class_names)
    test_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts,
                              dicom_root=dicom_root, max_fixations=max_fix,
                              case_ids=test_ids, classes=args.class_names)
    
    # Phase 1 datasets (no heatmap)
    train_dataset_p1 = GRADIADatasetAdapter(train_base, train_transform, return_heatmap=False,
                                            cache_dir=cache_dir, split="train")
    val_dataset_p1 = GRADIADatasetAdapter(val_base, test_transform, return_heatmap=False,
                                          cache_dir=cache_dir, split="val")
    
    # Phase 2 datasets (with heatmap)
    train_dataset_p2 = GRADIADatasetAdapter(train_base, train_transform, return_heatmap=True,
                                            cache_dir=cache_dir, split="train")
    val_dataset_p2 = GRADIADatasetAdapter(val_base, test_transform, return_heatmap=True,
                                          cache_dir=cache_dir, split="val")
    
    # Test dataset
    test_dataset = GRADIADatasetAdapter(test_base, test_transform, return_heatmap=False,
                                        cache_dir=cache_dir, split="test")
    
    # DataLoaders
    # drop_last=True for train/val to avoid batch norm issues with small batches
    train_loader_p1 = DataLoader(train_dataset_p1, batch_size=args.train_batch, shuffle=True,
                                 num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader_p1 = DataLoader(val_dataset_p1, batch_size=args.test_batch, shuffle=False,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    
    train_loader_p2 = DataLoader(train_dataset_p2, batch_size=args.train_batch, shuffle=True,
                                 num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader_p2 = DataLoader(val_dataset_p2, batch_size=args.test_batch, shuffle=False,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    
    # Use 0 workers for test to avoid hanging issues (single-threaded data loading)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False,
                            num_workers=0, pin_memory=False)
    
    return train_loader_p1, val_loader_p1, train_loader_p2, val_loader_p2, test_loader


if __name__ == '__main__':
    args = make_parser().parse_args()
    
    # Set seeds
    random.seed(args.rseed)
    np.random.seed(args.rseed)
    torch.manual_seed(args.rseed)
    
    use_cuda = torch.cuda.is_available() and args.gpus != '-1'
    if use_cuda:
        torch.cuda.manual_seed(args.rseed)
        torch.cuda.manual_seed_all(args.rseed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    # Create output directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    logger.info(f"Using CUDA: {use_cuda}")
    
    # Load data (this sets lr, weight_decay, and other config values)
    train_loader_p1, val_loader_p1, train_loader_p2, val_loader_p2, test_loader = load_data(args)
    
    # Log arguments AFTER load_data so config values are applied
    logger.info(f"[Arguments]: {args}")
    
    # Create model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, len(args.class_names))
    if use_cuda:
        model = model.cuda()
    
    if not args.test_only:
        if not args.phase2_only:
            # Phase 1: Standard training
            logger.info('=' * 60)
            logger.info('PHASE 1: Standard Training')
            logger.info('=' * 60)
            best_val_acc = model_train(model, train_loader_p1, val_loader_p1,
                                       args.n_epoch_phase1, use_cuda, args.model_dir,
                                       lr=args.lr, weight_decay=args.weight_decay)
            logger.info(f'Phase 1 complete. Best validation accuracy: {best_val_acc:.2f}%')
        
        if not args.phase1_only:
            # Phase 2: Training with attention
            logger.info('=' * 60)
            logger.info('PHASE 2: Training with Attention Supervision')
            logger.info('=' * 60)
            
            if args.phase2_only:
                # Load phase 1 model
                model_path = os.path.join(args.model_dir, 'model_phase1_best.pth')
                logger.info(f'Loading phase 1 model from {model_path}')
                model = torch.load(model_path, weights_only=False)
            
            best_val_iou = model_train_with_map(model, train_loader_p2, val_loader_p2,
                                                args.n_epoch_phase2, use_cuda, args.model_dir,
                                                args.attention_weight, lr=args.lr, weight_decay=args.weight_decay)
            logger.info(f'Phase 2 complete. Best validation IOU: {best_val_iou:.4f}')
    
    # Testing
    logger.info('=' * 60)
    logger.info('TESTING')
    logger.info('=' * 60)
    
    # Test best phase 2 model if available, otherwise phase 1
    if os.path.exists(os.path.join(args.model_dir, 'model_phase2_best.pth')):
        model_path = os.path.join(args.model_dir, 'model_phase2_best.pth')
        logger.info(f'Testing Phase 2 best model')
    else:
        model_path = os.path.join(args.model_dir, 'model_phase1_best.pth')
        logger.info(f'Testing Phase 1 best model')
    
    model = torch.load(model_path, weights_only=False)
    
    # Get class names from args
    class_names = args.class_names if hasattr(args, 'class_names') and args.class_names else ['CHF', 'pneumonia', 'Normal']
    
    metrics = model_test(model, test_loader, use_cuda, class_names=class_names)
    
    # Log per-class metrics in final summary
    logger.info("[FINAL TEST] Per-class metrics:")
    per_class_auc = metrics.get("per_class_auc", [])
    per_class_precision = metrics.get("per_class_precision", [])
    per_class_recall = metrics.get("per_class_recall", [])
    per_class_iou = metrics.get("per_class_iou", [])
    
    for i, cls_name in enumerate(class_names):
        if i < len(per_class_auc) and i < len(per_class_precision) and i < len(per_class_recall) and i < len(per_class_iou):
            logger.info(f"  {cls_name}: AUC={per_class_auc[i]:.4f}, Precision={per_class_precision[i]:.4f}, "
                       f"Recall={per_class_recall[i]:.4f}, IoU={per_class_iou[i]:.4f}")
    
    logger.info(
        "[FINAL TEST] Overall: loss={loss:.4f} acc={acc:.2f}% auc={auc:.4f} "
        "precision={prec:.4f} recall={rec:.4f} iou={iou:.4f}".format(
            loss=metrics.get("loss", float("nan")),
            acc=metrics.get("accuracy", 0.0) * 100.0 if metrics else float("nan"),
            auc=metrics.get("macro_auc", float("nan")),
            prec=metrics.get("macro_precision", float("nan")),
            rec=metrics.get("macro_recall", float("nan")),
            iou=metrics.get("macro_iou", float("nan")),
        )
    )

