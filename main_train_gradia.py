#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import roc_curve, auc
import torchvision.models as models
from torchvision.models import ResNet50_Weights

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Dataset (robust import with fallback to direct path)
try:
    from egd_cxr_dataset.datasets.edg_cxr import EGDCXRDataset, create_dataloader  # type: ignore
except ModuleNotFoundError:
    DATASETS_DIR = SRC / "egd_cxr_dataset" / "datasets"
    if str(DATASETS_DIR) not in sys.path:
        sys.path.insert(0, str(DATASETS_DIR))
    from edg_cxr import EGDCXRDataset, create_dataloader  # type: ignore


# ------------- GRADIA Components (extracted to avoid import issues) -------------
class FeatureExtractor():
    """Class for extracting activations and registering gradients from targetted intermediate layers"""
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


class ModelOutputs():
    """Class for making a forward pass and getting network output + activations + gradients"""
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
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        return target_activations, x


class GradCam:
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
        # Ensure we're in eval mode for attention extraction
        was_training = self.model.training
        self.model.eval()
        
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
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

        grads_val = self.extractor.get_gradients()
        if len(grads_val) == 0:
            # Fallback: return zero attention map
            if self.cuda:
                cam = torch.zeros((7, 7)).cuda()
            else:
                cam = torch.zeros((7, 7))
            # Restore training state
            if was_training:
                self.model.train()
            return cam, output
            
        grads_val = grads_val[-1]
        target = features[-1].squeeze()
        weights = torch.mean(grads_val, axis=(2, 3)).squeeze()

        if self.cuda:
            cam = torch.zeros(target.shape[1:]).cuda()
        else:
            cam = torch.zeros(target.shape[1:])

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = cam - torch.min(cam)
        cam_max = torch.max(cam)
        if cam_max > 0:
            cam = cam / cam_max

        # Restore training state
        if was_training:
            self.model.train()
            
        return cam, output


class AverageMeter(object):
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


# ---------------- Config ----------------
def _yaml_load(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Cfg:
    def __init__(self, p: Path):
        self._d = _yaml_load(p)
    def get(self, *keys, default=None):
        cur = self._d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur


# ------------- Original GRADIA-style Dataset Adapter -------------
class GRADIADataset(Dataset):
    """Dataset adapter following original GRADIA pattern.
    Returns: (image, label, attention_map, pred_weight, att_weight) for training with weights.
    Follows the original GRADIA ImageFolderWithMapsAndWeights pattern but wraps EGDCXRDataset.
    """
    def __init__(self, base: EGDCXRDataset, mode: str = "train"):
        self.base = base
        self.mode = mode
        
        # Original GRADIA normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        
        # Original GRADIA transforms
        if mode == "train":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize,
            ])

    def __len__(self) -> int:
        return len(self.base)

    def _resize_attention_label(self, attention_map: np.ndarray, width: int = 7, height: int = 7) -> np.ndarray:
        """Resize attention map following original GRADIA pattern."""
        # Keep float energy before resizing to avoid zeroing-out sparse maps
        att_map = np.clip(attention_map, 0.0, 1.0).astype(np.float32)
        # Resize to 7x7 (matching GRADIA's ResNet layer4 output size)
        img_att_resized = cv2.resize(att_map, (width, height), interpolation=cv2.INTER_AREA)
        # Blur to reward nearby pixels
        img_att_resized = cv2.GaussianBlur(img_att_resized, (3, 3), 0)
        # Normalize
        max_val = np.max(img_att_resized)
        if max_val > 0:
            img_att_resized = np.float32(img_att_resized / max_val)
        else:
            img_att_resized = np.float32(img_att_resized)
        return img_att_resized

    def _make_gaze_map(self, h: int, w: int, xy_px: torch.Tensor, dwell: torch.Tensor) -> np.ndarray:
        """Create gaze attention map from fixation points."""
        g = np.zeros((h, w), dtype=np.float32)
        if xy_px.numel() > 0:
            xs = xy_px[:, 0].clamp(0, w - 1).round().long().cpu().numpy()
            ys = xy_px[:, 1].clamp(0, h - 1).round().long().cpu().numpy()
            d = dwell.cpu().numpy().astype(np.float32)
            for i in range(min(len(xs), len(d))):
                g[ys[i], xs[i]] += d[i]
        
        # Normalize to [0, 1]
        if g.max() > 0:
            g = g / g.max()
        return g.astype(np.float32)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        img1 = sample["image"]  # [1,224,224]
        
        # Convert to RGB PIL Image (matching original GRADIA)
        img3 = img1.repeat(3, 1, 1)  # [3,224,224]
        # Convert to PIL image (0-1 range)
        img3_np = img3.permute(1, 2, 0).numpy()  # [H, W, C]
        img3_pil = transforms.ToPILImage()(img3)
        
        # Apply transforms
        img_tensor = self.transform(img3_pil)

        # Create gaze attention map at 224x224
        fx = sample["fixations"]
        xy = fx["xy"]  # [T,2]
        dwell = fx["dwell"]  # [T]
        gaze_map_224 = self._make_gaze_map(224, 224, xy, dwell)
        
        # Resize to 7x7 following original GRADIA pattern
        gaze_map_7x7 = self._resize_attention_label(gaze_map_224, width=7, height=7)
        gaze_tensor = torch.from_numpy(gaze_map_7x7).float()

        # Label: use single_index
        y_idx = int(sample["labels"]["single_index"].item())
        if y_idx < 0:
            y_idx = 0
        y_t = torch.tensor(y_idx, dtype=torch.long)
        
        # Weights (following original GRADIA ImageFolderWithMapsAndWeights pattern)
        # Default weights: pred_weight=1, att_weight=0 (train on all samples for task loss)
        # For samples with valid gaze maps, att_weight > 0 to train attention
        pred_weight = 1.0
        att_weight = 1.0 if xy.numel() > 0 else 0.0  # Only train attention if gaze data exists
        
        return img_tensor, y_t, gaze_tensor, pred_weight, att_weight


class GRADIAImageFolderDataset(Dataset):
    """Load GRADIA-style ImageFolder exports with precomputed attention maps."""
    def __init__(self, root: Path, split: str = "train", mode: str = "train"):
        self.split_root = Path(root) / split
        if not self.split_root.exists():
            raise FileNotFoundError(f"GRADIA ImageFolder split not found: {self.split_root}")
        self.mode = mode

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        if mode == "train":
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize,
            ])

        self.dataset = datasets.ImageFolder(str(self.split_root), transform=transform)
        self.samples = self.dataset.samples

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _resize_attention_label(attention_map: np.ndarray, width: int = 7, height: int = 7) -> np.ndarray:
        """Exact resizing pipeline from original GRADIA (uint8 → resize → blur)."""
        att_map = np.clip(attention_map, 0.0, 1.0)
        att_map_uint8 = np.uint8(att_map * 255.0)
        img_att_resized = cv2.resize(att_map_uint8, (width, height), interpolation=cv2.INTER_AREA)
        img_att_resized = cv2.GaussianBlur(img_att_resized, (3, 3), 0)
        max_val = np.max(img_att_resized)
        if max_val > 0:
            img_att_resized = np.float32(img_att_resized / max_val)
        else:
            img_att_resized = np.float32(img_att_resized)
        return img_att_resized

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        path, _ = self.samples[idx]
        att_path = Path(path).with_suffix(".npy")
        if att_path.exists():
            att_map_224 = np.load(att_path)
        else:
            att_map_224 = np.zeros((224, 224), dtype=np.float32)

        gaze_map_7x7 = self._resize_attention_label(att_map_224, width=7, height=7)
        att_weight = 1.0 if float(np.max(att_map_224)) > 0 else 0.0

        return (
            img,
            torch.tensor(label, dtype=torch.long),
            torch.from_numpy(gaze_map_7x7).float(),
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(att_weight, dtype=torch.float32),
        )


# ------------- Utils -------------
def build_splits_ids(cfg: Cfg) -> Tuple[List[str], List[str], List[str]]:
    split_dir = Path(cfg.get("split_files", "dir", default=ROOT / "configs" / "splits"))
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    def _read(name: str) -> List[str]:
        p = split_dir / f"{name}_ids.txt"
        ids: List[str] = []
        for raw in p.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if s and not s.startswith("#"):
                ids.append(s)
        return ids
    return _read("train"), _read("val"), _read("test")


def load_preprocessed_classes(root: Path, fallback: List[str]) -> List[str]:
    """Read class ordering from preprocessing metadata if present."""
    meta_path = Path(root) / "classes.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            classes = data.get("classes")
            if isinstance(classes, list) and classes:
                return [str(c) for c in classes]
        except Exception:
            pass
    return fallback


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 3) -> Tuple[float, float, List[float]]:
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_true = labels.cpu().numpy()
        # acc
        preds = probs.argmax(axis=1)
        acc = float((preds == y_true).mean())
        # auc
        y_bin = np.zeros((len(y_true), num_classes), dtype=np.float32)
        for i, c in enumerate(y_true):
            if 0 <= c < num_classes:
                y_bin[i, c] = 1.0
        per_auc: List[float] = []
        for c in range(num_classes):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
                per_auc.append(auc(fpr, tpr))
            except Exception:
                per_auc.append(float("nan"))
        macro = float(np.nanmean(per_auc))
    return acc, macro, per_auc


def compute_iou(pred_mask: np.ndarray, target_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Compute IoU between predicted and target attention maps."""
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    target_binary = (target_mask > threshold).astype(np.uint8)
    intersection = np.bitwise_and(pred_binary, target_binary)
    union = np.bitwise_or(pred_binary, target_binary)
    if np.sum(union) == 0:
        return 0.0
    return float(np.sum(intersection) / np.sum(union))


# ------------- Training with GRADIA -------------
def train_one_epoch_gradia(model, grad_cam, loader, device, optimizer, 
                          task_criterion, attention_criterion, 
                          attention_weight: float = 1.0) -> Tuple[float, float, float, float]:
    """Train one epoch with GRADIA-style attention supervision.
    Follows the original GRADIA training pattern from model_train_with_map function exactly.
    """
    model.train()
    total_task_loss = 0.0
    total_att_loss = 0.0
    total_correct = 0
    total_count = 0
    
    for batch_idx, (inputs, targets, target_maps, pred_weight, att_weight) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        target_maps = target_maps.to(device)
        pred_weight = pred_weight.to(device)
        att_weight = att_weight.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute attention maps using GradCAM (following original GRADIA exactly)
        att_maps = []
        att_map_labels = []
        att_weights = []
        
        for input, target, target_map, valid_weight in zip(inputs, targets, target_maps, att_weight):
            # Only train on img with attention labels (valid_weight > 0.0)
            if valid_weight > 0.0:
                # Get attention maps from grad-CAM
                att_map, _ = grad_cam.get_attention_map(torch.unsqueeze(input, 0), target.item())
                # Detach to avoid memory accumulation from GradCAM's retained graphs
                att_maps.append(att_map.detach())
                att_map_labels.append(target_map)
                att_weights.append(valid_weight)
                # Clear gradients after each GradCAM call to prevent memory buildup
                model.zero_grad()
        
        # Compute losses (following original GRADIA pattern exactly)
        task_loss = task_criterion(outputs, targets)
        task_loss = torch.mean(pred_weight * task_loss)
        
        attention_loss = torch.tensor(0.0).to(device)
        if att_maps:
            att_maps = torch.stack(att_maps)
            att_map_labels = torch.stack(att_map_labels)
            att_weights = torch.stack(att_weights)
            attention_loss = attention_criterion(att_maps, att_map_labels)
            # Original GRADIA weighted averaging: mean over spatial dims, then weight by att_weights
            attention_loss = torch.mean(att_weights * torch.mean(torch.mean(attention_loss, dim=-1), dim=-1))
            
            # Combined loss with attention_weight parameter
            loss = task_loss + attention_weight * attention_loss
        else:
            loss = task_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stats
        with torch.no_grad():
            total_task_loss += float(task_loss.item()) * targets.size(0)
            total_att_loss += float(attention_loss.item()) * targets.size(0)
            total_correct += int((outputs.argmax(dim=1) == targets).sum().item())
            total_count += int(targets.size(0))
        
        # Clear GPU cache every 10 batches to prevent memory fragmentation
        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    avg_task_loss = total_task_loss / max(1, total_count)
    avg_att_loss = total_att_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    avg_total_loss = avg_task_loss + avg_att_loss
    
    return avg_total_loss, avg_task_loss, avg_att_loss, avg_acc


def evaluate_gradia(model, grad_cam, loader, device, criterion, 
                    compute_iou_metric: bool = False, iou_sample_limit: int = 50) -> Tuple[float, float, float, List[float], float]:
    """Evaluate model with optional IoU computation.
    
    Args:
        iou_sample_limit: Maximum number of samples to compute IoU for (to speed up eval)
    """
    model.eval()
    total_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total = 0
    
    iou_meter = AverageMeter()
    iou_samples_computed = 0
    
    for img, y, gaze_map, pred_weight, att_weight in loader:
        img = img.to(device)
        y = y.to(device)
        gaze_map = gaze_map.to(device)
        
        with torch.no_grad():
            logits = model(img)
            loss = criterion(logits, y)
            # Handle both reduction='none' and reduction='mean' criterions
            if loss.dim() > 0:  # reduction='none'
                loss = torch.mean(loss)
            total_loss += float(loss.item()) * y.size(0)
            total += int(y.size(0))
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
        
        # Compute IoU if requested (needs gradients for GradCAM) - limited samples
        if compute_iou_metric and iou_samples_computed < iou_sample_limit:
            for i in range(min(img.size(0), iou_sample_limit - iou_samples_computed)):
                single_img = img[i:i+1].requires_grad_(True)
                pred_class = logits[i].argmax().item()
                
                # Get predicted attention map (at 224x224)
                att_map_7x7, _ = grad_cam.get_attention_map(single_img, pred_class)
                att_map_224 = torch.nn.functional.interpolate(
                    att_map_7x7.unsqueeze(0).unsqueeze(0),
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().cpu().detach().numpy()
                
                # Get target gaze map (7x7) and upsample to 224x224 for fair comparison
                target_map_7x7 = gaze_map[i].cpu()
                target_map_224 = torch.nn.functional.interpolate(
                    target_map_7x7.unsqueeze(0).unsqueeze(0),
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
                
                # Compute IoU
                if target_map_224.max() > 0:  # Only compute if there's a valid gaze map
                    single_iou = compute_iou(att_map_224, target_map_224, threshold=0.5)
                    iou_meter.update(single_iou, 1)
                    iou_samples_computed += 1
    
    if total == 0:
        return 0.0, 0.0, 0.0, [float("nan")] * 3, 0.0
    
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    acc, macro_auc, per_auc = compute_metrics(logits_cat, labels_cat, num_classes=logits_cat.shape[1])
    
    return total_loss / total, acc, macro_auc, per_auc, iou_meter.avg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--attention_weight", type=float, default=1.0,
                   help="Weight for attention loss (default: 1.0, following original GRADIA)")
    ap.add_argument("--iou_samples", type=int, default=50,
                   help="Number of samples to compute IoU on during validation (default: 50)")
    ap.add_argument("--pretrained", action="store_true", default=True,
                   help="Use pretrained ResNet50 (default: True)")
    ap.add_argument("--evaluate", action="store_true",
                   help="Only evaluate on test set")
    ap.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint for evaluation or resume training")
    ap.add_argument("--gradia_folder", type=Path, default=None,
                   help="Path to GRADIA-style ImageFolder export (use scripts/preprocess_gradia_data.py)")
    args = ap.parse_args()

    cfg = Cfg(args.config)

    # Paths
    root = Path(cfg.get("input_path", "gaze_raw"))
    seg = Path(cfg.get("input_path", "segmentation_dir"))
    transcripts = Path(cfg.get("input_path", "transcripts_dir", default=seg))
    dicom_root = Path(cfg.get("input_path", "dicom_raw"))

    classes = cfg.get("train", "classes", default=["CHF", "pneumonia", "Normal"])
    batch_size = int(cfg.get("train", "batch_size", default=32))
    epochs = int(cfg.get("train", "epochs", default=50))
    max_fix = int(cfg.get("train", "max_fixations", default=8))
    num_workers = int(cfg.get("train", "num_workers", default=2))
    lr = float(cfg.get("train", "lr", default=1e-4))
    wd = float(cfg.get("train", "weight_decay", default=1e-4))

    ckpt_dir = Path(cfg.get("output_path", "checkpoint_dir", default=ROOT / "runs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_name = cfg.get("output_path", "best_ckpt_name", default="gradia_best.pt")

    preprocessed_root = args.gradia_folder
    if preprocessed_root is not None:
        classes = load_preprocessed_classes(preprocessed_root, classes)
        num_classes = len(classes)
        tr_ds = GRADIAImageFolderDataset(preprocessed_root, split="train", mode="train")
        val_ds = GRADIAImageFolderDataset(preprocessed_root, split="val", mode="test")
        test_ds = GRADIAImageFolderDataset(preprocessed_root, split="test", mode="test")
    else:
        # Splits
        train_ids, val_ids, test_ids = build_splits_ids(cfg)

        # Build base datasets
        dtr_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts, dicom_root=dicom_root,
                                 max_fixations=max_fix, case_ids=train_ids, classes=classes)
        dval_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts, dicom_root=dicom_root,
                                  max_fixations=max_fix, case_ids=val_ids, classes=classes)
        dtest_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts, dicom_root=dicom_root,
                                   max_fixations=max_fix, case_ids=test_ids, classes=classes)

        # Wrap for GRADIA (following original GRADIA dataloader pattern)
        tr_ds = GRADIADataset(dtr_base, mode="train")
        val_ds = GRADIADataset(dval_base, mode="test")
        test_ds = GRADIADataset(dtest_base, mode="test")
        num_classes = len(classes)

    # DataLoader setup following original GRADIA (shuffle=True for train, False for val/test)
    # Note: pin_memory=False instead of True to avoid SLURM deadlock issues
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    # Model: ResNet50 with modified classifier
    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # Use weights parameter instead of deprecated pretrained
    weights = ResNet50_Weights.DEFAULT if args.pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(2048, num_classes)
    model = model.to(device)
    
    # Use DataParallel for multi-GPU (following original GRADIA)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # GradCAM setup
    feature_module = model.module.layer4 if isinstance(model, nn.DataParallel) else model.layer4
    grad_cam = GradCam(model=model, feature_module=feature_module, 
                       target_layer_names=["2"], use_cuda=torch.cuda.is_available())

    # Load checkpoint if provided
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found!")

    # Evaluation only mode
    if args.evaluate:
        print("=" * 60)
        print("EVALUATION MODE")
        print("=" * 60)
        
        # Evaluate on validation set
        val_loss, val_acc, val_macroauc, val_perauc, val_iou = evaluate_gradia(
            model, grad_cam, val_loader, device, nn.CrossEntropyLoss(), 
            compute_iou_metric=True, iou_sample_limit=args.iou_samples
        )
        print(f"[VAL]  loss={val_loss:.4f}  acc={val_acc:.3f}  macroAUC={val_macroauc:.3f}  IoU={val_iou:.3f}")
        print(f"       perClassAUC={[round(x,3) for x in val_perauc]}")
        
        # Evaluate on test set
        test_loss, test_acc, test_macroauc, test_perauc, test_iou = evaluate_gradia(
            model, grad_cam, test_loader, device, nn.CrossEntropyLoss(), 
            compute_iou_metric=True, iou_sample_limit=args.iou_samples
        )
        print(f"[TEST] loss={test_loss:.4f}  acc={test_acc:.3f}  macroAUC={test_macroauc:.3f}  IoU={test_iou:.3f}")
        print(f"       perClassAUC={[round(x,3) for x in test_perauc]}")
        return

    # Loss & Optimizer (following original GRADIA exactly)
    task_criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    attention_criterion = nn.L1Loss(reduction='none').to(device)
    # Original GRADIA uses lr=0.0001, weight_decay from config
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    print("=" * 60)
    print("TRAINING WITH GRADIA (Original Pattern)")
    print("=" * 60)
    print(f"Dataset: {len(tr_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    print(f"Classes: {classes}")
    print(f"Attention weight: {args.attention_weight}")
    print(f"IoU samples per validation: {args.iou_samples}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}, Weight decay: {wd}")
    print(f"Using same splits as main_train_gnn.py")
    print("=" * 60)
    import sys
    sys.stdout.flush()

    best_val_auc = -1.0
    best_val_iou = -1.0
    patience = 50  # Increased from 20 to allow more training
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        try:
            st = time.time()
            train_loss, train_task_loss, train_att_loss, train_acc = train_one_epoch_gradia(
                model, grad_cam, tr_loader, device, optimizer, 
                task_criterion, attention_criterion, args.attention_weight
            )
            train_time = time.time() - st
            
            st = time.time()
            val_loss, val_acc, val_macroauc, val_perauc, val_iou = evaluate_gradia(
                model, grad_cam, val_loader, device, task_criterion, 
                compute_iou_metric=True, iou_sample_limit=args.iou_samples
            )
            val_time = time.time() - st
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[OOM Error] Out of memory at epoch {epoch}!")
                if torch.cuda.is_available():
                    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, "
                          f"{torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
                    torch.cuda.empty_cache()
                print("Try reducing batch_size in config or increasing GPU memory.")
                raise
            else:
                raise

        # Print in format similar to main_train_gnn.py
        print(f"[Epoch {epoch:2d}] TRAIN  loss={train_loss:.4f}  acc={train_acc:.3f}  "
              f"(cls={train_task_loss:.4f}, att={train_att_loss:.4f})  time={train_time:.1f}s")
        print(f"[Epoch {epoch:2d}] VAL    loss={val_loss:.4f}  acc={val_acc:.3f}  "
              f"macroAUC={val_macroauc:.3f}  IoU={val_iou:.3f}  perClassAUC={[round(x,3) for x in val_perauc]}  "
              f"time={val_time:.1f}s")

        # Save best model based on macro AUC
        if val_macroauc > best_val_auc:
            best_val_auc = val_macroauc
            best_val_iou = val_iou
            patience_counter = 0
            save_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(save_dict, ckpt_dir / best_name)
            print(f"           → Best model saved! (AUC={val_macroauc:.3f}, IoU={val_iou:.3f})")
        else:
            patience_counter += 1
            print(f"           → No improvement ({patience_counter}/{patience})")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping training.")
            print(f"[Best Val] macroAUC={best_val_auc:.3f}  IoU={best_val_iou:.3f}")
            break

    print("=" * 60)
    print("TRAINING COMPLETE - FINAL TEST EVALUATION")
    print("=" * 60)
    
    # Load best model for final test
    best_path = ckpt_dir / best_name
    if best_path.exists():
        print(f"Loading best model from {best_path}")
        state_dict = torch.load(best_path, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    
    # Final test evaluation
    test_loss, test_acc, test_macroauc, test_perauc, test_iou = evaluate_gradia(
        model, grad_cam, test_loader, device, task_criterion, 
        compute_iou_metric=True, iou_sample_limit=args.iou_samples
    )
    print(f"[FINAL TEST] LOSS={test_loss:.4f}  ACC={test_acc:.3f}  macroAUC={test_macroauc:.3f}  "
          f"IoU={test_iou:.3f}")
    print(f"             perClassAUC={[round(x,3) for x in test_perauc]}")
    print(f"[BEST VAL]   macroAUC={best_val_auc:.3f}  IoU={best_val_iou:.3f}")


if __name__ == "__main__":
    main()
