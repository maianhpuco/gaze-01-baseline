#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_curve, auc

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

# GazeGNN model
sys.path.insert(0, str(ROOT / "src/externals/GazeGNN"))
from models.pvig_gaze import pvig_ti_224_gelu, DeepGCN  # type: ignore


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


# ------------- Adapters -------------
class GazeGNNAdapter(Dataset):
    """Wrap EGDCXRDataset to output (img3, label_idx, gaze56) for GazeGNN.
    - img3: [3,224,224], ImageNet normalized
    - label_idx: Long in {0,1,2}
    - gaze56: [56,56] float in [0,1]
    """
    def __init__(self, base: EGDCXRDataset, mode: str = "train"):
        self.base = base
        self.mode = mode
        self.tf_train = transforms.Compose([
            transforms.RandomRotation((-5, 5)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
        self.tf_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.base)

    def _make_gaze_map(self, h: int, w: int, xy_px: torch.Tensor, dwell: torch.Tensor) -> np.ndarray:
        g = np.zeros((h, w), dtype=np.float32)
        if xy_px.numel() > 0:
            xs = xy_px[:, 0].clamp(0, w - 1).round().long().cpu().numpy()
            ys = xy_px[:, 1].clamp(0, h - 1).round().long().cpu().numpy()
            d = dwell.cpu().numpy().astype(np.float32)
            for i in range(min(len(xs), len(d))):
                g[ys[i], xs[i]] += d[i]
        # match GazeGNN preprocessing
        g = np.log(g + 0.01)
        g_min, g_max = float(g.min()), float(g.max())
        if g_max > g_min:
            g = (g - g_min) / (g_max - g_min)
        return g.astype(np.float32)

    def _patch_gaze(self, gaze224: np.ndarray) -> np.ndarray:
        # replicate read_data.getPatchGaze behavior
        g = np.zeros((56, 56), dtype=np.float32)
        for i in range(56):
            for j in range(56):
                x1 = max(0, 4 * i - 7)
                x2 = min(223, 4 * i + 7)
                y1 = max(0, 4 * j - 7)
                y2 = min(223, 4 * j + 7)
                g[i, j] = gaze224[x1:x2, y1:y2].sum()
        if g.max() > g.min():
            g = (g - g.min()) / (g.max() - g.min())
        return g

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        img1 = sample["image"]  # [1,224,224]
        # expand to RGB and convert back to PIL for transforms alignment
        img3 = img1.repeat(3, 1, 1)  # [3,224,224]
        pil_img = transforms.ToPILImage()(img3)
        tform = self.tf_train if self.mode == "train" else self.tf_test
        state = torch.get_rng_state()
        pil_img = tform(pil_img)
        img_tensor = self.normalize(self.to_tensor(pil_img))
        torch.set_rng_state(state)

        fx = sample["fixations"]
        xy = fx["xy"]  # [T,2]
        dwell = fx["dwell"]  # [T]
        gaze_np = self._make_gaze_map(224, 224, xy, dwell)
        gaze_pil = transforms.ToPILImage()(gaze_np)
        gaze_pil = tform(gaze_pil)
        gaze_tensor = self.to_tensor(gaze_pil)[0]  # [224,224]
        gaze56 = self._patch_gaze(gaze_tensor.numpy())  # [56,56]
        gaze56_t = torch.from_numpy(gaze56).float()

        # label: use single_index (skip ambiguous = -1 by simple remap to background class 0)
        y_idx = int(sample["labels"]["single_index"].item())
        if y_idx < 0:
            y_idx = 0
        y_t = torch.tensor(y_idx, dtype=torch.long)
        return img_tensor, y_t, gaze56_t


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


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 3) -> Tuple[float, float, float, List[float]]:
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
    return acc, macro, macro, per_auc  # macro repeated to match requested output tuple len


# ------------- Training -------------
def train_one_epoch(model, loader, device, optimizer, criterion) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for img, y, gaze in loader:
        img = img.to(device)
        y = y.to(device)
        gaze = gaze.to(device)
        logits = model(img, gaze)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss += float(loss.item()) * y.size(0)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_count += int(y.size(0))
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> Tuple[float, float, float, List[float]]:
    model.eval()
    total_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total = 0
    for img, y, gaze in loader:
        img = img.to(device)
        y = y.to(device)
        gaze = gaze.to(device)
        logits = model(img, gaze)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        total += int(y.size(0))
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    if total == 0:
        return 0.0, 0.0, 0.0, [float("nan")] * 3
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    acc, macro_auc, _, per_auc = compute_metrics(logits_cat, labels_cat, num_classes=logits_cat.shape[1])
    return total_loss / total, acc, macro_auc, per_auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
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
    lr = float(cfg.get("train", "lr", default=3e-4))
    wd = float(cfg.get("train", "weight_decay", default=1e-4))
    label_smoothing = float(cfg.get("train", "label_smoothing", default=0.0))

    ckpt_dir = Path(cfg.get("output_path", "checkpoint_dir", default=ROOT / "runs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_name = cfg.get("output_path", "best_ckpt_name", default="gazegnn_best.pt")

    # Splits
    train_ids, val_ids, test_ids = build_splits_ids(cfg)

    # Build base datasets
    dtr_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts, dicom_root=dicom_root,
                             max_fixations=max_fix, case_ids=train_ids, classes=classes)
    dval_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts, dicom_root=dicom_root,
                              max_fixations=max_fix, case_ids=val_ids, classes=classes)
    dtest_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts, dicom_root=dicom_root,
                               max_fixations=max_fix, case_ids=test_ids, classes=classes)

    # Wrap for GazeGNN
    tr_ds = GazeGNNAdapter(dtr_base, mode="train")
    val_ds = GazeGNNAdapter(dval_base, mode="test")
    test_ds = GazeGNNAdapter(dtest_base, mode="test")

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    try:
        model = pvig_ti_224_gelu().to(device)
    except FileNotFoundError:
        # Build the tiny variant without loading pretrained weights
        class _Opt:
            def __init__(self):
                self.k = 9
                self.conv = 'mr'
                self.act = 'gelu'
                self.norm = 'batch'
                self.bias = True
                self.dropout = 0.0
                self.use_dilation = True
                self.epsilon = 0.2
                self.use_stochastic = False
                self.drop_path = 0.0
                self.blocks = [2, 2, 6, 2]
                self.pyramid_levels = [4, 11, 14]
                self.imagesize = 224
                self.channels = [48, 96, 240, 384]
                self.n_classes = 1000
                self.emb_dims = 1024
        model = DeepGCN(_Opt())
        # replace head to 3 classes and add b1 as in pvig_ti_224_gelu
        model.prediction[-1] = nn.Conv2d(model.prediction[-1].in_channels, 3, kernel_size=(1,1), stride=(1,1))
        m = model.prediction[-1]
        torch.nn.init.kaiming_normal_(m.weight)
        m.weight.requires_grad = True
        if m.bias is not None:
            m.bias.data.zero_()
            m.bias.requires_grad = True
        model.b1 = nn.BatchNorm2d(1)
        model = model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Loss & Optim
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val = -1.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, tr_loader, device, optimizer, criterion)
        val_loss, val_acc, val_macroauc, val_perauc = evaluate(model, val_loader, device, criterion)

        # Expanded prints in requested style; cls=CE loss, txt=0.0 placeholder; gate stats unavailable
        print(f"[Epoch {epoch}] TRAIN  loss={train_loss:.4f}  acc={train_acc:.3f}  (cls={train_loss:.4f}, txt={0.0:.4f}) gate(μ={0.0:.3f},σ={0.0:.3f})")
        print(f"[Epoch {epoch}] VAL    loss={val_loss:.4f}  acc={val_acc:.3f}  macroAUC={val_macroauc:.3f}  perClassAUC={[round(x,3) for x in val_perauc]}")

        if val_macroauc > best_val:
            best_val = val_macroauc
            torch.save(model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                       ckpt_dir / best_name)

    # Test at end
    test_loss, test_acc, test_macroauc, test_perauc = evaluate(model, test_loader, device, criterion)
    print(f"[Test] LOSS={test_loss:.4f}  ACC={test_acc:.3f}  macroAUC={test_macroauc:.3f}  perClassAUC={[round(x,3) for x in test_perauc]}")


if __name__ == "__main__":
    main()


