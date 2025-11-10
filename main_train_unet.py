#!/usr/bin/env python3
"""
Train the UNet-based classifier with the ORIGINAL eye-gaze-dataset procedure,
but reuse the fixed train/val/test splits so every model (GNN/GRADIA/Temporal)
Evaluates on the exact same study IDs.
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXPT_DIR = ROOT / "externals/eye-gaze-dataset/Experiments"
if str(EXPT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPT_DIR))

import segmentation_models_pytorch as smp  # type: ignore
from models.classifier import UnetClassifier  # type: ignore
from utils.dice_loss import GeneralizedDiceLoss  # type: ignore
from utils.dataset import EyegazeDataset  # type: ignore
from utils.utils import cyclical_lr  # type: ignore
from utils.visualization import plot_roc_curve  # type: ignore


# ---------------- Config helpers ----------------
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


# ---------------- Data ----------------
def _load_split_csvs(csv_dir: Path) -> Dict[str, pd.DataFrame]:
    splits: Dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        csv_path = csv_dir / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for split '{name}': {csv_path}")
        splits[name] = pd.read_csv(csv_path)
    return splits


def _filter_to_ids(df: pd.DataFrame, allowed: Iterable[str]) -> pd.DataFrame:
    allowed_set = set(allowed)
    subset = df[df["dicom_id"].astype(str).isin(allowed_set)].copy()
    if subset.empty:
        raise ValueError("Filtered split is empty after enforcing split IDs.")
    return subset


def build_dataloaders(
    cfg: Cfg,
    classes: List[str],
    *,
    enforce_ids: Tuple[List[str], List[str], List[str]],
    heatmaps_threshold: Optional[float],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dicom_root = Path(cfg.get("input_path", "dicom_raw"))
    heatmaps_path = Path(cfg.get("input_path", "heatmaps_path"))
    csv_dir = Path(cfg.get("input_path", "csv_dir", default=ROOT / "datasets/filtered_csvs/fold1"))
    num_workers = int(cfg.get("train", "num_workers", default=0))
    batch_size = int(cfg.get("train", "batch_size", default=24))
    input_size = int(cfg.get("train", "input_size", default=224))

    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root not found: {dicom_root}")
    if not heatmaps_path.exists():
        raise FileNotFoundError(f"Heatmaps path not found: {heatmaps_path}")
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    raw_splits = _load_split_csvs(csv_dir)
    train_ids, val_ids, test_ids = enforce_ids
    split_frames = {
        "train": _filter_to_ids(raw_splits["train"], train_ids),
        "val": _filter_to_ids(raw_splits["val"], val_ids),
        "test": _filter_to_ids(raw_splits["test"], test_ids),
    }

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    seq = iaa.Sequential([iaa.Resize((input_size, input_size))])
    image_transform = transforms.Compose(
        [
            seq.augment_image,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    heatmap_static_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    ds_kwargs = dict(
        image_path_name=str(dicom_root),
        class_names=classes,
        static_heatmap_path=str(heatmaps_path),
        heatmap_static_transform=heatmap_static_transform,
        image_transform=image_transform,
        heatmaps_threshold=heatmaps_threshold,
    )

    train_ds = EyegazeDataset(split_frames["train"], **ds_kwargs)
    val_ds = EyegazeDataset(split_frames["val"], **ds_kwargs)
    test_ds = EyegazeDataset(split_frames["test"], **ds_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


# ---------------- Training / evaluation ----------------
def eval_net(
    model: nn.Module,
    loader: DataLoader,
    classifier_criterion: nn.Module,
    segment_criterion: nn.Module,
    model_type: str,
    gamma: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_seg = 0.0
    total_cls = 0.0
    counter = 0

    with torch.no_grad():
        for images, labels, idx, X_hm, y_hm in loader:
            images = images.to(device)
            labels = labels.to(device)
            if model_type != "baseline":
                y_hm = y_hm.to(device)

            masks_pred, y_pred = model(images)
            if model_type != "baseline":
                seg_loss = segment_criterion(masks_pred, y_hm)
            else:
                seg_loss = torch.tensor(0.0, device=device)
            cls_loss = classifier_criterion(y_pred, labels)
            total_loss += (gamma * cls_loss + (1 - gamma) * seg_loss).item()
            total_seg += float(seg_loss)
            total_cls += float(cls_loss)
            counter += 1

    model.train()
    if counter == 0:
        return 0.0, 0.0, 0.0
    return total_loss / counter, total_seg / counter, total_cls / counter


def eval_eyegaze_network(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    model_dir: Path,
    model_name: str,
    plot_data: bool,
    device: torch.device,
) -> float:
    model.eval()
    y_true: List[List[np.ndarray]] = [[] for _ in class_names]
    y_pred: List[List[np.ndarray]] = [[] for _ in class_names]

    with torch.no_grad():
        for images, labels, idx, X_hm, y_hm in loader:
            images = images.to(device)
            labels = labels.to(device)
            _, logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()
            probs_per_class = [np.array(c) for c in np.array(probs).T.tolist()]
            labels_per_class = labels_np.transpose()
            for c, vals in enumerate(labels_per_class):
                y_true[c].append(vals)
            for c, vals in enumerate(probs_per_class):
                y_pred[c].append(vals)

    y_true = [np.concatenate(c, axis=0) for c in y_true]
    y_pred = [np.concatenate(c, axis=0) for c in y_pred]

    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    log_path = plots_dir / f"{Path(model_name).stem}.log"
    aurocs: List[float] = []
    fpr: Dict[int, np.ndarray] = {}
    tpr: Dict[int, np.ndarray] = {}

    with log_path.open("w", encoding="utf-8") as f:
        for i, cname in enumerate(class_names):
            try:
                score = roc_auc_score(y_true[i], y_pred[i])
                aurocs.append(score)
                fpr[i], tpr[i], _ = roc_curve(y_true[i], y_pred[i])
            except ValueError:
                score = 0.0
                aurocs.append(score)
            f.write(f"{cname}: {score}\n")
        mean_auc = float(np.mean(aurocs))
        f.write("-------------------------\n")
        f.write(f"mean auroc: {mean_auc}\n")

    if plot_data and all(i in fpr for i in range(len(class_names))):
        plot_name = f"{Path(model_name).stem}.png"
        plot_roc_curve(tpr, fpr, class_names, aurocs, plots_dir, plot_name)

    return mean_auc


def train_unet_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    gamma: float,
    step_size: int,
    use_scheduler: bool,
    second_loss: str,
    model_type: str,
    run_dir: Path,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    clr = cyclical_lr(step_sz=step_size, min_lr=lr, max_lr=1, mode="triangular2")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

    classifier_criterion = nn.BCEWithLogitsLoss().to(device)
    if second_loss == "dice":
        segment_criterion = GeneralizedDiceLoss().to(device)
    else:
        segment_criterion = nn.BCEWithLogitsLoss().to(device)

    print(f"Training for {epochs} epochs | gamma={gamma} | second_loss={second_loss}")
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels, idx, X_hm, y_hm in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            if model_type != "baseline":
                y_hm = y_hm.to(device)

            masks_pred, logits = model(images)
            cls_loss = classifier_criterion(logits, labels)
            if model_type != "baseline":
                seg_loss = segment_criterion(masks_pred, y_hm)
            else:
                seg_loss = torch.tensor(0.0, device=device)

            total_loss = gamma * cls_loss + (1 - gamma) * seg_loss
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

        val_loss, val_seg, val_cls = eval_net(
            model,
            val_loader,
            classifier_criterion,
            segment_criterion,
            model_type,
            gamma,
            device,
        )
        print(
            f"[Epoch {epoch:03d}] val_loss={val_loss:.4f} "
            f"val_cls={val_cls:.4f} val_seg={val_seg:.4f}"
        )
        if use_scheduler:
            scheduler.step()

        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = run_dir / f"Epoch_{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)


def evaluate_checkpoints(
    cfg: Cfg,
    classes: List[str],
    *,
    run_dir: Path,
    test_loader: DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    model_type = cfg.get("train", "model_type", default="unet")
    encoder_name = cfg.get("train", "encoder_name", default="timm-efficientnet-b0")
    pretrained_name = cfg.get("train", "pretrained_name", default="noisy-student")
    dropout = float(cfg.get("train", "dropout", default=0.5))
    n_classes = len(classes)
    n_segments = 1
    aux_params = dict(pooling="avg", dropout=dropout, activation=None, classes=n_classes)

    best_auc = -1.0
    best_name = ""

    for epoch in range(1, epochs + 1):
        ckpt_name = f"Epoch_{epoch}.pth"
        ckpt_path = run_dir / ckpt_name
        if not ckpt_path.exists():
            continue

        if model_type == "baseline":
            model = UnetClassifier(
                encoder_name=encoder_name,
                classes=n_segments,
                encoder_weights=pretrained_name,
                aux_params=aux_params,
            ).to(device)
        else:
            model = smp.Unet(
                encoder_name,
                classes=n_segments,
                encoder_weights=pretrained_name,
                aux_params=aux_params,
            ).to(device)

        use_dataparallel = torch.cuda.is_available() and torch.cuda.device_count() > 1
        if use_dataparallel:
            model = nn.DataParallel(model)

        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        auc = eval_eyegaze_network(
            model,
            test_loader,
            classes,
            run_dir,
            ckpt_name,
            plot_data=True,
            device=device,
        )
        if auc >= best_auc:
            best_auc = auc
            best_name = ckpt_name

    if best_name:
        print(f"Best mean AUC={best_auc:.3f} from checkpoint {best_name}")
    else:
        print("Warning: no checkpoints evaluated.")


# ---------------- Main ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = Cfg(args.config)
    classes = cfg.get("train", "classes", default=["Normal", "CHF", "pneumonia"])
    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError("`train.classes` must be a non-empty list.")

    seed = int(cfg.get("train", "seed", default=42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model_type = cfg.get("train", "model_type", default="unet")
    encoder_name = cfg.get("train", "encoder_name", default="timm-efficientnet-b0")
    pretrained_name = cfg.get("train", "pretrained_name", default="noisy-student")
    dropout = float(cfg.get("train", "dropout", default=0.5))
    epochs = int(cfg.get("train", "epochs", default=500))
    lr = float(cfg.get("train", "lr", default=1e-4))
    wd = float(cfg.get("train", "weight_decay", default=1e-4))
    gamma = float(cfg.get("train", "gamma", default=1.0))
    step_size = int(cfg.get("train", "step_size", default=5))
    use_scheduler = bool(cfg.get("train", "scheduler", default=False))
    second_loss = cfg.get("train", "second_loss", default="ce")
    heatmaps_threshold = None if second_loss == "ce" else cfg.get("train", "heatmaps_threshold", default=None)

    train_ids, val_ids, test_ids = build_splits_ids(cfg)
    loaders = build_dataloaders(
        cfg,
        classes,
        enforce_ids=(train_ids, val_ids, test_ids),
        heatmaps_threshold=heatmaps_threshold,
    )
    train_loader, val_loader, test_loader = loaders

    n_classes = len(classes)
    n_segments = 1
    aux_params = dict(pooling="avg", dropout=dropout, activation=None, classes=n_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "baseline":
        model = UnetClassifier(
            encoder_name=encoder_name,
            classes=n_segments,
            encoder_weights=pretrained_name,
            aux_params=aux_params,
        ).to(device)
    else:
        model = smp.Unet(
            encoder_name,
            classes=n_segments,
            encoder_weights=pretrained_name,
            aux_params=aux_params,
        ).to(device)

    use_dataparallel = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if use_dataparallel:
        model = nn.DataParallel(model)

    ckpt_root = Path(
        cfg.get(
            "output_path",
            "checkpoint_dir",
            default=ROOT / "runs" / "checkpoints" / "unet",
        )
    )
    run_tag = f"{model_type}_bs{cfg.get('train', 'batch_size', default=24)}_lr{lr:.0e}"
    run_dir = (ckpt_root / f"{run_tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_unet_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=wd,
        gamma=gamma,
        step_size=step_size,
        use_scheduler=use_scheduler,
        second_loss=second_loss,
        model_type=model_type,
        run_dir=run_dir,
        device=device,
    )

    evaluate_checkpoints(
        cfg,
        classes,
        run_dir=run_dir,
        test_loader=test_loader,
        epochs=epochs,
        device=device,
    )


if __name__ == "__main__":
    main()
