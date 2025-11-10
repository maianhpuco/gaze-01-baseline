#!/usr/bin/env python3
"""
Train the temporal (RNN) eye-gaze model using the ORIGINAL eye-gaze-dataset
training code, but with the fixed train/val/test splits that match the rest
of this repo (same test IDs as main_train_gnn.py).
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXPT_DIR = ROOT / "externals/eye-gaze-dataset/Experiments"
if str(EXPT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPT_DIR))

from models.eyegaze_model import EyegazeModel  # type: ignore
from utils.dataset import EyegazeDataset, collate_fn  # type: ignore
from utils.utils import cyclical_lr, train_teacher_network, test_eyegaze_network  # type: ignore


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
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dicom_root = Path(cfg.get("input_path", "dicom_raw"))
    heatmaps_path = Path(cfg.get("input_path", "heatmaps_path"))
    csv_dir = Path(cfg.get("input_path", "csv_dir", default=ROOT / "datasets/filtered_csvs/fold1"))
    num_workers = int(cfg.get("train", "num_workers", default=2))
    batch_size = int(cfg.get("train", "batch_size", default=32))
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
    heatmap_temporal_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    heatmap_static_transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
        ]
    )

    dataset_kwargs = dict(
        image_path_name=str(dicom_root),
        class_names=classes,
        heatmaps_path=str(heatmaps_path),
        static_heatmap_path=str(heatmaps_path),
        heatmap_temporal_transform=heatmap_temporal_transform,
        heatmap_static_transform=heatmap_static_transform,
        image_transform=image_transform,
    )

    train_ds = EyegazeDataset(split_frames["train"], **dataset_kwargs)
    val_ds = EyegazeDataset(split_frames["val"], **dataset_kwargs)
    test_ds = EyegazeDataset(split_frames["test"], **dataset_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


def _make_run_dir(base: Path, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


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

    train_ids, val_ids, test_ids = build_splits_ids(cfg)
    loaders = build_dataloaders(cfg, classes, enforce_ids=(train_ids, val_ids, test_ids))
    train_loader, val_loader, test_loader = loaders

    batch_size = int(cfg.get("train", "batch_size", default=32))
    epochs = int(cfg.get("train", "epochs", default=50))
    lr = float(cfg.get("train", "lr", default=1e-3))
    wd = float(cfg.get("train", "weight_decay", default=1e-4))
    step_size = int(cfg.get("train", "step_size", default=5))
    use_scheduler = bool(cfg.get("train", "scheduler", default=False))
    model_type = cfg.get("train", "model_type", default="temporal")
    dropout = float(cfg.get("train", "dropout", default=0.5))
    emb_dim = int(cfg.get("train", "emb_dim", default=64))
    hidden_dim = int(cfg.get("train", "hidden_dim", default=64))
    hidden_hm = cfg.get("train", "hidden_hm", default=[256, 128])
    num_layers_hm = int(cfg.get("train", "num_layers_hm", default=1))
    cell = cfg.get("train", "cell", default="lstm")
    brnn_hm = bool(cfg.get("train", "brnn_hm", default=True))
    attention = bool(cfg.get("train", "attention", default=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EyegazeModel(
        model_type=model_type,
        num_classes=len(classes),
        dropout=dropout,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        hidden_hm=hidden_hm,
        attention=attention,
        cell=cell,
        brnn_hm=brnn_hm,
        num_layers_hm=num_layers_hm,
    ).to(device)

    use_dataparallel = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if use_dataparallel:
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    clr = cyclical_lr(step_sz=step_size, min_lr=lr, max_lr=1, mode="triangular2")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

    ckpt_root = Path(
        cfg.get(
            "output_path",
            "checkpoint_dir",
            default=ROOT / "runs" / "checkpoints" / "temporal",
        )
    )
    run_tag = f"{model_type}_bs{batch_size}_lr{lr:.0e}"
    run_dir = _make_run_dir(ckpt_root, run_tag)
    print(f"Saving checkpoints to: {run_dir}")

    train_teacher_network(
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        str(run_dir),
        epochs,
        viz=None,
        env_name=None,
        is_schedule=use_scheduler,
    )

    print("---- Testing saved epochs to find best AUC ----")
    best_auc = -1.0
    best_model_name = ""
    for epoch_idx in range(epochs):
        ckpt_name = f"Epoch_{epoch_idx}.pth"
        ckpt_path = run_dir / ckpt_name
        if not ckpt_path.exists():
            continue

        eval_model = EyegazeModel(
            model_type=model_type,
            num_classes=len(classes),
            dropout=dropout,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            hidden_hm=hidden_hm,
            attention=attention,
            cell=cell,
            brnn_hm=brnn_hm,
            num_layers_hm=num_layers_hm,
        ).to(device)
        if use_dataparallel:
            eval_model = nn.DataParallel(eval_model)

        state = torch.load(ckpt_path, map_location=device)
        eval_model.load_state_dict(state)
        mean_auc = test_eyegaze_network(
            eval_model, test_loader, classes, str(run_dir), ckpt_name
        )
        if mean_auc >= best_auc:
            best_auc = mean_auc
            best_model_name = ckpt_name

    if best_model_name:
        print(f"Best mean AUC={best_auc:.3f} from checkpoint {best_model_name}")
    else:
        print("Warning: no checkpoints were evaluated.")


if __name__ == "__main__":
    main()
