#!/usr/bin/env python3
"""
Export EGDCXRDataset samples into the original GRADIA ImageFolder format.

Creates {train,val,test}/{class_name} directories with RGB PNGs plus
co-located .npy attention maps (224x224 float32) so main_train_gradia.py
can load data through ImageFolderWithMaps-style logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
from typing import Sequence

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from main_train_gradia import Cfg, build_splits_ids

try:
    from egd_cxr_dataset.datasets.edg_cxr import EGDCXRDataset  # type: ignore
except ModuleNotFoundError:
    DATASETS_DIR = SRC / "egd_cxr_dataset" / "datasets"
    if str(DATASETS_DIR) not in sys.path:
        sys.path.insert(0, str(DATASETS_DIR))
    from edg_cxr import EGDCXRDataset  # type: ignore


def _make_gaze_map(h: int, w: int, xy_px: torch.Tensor, dwell: torch.Tensor) -> np.ndarray:
    """Replicate GRADIA-style attention target generation at 224x224."""
    grid = np.zeros((h, w), dtype=np.float32)
    if xy_px.numel() > 0:
        xs = xy_px[:, 0].clamp(0, w - 1).round().long().cpu().numpy()
        ys = xy_px[:, 1].clamp(0, h - 1).round().long().cpu().numpy()
        d = dwell.cpu().numpy().astype(np.float32)
        for idx in range(min(len(xs), len(d))):
            grid[ys[idx], xs[idx]] += d[idx]
    if grid.max() > 0:
        grid /= grid.max()
    return grid


def _smooth_map(att_map: np.ndarray, ksize: int = 21) -> np.ndarray:
    """Apply a broad Gaussian blur so INTER_AREA downsampling matches GRADIA."""
    if ksize % 2 == 0 or ksize <= 1:
        return att_map
    blurred = cv2.GaussianBlur(att_map, (ksize, ksize), 0)
    if blurred.max() > 0:
        blurred = blurred / blurred.max()
    return blurred.astype(np.float32)


def export_split(ds: EGDCXRDataset, split_name: str, output_root: Path, classes: Sequence[str]) -> None:
    """Write images + attention maps for a dataset split."""
    split_dir = output_root / split_name
    class_dirs = []
    for idx, cls in enumerate(classes):
        cls_dir = split_dir / f"{idx:02d}_{cls}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        class_dirs.append(cls_dir)

    print(f"[{split_name}] exporting {len(ds)} samples...")
    for sample in ds:
        label_info = sample["labels"]
        idx = int(label_info["single_index"])
        if idx < 0:
            # Ambiguous / multi-label samples are skipped (matches training behavior)
            continue
        cls_dir = class_dirs[idx]
        dicom_id = sample["dicom_id"]
        base_path = cls_dir / f"{dicom_id}"

        # Save RGB image (uint8) for ImageFolder
        img = sample["image"]
        img_np = img.numpy()
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=0)
        if img_np.shape[0] == 1:
            img_np = np.repeat(img_np, 3, axis=0)
        elif img_np.shape[0] == 3:
            pass
        else:
            raise ValueError(f"Unexpected image shape {img_np.shape} for {dicom_id}")
        img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        img_hw3 = np.transpose(img_uint8, (1, 2, 0))
        cv2.imwrite(str(base_path.with_suffix(".png")), cv2.cvtColor(img_hw3, cv2.COLOR_RGB2BGR))

        # Save smoothed attention map
        fx = sample["fixations"]
        att_224 = _make_gaze_map(224, 224, fx["xy"], fx["dwell"])
        att_224 = _smooth_map(att_224)
        np.save(base_path.with_suffix(".npy"), att_224.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("runs/gradia_preprocessed"),
                    help="Destination root for GRADIA-style ImageFolder output.")
    args = ap.parse_args()

    cfg = Cfg(args.config)

    root = Path(cfg.get("input_path", "gaze_raw"))
    seg = Path(cfg.get("input_path", "segmentation_dir"))
    transcripts = Path(cfg.get("input_path", "transcripts_dir", default=seg))
    dicom_root = Path(cfg.get("input_path", "dicom_raw"))

    classes = cfg.get("train", "classes", default=["CHF", "pneumonia", "Normal"])
    max_fix = int(cfg.get("train", "max_fixations", default=8))

    train_ids, val_ids, test_ids = build_splits_ids(cfg)

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    for split_name, ids in splits.items():
        ds = EGDCXRDataset(
            root=root,
            seg_path=seg,
            transcripts_path=transcripts,
            dicom_root=dicom_root,
            max_fixations=max_fix,
            case_ids=ids,
            classes=classes,
        )
        export_split(ds, split_name, output_root, classes)

    meta = {"classes": classes}
    (output_root / "classes.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved class metadata → {output_root / 'classes.json'}")

    print(f"GRADIA ImageFolder export complete → {output_root}")


if __name__ == "__main__":
    main()
