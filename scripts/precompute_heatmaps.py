#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from egd_cxr_dataset.datasets.edg_cxr import EGDCXRDataset
except ModuleNotFoundError:
    DATASETS_DIR = SRC / "egd_cxr_dataset" / "datasets"
    if str(DATASETS_DIR) not in sys.path:
        sys.path.insert(0, str(DATASETS_DIR))
    from edg_cxr import EGDCXRDataset  # type: ignore

from gaze_utils.heatmap_utils import (
    create_static_heatmap,
    create_temporal_heatmaps,
    create_gradia_heatmap,
    ensure_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute gaze heatmaps for temporal and UNet models.")
    parser.add_argument("--config", type=str, default="configs/train_gazegnn.yaml", help="Config path")
    parser.add_argument("--cache_dir", type=str, default="cache/heatmaps", help="Output directory for cached heatmaps")
    parser.add_argument("--max_fixations", type=int, default=None, help="Max fixation sequence length (temporal)")
    parser.add_argument("--resize", type=int, default=None, help="Image resize (temporal/unet heatmaps)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate all heatmaps even if they exist")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_datasets(cfg: Dict, args: argparse.Namespace):
    root = Path(cfg["input_path"]["gaze_raw"])
    seg = Path(cfg["input_path"]["segmentation_dir"])
    transcripts = Path(cfg["input_path"].get("transcripts_dir", seg))
    dicom_root = Path(cfg["input_path"]["dicom_raw"])

    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    resize = args.resize or int(train_cfg.get("resize", 224))
    max_fix = args.max_fixations or int(train_cfg.get("max_fixations", 10))

    split_cfg = cfg.get("split_files", {}) if cfg else {}
    default_split_dir = ROOT / "configs" / "splits" / "fold1"
    split_dir = split_cfg.get("dir", default_split_dir)
    split_dir = Path(split_dir)
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    if not split_dir.exists():
        print(f"Warning: split directory {split_dir} not found. Falling back to {default_split_dir}")
        split_dir = default_split_dir
    if not split_dir.exists():
        raise FileNotFoundError(f"Default split directory not found: {split_dir}")

    def read_ids(name: str):
        path = split_dir / f"{name}_ids.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        return [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    splits = {
        "train": read_ids("train"),
        "val": read_ids("val"),
        "test": read_ids("test"),
    }

    datasets = {}
    for split_name, ids in splits.items():
        datasets[split_name] = EGDCXRDataset(
            root=root,
            seg_path=seg,
            transcripts_path=transcripts,
            dicom_root=dicom_root,
            max_fixations=max_fix,
            case_ids=ids,
            classes=train_cfg.get("classes", ["CHF", "pneumonia", "Normal"]),
        )

    return datasets, resize, max_fix


def maybe_save(array: np.ndarray, path: Path, overwrite: bool, *, compressed: bool = False, key: str = "heatmaps"):
    if path.exists() and not overwrite:
        return
    ensure_dir(path.parent)
    if compressed:
        np.savez_compressed(path, **{key: array.astype(np.float32)})
    else:
        np.save(path, array.astype(np.float32))


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    cfg = load_config(config_path)

    datasets, resize, max_seq = build_datasets(cfg, args)

    meta_path = cache_dir / "meta.json"
    meta = {
        "config": str(config_path),
        "resize": resize,
        "max_seq": max_seq,
    }
    ensure_dir(cache_dir)

    for split, dataset in datasets.items():
        static_dir = cache_dir / "static" / split
        temporal_dir = cache_dir / "temporal" / split
        gradia_dir = cache_dir / "gradia" / split
        ensure_dir(static_dir)
        ensure_dir(temporal_dir)
        ensure_dir(gradia_dir)

        for idx in tqdm(range(len(dataset)), desc=f"Precomputing {split}", unit="case"):
            sample = dataset[idx]
            dicom_id = sample["dicom_id"]
            fx = sample["fixations"]
            xy = fx["xy"]
            dwell = fx["dwell"]

            static_heatmap = create_static_heatmap(xy, dwell, h=resize, w=resize)
            maybe_save(static_heatmap, static_dir / f"{dicom_id}.npy", args.overwrite)

            temporal_heatmap = create_temporal_heatmaps(xy, dwell, h=resize, w=resize, max_seq=max_seq)
            maybe_save(temporal_heatmap, temporal_dir / f"{dicom_id}.npz", args.overwrite, compressed=True)

            gradia_heatmap = create_gradia_heatmap(xy, dwell, h=resize, w=resize, out_size=(7, 7))
            maybe_save(gradia_heatmap, gradia_dir / f"{dicom_id}.npy", args.overwrite)

    meta["complete"] = True
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Heatmap cache ready at {cache_dir}")


if __name__ == "__main__":
    main()

