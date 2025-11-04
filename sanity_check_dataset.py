#!/usr/bin/env python3
"""
Inspect EGDCXRDataset and save an example to sample/.

Prints:
- Total samples in train/val/test
- One example with:
  - image tensor shape
  - number of fixations and dwell-time statistics
  - per-fixation 1-hot vectors for segmentation hits and bbox hits (first few shown)
  - transcript segments (begin/end/time if present) and text content

Saves:
- JSON dump of the example under ./sample/sample_000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Try reference-style dataset path first, then fallback to current file edg_cxr.py
try:
    from egd_cxr_dataset.datasets.edg_cxr_grid_4 import (  # type: ignore
        EGDCXRDataset,
        create_dataloader,
    )
except Exception:
    try:
        from egd_cxr_dataset.datasets.edg_cxr import (  # type: ignore
            EGDCXRDataset,
            create_dataloader,
        )
    except ModuleNotFoundError:
        # Fallback: import directly from the datasets directory if package import fails
        DATASETS_DIR = SRC / "egd_cxr_dataset" / "datasets"
        if str(DATASETS_DIR) not in sys.path:
            sys.path.insert(0, str(DATASETS_DIR))
        from edg_cxr import EGDCXRDataset, create_dataloader  # type: ignore

# Config loader: prefer project ConfigLoader if available, else fallback to SimpleConfig
try:
    from egd_cxr_dataset import ConfigLoader  # type: ignore
    _HAS_CONFIG_LOADER = True
except Exception:
    _HAS_CONFIG_LOADER = False


# ------------------------- Config loader -------------------------

def _yaml_load(path: Path) -> dict:
    import yaml  # lazy import
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class SimpleConfig:
    """Minimal config helper compatible with legacy and newer keys.

    Supports lookups like get(section, key, default=None) where section may
    be e.g. "input_path" or legacy "path".
    """

    def __init__(self, yaml_path: Path):
        self._data = _yaml_load(yaml_path)

    def get(self, section: str, key: str, default: Any | None = None) -> Any:
        sec = self._data.get(section, {}) if isinstance(self._data, dict) else {}
        if isinstance(sec, dict) and key in sec:
            return sec[key]
        return default


# ------------------------- Split helpers -------------------------

def read_split_ids(split_dir: Path, split: str) -> List[str]:
    path = split_dir / f"{split}_ids.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    ids: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            ids.append(s)
    return ids


def _resolve_fold_dir(base_dir: Path) -> Path:
    """If base_dir contains fold subdirs, pick the first one with split files."""
    if (base_dir / "train_ids.txt").exists() and (base_dir / "val_ids.txt").exists() and (base_dir / "test_ids.txt").exists():
        return base_dir
    if base_dir.exists() and base_dir.is_dir():
        for d in sorted(base_dir.iterdir()):
            if d.is_dir() and d.name.lower().startswith("fold"):
                if (d / "train_ids.txt").exists():
                    return d
    return base_dir


def _ensure_split_files(split_dir: Path, root: Path) -> Path:
    """Ensure split files exist. If not, auto-create simple splits from master_sheet.csv.

    Returns the directory that actually contains the three files (may be a fold dir).
    """
    # 1) If a fold directory exists, use it
    candidate = _resolve_fold_dir(split_dir)
    if (candidate / "train_ids.txt").exists():
        return candidate

    # 2) Auto-generate simple splits at split_dir
    split_dir.mkdir(parents=True, exist_ok=True)
    master = root / "master_sheet.csv"
    if not master.exists():
        raise FileNotFoundError(f"Missing split files and master_sheet.csv not found at {master}")

    df = pd.read_csv(master, engine="python")
    if df.empty or "dicom_id" not in df.columns:
        raise ValueError("master_sheet.csv has no rows or missing 'dicom_id' column")

    ids = [str(x).strip() for x in df["dicom_id"].tolist() if str(x).strip()]
    # dedupe preserve order
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError("No valid dicom_id entries found to create splits")

    rng = np.random.RandomState(17)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    def _write(lst: List[str], name: str) -> None:
        p = split_dir / f"{name}_ids.txt"
        payload = "\n".join(lst)
        if payload:
            payload += "\n"
        p.write_text(payload, encoding="utf-8")

    _write(train_ids, "train")
    _write(val_ids, "val")
    _write(test_ids, "test")
    print(f"Created simple splits at: {split_dir} (train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)})")
    return split_dir


# ------------------------- Serialization -------------------------

def _make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    return str(obj)


# ------------------------- Inspection -------------------------

def inspect_example(sample: Dict[str, Any]) -> None:
    dicom_id = sample.get("dicom_id", "unknown")
    print(f"\nExample dicom_id: {dicom_id}")

    # Image
    img = sample.get("image")
    if isinstance(img, torch.Tensor):
        print(f"\nimage shape: {tuple(img.shape)}")

    # Metadata-derived dimensions if available
    meta = sample.get("meta", {}) if isinstance(sample.get("meta", {}), dict) else {}
    dicom_pairs = [
        ("dicom_height", "dicom_width"),
        ("dicom_rows", "dicom_cols"),
        ("original_height", "original_width"),
        ("height", "width"),
    ]
    dicom_h = dicom_w = None
    for hk, wk in dicom_pairs:
        if hk in meta and wk in meta:
            dicom_h, dicom_w = int(meta[hk]), int(meta[wk])
            break
    seg_h = meta.get("segmentation_height")
    seg_w = meta.get("segmentation_width")
    img_h_meta = meta.get("image_height")
    img_w_meta = meta.get("image_width")
    if dicom_h and dicom_w:
        print(f"\noriginal DICOM size (H,W): {dicom_h} x {dicom_w}")
    if seg_h and seg_w:
        print(f"\nsegmentation grid size (H,W): {seg_h} x {seg_w}")
    if img_h_meta and img_w_meta:
        print(f"\nreported tensor size (H,W): {img_h_meta} x {img_w_meta}")

    # Fixations (handle batched or unbatched)
    fx = sample["fixations"]
    xy_px: torch.Tensor = fx["xy"]
    xy_norm: torch.Tensor | None = fx.get("xy_norm")
    times: torch.Tensor = fx["time"]
    dwell: torch.Tensor = fx["dwell"]
    seg_hits: torch.Tensor = fx["seg_hits"]
    box_hits: torch.Tensor = fx["box_hits"]

    # If batched: (B, T, ...) or (B, T) -> take first sample
    if xy_px.dim() == 3:
        xy_px = xy_px[0]
    if isinstance(xy_norm, torch.Tensor) and xy_norm.dim() == 3:
        xy_norm = xy_norm[0]
    if times.dim() == 2:
        times = times[0]
    if dwell.dim() == 2:
        dwell = dwell[0]
    if isinstance(seg_hits, torch.Tensor) and seg_hits.dim() == 3:
        seg_hits = seg_hits[0]
    if isinstance(box_hits, torch.Tensor) and box_hits.dim() == 3:
        box_hits = box_hits[0]

    # Respect true sequence length if provided (to avoid padded zeros)
    true_len = None
    lengths = fx.get("lengths")
    if isinstance(lengths, torch.Tensor):
        if lengths.dim() == 0:
            true_len = int(lengths.item())
        elif lengths.dim() == 1 and lengths.numel() > 0:
            true_len = int(lengths[0].item())
    elif isinstance(lengths, (int, np.integer)):
        true_len = int(lengths)

    T = int(true_len if true_len is not None else xy_px.shape[0])
    T = max(0, min(T, xy_px.shape[0]))
    xy_px = xy_px[:T]
    times = times[:T]
    dwell = dwell[:T]
    if isinstance(xy_norm, torch.Tensor):
        xy_norm = xy_norm[:T]
    if seg_hits.ndim > 0:
        seg_hits = seg_hits[:T]
    if box_hits.ndim > 0:
        box_hits = box_hits[:T]

    # Now unbatched shapes: xy_px (T,2), times (T,), dwell (T,), seg_hits (T,S), box_hits (T,Cb)
    T = int(xy_px.shape[0])
    print(f"\nfixations: count={T}")
    if T > 0 and dwell.numel() > 0:
        d_np = dwell.detach().cpu().numpy().astype(np.float32)
        print(f"\ndwell (sec): mean={d_np.mean():.4f}, std={d_np.std():.4f}, min={d_np.min():.4f}, max={d_np.max():.4f}")

    # Print raw xy, image size, and normalized xy in [0,1]
    if T > 0:
        # infer H,W from image if available
        H = W = None
        img = sample.get("image")
        if isinstance(img, torch.Tensor) and img.dim() >= 3:
            H = int(img.shape[-2])
            W = int(img.shape[-1])

        # xy_raw preview and stats
        if xy_px.dim() == 2 and xy_px.shape[1] >= 2:
            xy_raw = xy_px[:, :2].detach().cpu().tolist()
            preview_n = min(T, 16)
            print(f"\nxy_raw (first {preview_n}/{T}): {xy_raw[:preview_n]}")
            x_vals = xy_px[:, 0].detach().cpu().numpy()
            y_vals = xy_px[:, 1].detach().cpu().numpy()
            print(f"\nxy_raw stats: x[min={x_vals.min():.1f}, max={x_vals.max():.1f}], y[min={y_vals.min():.1f}, max={y_vals.max():.1f}]")
        else:
            print("\nxy_raw: unavailable or wrong shape")

        print(f"\nimage tensor size (H,W): {H} x {W}")
        print(f"\ntimes (sec) [{T}]: {times.detach().cpu().tolist()}")

        # Prefer dataset-provided xy_norm
        if isinstance(xy_norm, torch.Tensor) and xy_norm.ndim == 2 and xy_norm.shape[1] >= 2:
            xy_norm_clamped = xy_norm.detach().float().clamp(0.0, 1.0)
            preview_n = min(T, 16)
            print(f"\nxy_norm dataset [0,1] (first {preview_n}/{T}): {xy_norm_clamped[:preview_n].cpu().tolist()}")
            if H is not None and W is not None and H > 0 and W > 0:
                scale_x = float(W - 1) if W and W > 1 else 1.0
                scale_y = float(H - 1) if H and H > 1 else 1.0
                x_px_tensor = (xy_norm_clamped[:, 0] * scale_x).cpu().tolist()
                y_px_tensor = (xy_norm_clamped[:, 1] * scale_y).cpu().tolist()
                preview_n = min(T, 16)
                xy_tensor_coords = [[x_px_tensor[i], y_px_tensor[i]] for i in range(preview_n)]
                print(f"\nxy on tensor grid (first {preview_n}/{T}): {xy_tensor_coords}")
        elif xy_px.dim() == 2 and xy_px.shape[1] >= 2 and H and W and H > 0 and W > 0:
            x_all = xy_px[:, 0].detach().float()
            y_all = xy_px[:, 1].detach().float()
            x_norm = (x_all / float(W - 1)).clamp(0.0, 1.0)
            y_norm = (y_all / float(H - 1)).clamp(0.0, 1.0)
            xy_norm_list = torch.stack([x_norm, y_norm], dim=1).cpu().tolist()
            print(f"\nxy_norm approximated [0,1] by tensor size [{T}]: {xy_norm_list[:min(T, 16)]}")

        # If a reference grid (segmentation or dicom) is available, also normalize by that
        ref_h = int(seg_h) if seg_h else (int(dicom_h) if dicom_h else None)
        ref_w = int(seg_w) if seg_w else (int(dicom_w) if dicom_w else None)
        if ref_h and ref_w and xy_px.dim() == 2 and xy_px.shape[1] >= 2:
            x_norm2 = (xy_px[:, 0].detach().float() / float(max(ref_w - 1, 1))).clamp(0.0, 1.0)
            y_norm2 = (xy_px[:, 1].detach().float() / float(max(ref_h - 1, 1))).clamp(0.0, 1.0)
            xy_norm2_list = torch.stack([x_norm2, y_norm2], dim=1).cpu().tolist()
            label = "segmentation grid" if seg_h and seg_w else "DICOM"
            print(f"\n{label} size (H,W): {ref_h} x {ref_w}")
            preview_n = min(T, 16)
            print(f"\nxy_norm_{label.replace(' ', '_')} [0,1] (first {preview_n}/{T}): {xy_norm2_list[:preview_n]}")

    if T > 0:
        # First fixation details (normalized if possible)
        if isinstance(xy_norm, torch.Tensor) and xy_norm.ndim == 2 and xy_norm.shape[1] >= 2:
            x0 = float(xy_norm[0, 0].clamp(0.0, 1.0).item())
            y0 = float(xy_norm[0, 1].clamp(0.0, 1.0).item())
        elif 'H' in locals() and H and W and xy_px.shape[1] >= 2:
            x0 = float((xy_px[0, 0] / float(max(W - 1, 1))).clamp(0.0, 1.0).item())
            y0 = float((xy_px[0, 1] / float(max(H - 1, 1))).clamp(0.0, 1.0).item())
        else:
            x0 = float(xy_px[0, 0].item()) if xy_px.shape[1] > 0 else float("nan")
            y0 = float(xy_px[0, 1].item()) if xy_px.shape[1] > 1 else float("nan")
        t0 = float(times[0].item()) if times.numel() > 0 else float("nan")
        d0 = float(dwell[0].item()) if dwell.numel() > 0 else float("nan")
        print(f"\nfirst fixation: xy_norm=({x0:.3f},{y0:.3f}) time={t0:.4f}s dwell={d0:.4f}s")

        if isinstance(seg_hits, torch.Tensor) and seg_hits.numel() > 0:
            print(f"\nseg_hits shape: {tuple(seg_hits.shape)} | first: {seg_hits[0].int().tolist()}")
            seg_frac = seg_hits.detach().float().mean(dim=0).cpu().tolist()
            print(f"\nseg_hits fraction per segment (mean over T): {seg_frac}")
        else:
            print("no seg_hits available")

        if isinstance(box_hits, torch.Tensor) and box_hits.numel() > 0:
            print(f"\nbox_hits shape: {tuple(box_hits.shape)} | first: {box_hits[0].int().tolist()}")
            box_frac = box_hits.detach().float().mean(dim=0).cpu().tolist()
            print(f"\nbox_hits fraction per class (mean over T): {box_frac}")
        else:
            print("no box_hits available")

    # Transcript
    tr = sample.get("transcript", {})
    print(f"\ntranscript-raw: {tr}")
    text = tr.get("text", "") if isinstance(tr, dict) else ""
    segs = tr.get("segments", []) if isinstance(tr, dict) else []
    print(f"\ntranscript: chars={len(text)} | segments={len(segs)}")
    for i, seg in enumerate(segs[:5]):
        begin = seg.get("begin")
        end = seg.get("end")
        content = seg.get("text", "")
        print(f"  seg[{i}]: begin={begin} end={end} text='{content[:80]}'")


# ------------------------- Main -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-fixations", type=int, default=64)
    ap.add_argument("--classes", nargs="+", default=["CHF", "pneumonia", "Normal"])  # can override
    ap.add_argument("--out-dir", type=Path, default=ROOT / "sample")
    ap.add_argument("--autofix-missing-seg", action="store_true",
                    help="If no segmentation cases are found, auto-create minimal dummy masks to proceed.")
    ap.add_argument("--save-json", action="store_true",
                    help="Save inspected example JSON to --out-dir/sample_000.json")
    args = ap.parse_args()

    cfg = ConfigLoader(args.config) if _HAS_CONFIG_LOADER else SimpleConfig(args.config)

    # Splits: expect train_ids.txt, val_ids.txt, test_ids.txt under configured dir
    split_dir = cfg.get("split_files", "dir", default=ROOT / "configs" / "splits")
    split_dir = Path(split_dir)
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir

    # Ensure split files exist; fallback to auto-generation if missing
    split_dir = _ensure_split_files(split_dir, root=Path(cfg.get("input_path", "gaze_raw") or cfg.get("path", "raw")))
    train_ids = read_split_ids(split_dir, "train")
    val_ids = read_split_ids(split_dir, "val")
    test_ids = read_split_ids(split_dir, "test")

    # Build datasets (resolve paths from config; support legacy keys)
    root = cfg.get("input_path", "gaze_raw")
    if root is None:
        root = cfg.get("path", "raw")
    seg = cfg.get("input_path", "segmentation_dir")
    if seg is None:
        seg = cfg.get("path", "sampling_data", default=None)
    if seg is None:
        seg = cfg.get("path", "segmentation", default=None)
    transcripts = cfg.get("input_path", "transcripts_dir", default=seg)
    if transcripts is None:
        transcripts = cfg.get("path", "transcript", default=seg)
    dicom_root = cfg.get("input_path", "dicom_raw")
    if dicom_root is None:
        dicom_root = cfg.get("path", "dcom_raw")  # legacy key

    root = Path(root)
    seg = Path(transcripts if seg is None else seg)  # ensure not None
    transcripts = Path(transcripts if transcripts is not None else seg)
    dicom_root = Path(dicom_root) if dicom_root is not None else None

    # -------- Preflight: check modality availability and print diagnostics --------
    def _discover_fx_ids(root_dir: Path) -> set[str]:
        fx_csv = root_dir / "fixations.csv"
        if not fx_csv.exists():
            return set()
        try:
            df = pd.read_csv(fx_csv, engine="python")
            return set(df.get("DICOM_ID", pd.Series(dtype=str)).dropna().astype(str))
        except Exception:
            return set()

    def _discover_ms_ids(root_dir: Path) -> set[str]:
        ms_csv = root_dir / "master_sheet.csv"
        if not ms_csv.exists():
            return set()
        try:
            df = pd.read_csv(ms_csv, engine="python")
            return set(df.get("dicom_id", pd.Series(dtype=str)).dropna().astype(str))
        except Exception:
            return set()

    def _discover_tr_ids(path_dir: Path) -> set[str]:
        if path_dir.is_file():
            try:
                df = pd.read_csv(path_dir, engine="python")
                return set(df.get("dicom_id", pd.Series(dtype=str)).dropna().astype(str))
            except Exception:
                return set()
        ids: set[str] = set()
        if path_dir.exists():
            for p in path_dir.iterdir():
                if p.is_dir() and (p / "transcript.json").exists():
                    ids.add(p.name)
        return ids

    def _discover_seg_ids(path_dir: Path) -> set[str]:
        ids: set[str] = set()
        if not path_dir.exists():
            return ids
        if path_dir.is_dir():
            for p in path_dir.iterdir():
                if p.is_dir() and any(p.glob("*.png")):
                    ids.add(p.name)
        else:
            for f in path_dir.parent.glob("*_segs.npz"):
                ids.add(f.stem.replace("_segs", ""))
            for f in path_dir.parent.glob("*_segs.npy"):
                ids.add(f.stem.replace("_segs", ""))
        return ids

    fx_ids = _discover_fx_ids(root)
    ms_ids = _discover_ms_ids(root)
    tr_ids = _discover_tr_ids(transcripts)
    sg_ids = _discover_seg_ids(seg)

    base_ids = sorted(ms_ids & fx_ids & tr_ids & sg_ids)
    print(f"Preflight: fx={len(fx_ids)} ms={len(ms_ids)} tr={len(tr_ids)} seg={len(sg_ids)} | intersection={len(base_ids)}")

    # Optionally auto-create minimal segmentation if none exists
    if len(base_ids) == 0 and len(sg_ids) == 0 and args.autofix_missing_seg:
        auto_seg_dir = ROOT / "_auto_seg_minimal"
        auto_seg_dir.mkdir(parents=True, exist_ok=True)
        # Build seg for IDs present in ms & fx (and transcripts if any)
        candidate_ids = sorted((ms_ids & fx_ids & (tr_ids if len(tr_ids) > 0 else ms_ids)))
        # limit to a small subset to keep it light
        subset = candidate_ids[:64] if len(candidate_ids) > 64 else candidate_ids
        import numpy as _np
        import imageio.v2 as _imageio
        for cid in subset:
            d = auto_seg_dir / cid
            d.mkdir(parents=True, exist_ok=True)
            # 16x16 black mask named segment_0.png
            mask = (_np.zeros((16, 16), dtype=_np.uint8))
            _imageio.imwrite(d / "segment_0.png", mask)
        print(f"Auto-created minimal segmentation masks for {len(subset)} IDs at {auto_seg_dir}")
        seg = auto_seg_dir
        # refresh discovered seg ids
        sg_ids = set(subset)

    # If still no intersection, provide actionable message and exit early
    base_ids = sorted(ms_ids & fx_ids & tr_ids & sg_ids)
    if len(base_ids) == 0:
        print("\nNo cases with all required modalities.")
        print("- Check your config paths:")
        print(f"  root (gaze_raw/raw): {root}")
        print(f"  seg (segmentation_dir): {seg}")
        print(f"  transcripts: {transcripts}")
        print(f"  dicom_root: {dicom_root}")
        print("- Counts found:")
        print(f"  fixations.csv -> {len(fx_ids)} IDs")
        print(f"  master_sheet.csv -> {len(ms_ids)} IDs")
        print(f"  transcripts -> {len(tr_ids)} IDs (needs transcript.json per-case or a transcripts CSV)")
        print(f"  segmentation -> {len(sg_ids)} IDs (expects per-case PNGs or *_segs.(npz|npy))")
        print("- Tip: re-run with --autofix-missing-seg to generate minimal masks if only seg is missing.")
        return

    def build(case_ids: List[str]) -> EGDCXRDataset:
        return EGDCXRDataset(
            root=root,
            seg_path=seg,
            transcripts_path=transcripts,
            dicom_root=dicom_root,
            max_fixations=args.max_fixations,
            case_ids=case_ids,
            classes=args.classes,
        )

    print("Loading datasets...")
    dtr = build(train_ids)
    dval = build(val_ids)
    dtest = build(test_ids)
    print(f"Totals | train={len(dtr)} | val={len(dval)} | test={len(dtest)} | classes={args.classes}")

    # Create a small loader and take one batch
    ltr = create_dataloader(dtr, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=args.num_workers)
    batch = next(iter(ltr))

    # Take first sample from batch and normalize keys to match __getitem__ layout for printing
    bs = batch["labels"]["single_index"].shape[0]
    sample: Dict[str, Any] = {k: (v[0] if isinstance(v, torch.Tensor) and v.shape[0] == bs else v) for k, v in batch.items()}

    # Align keys similar to __getitem__ layout
    if "images" in sample:
        sample["image"] = sample.pop("images")
    if "dicom_ids" in sample:
        sample["dicom_id"] = sample.pop("dicom_ids")[0]
    if "transcripts" in sample:
        sample["transcript"] = sample.pop("transcripts")[0]

    # Bring metadata if provided by dataset/batch
    if "meta" in batch:
        meta_src = batch["meta"]
        if isinstance(meta_src, dict):
            meta_out: Dict[str, Any] = {}
            for k, v in meta_src.items():
                if isinstance(v, torch.Tensor):
                    meta_out[k] = v[0].item() if v.dim() == 1 and v.shape[0] == bs else (
                        v.item() if v.dim() == 0 else v[0].item() if v.shape[0] > 0 else None
                    )
                else:
                    meta_out[k] = v
            sample["meta"] = meta_out
        else:
            sample["meta"] = meta_src

    if "fixations" in sample and isinstance(sample["fixations"], dict):
        fx_dict = sample["fixations"]
        sample["fixations"] = {
            key: (val[0] if isinstance(val, torch.Tensor) and val.shape[0] == bs else val)
            for key, val in fx_dict.items()
        }
        lengths_val = sample["fixations"].get("lengths")
        if isinstance(lengths_val, torch.Tensor):
            if lengths_val.numel() == 1:
                sample["fixations"]["lengths"] = int(lengths_val.item())
            else:
                sample["fixations"]["lengths"] = int(lengths_val[0].item())

    inspect_example(sample)

    # Optionally save to sample/ as JSON
    if args.save_json:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "sample_000.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_make_json_serializable(sample), f, indent=2)
        print(f"\nSaved inspected example to: {out_path}")


if __name__ == "__main__":
    main()


