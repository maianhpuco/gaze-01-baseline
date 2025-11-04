#!/usr/bin/env python3
"""
EGD-CXR multimodal Dataset & DataLoader (gaze + image + seg + boxes + transcript).

Fix: label handling now mirrors the reference EyegazeDataset:
  • labels are MULTI-HOT in the exact order of `classes`
  • keep single_index only if exactly one positive (else -1)
  • class_counts computed from multi-hot totals
Other data modalities/fields are unchanged.
"""

from __future__ import annotations
import json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    pydicom = None  # type: ignore
    HAS_PYDICOM = False

# Threading guards (safe in multiprocess dataloaders)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Eye-tracking CSV columns
TIME_COLUMN = "Time (in secs)"
X_COLUMN    = "FPOGX"
Y_COLUMN    = "FPOGY"
DURATION_COLUMN = "FPOGD"  # seconds

class Logger:
    @staticmethod
    def info(msg: str) -> None:  print(msg)
    @staticmethod
    def error(msg: str) -> None: print(f"Warning: {msg}")

# -------------------------- (optional) diagnoses helper --------------------------
@dataclass(frozen=True)
class LabelSchema:
    class_columns: List[str]

class LabelProcessor:
    """Only used to read free-text diagnoses from master_sheet (dx* columns)."""
    def __init__(self, master_sheet_csv: Path):
        self.master_sheet_csv = Path(master_sheet_csv).expanduser()
        if not self.master_sheet_csv.exists():
            raise FileNotFoundError(f"master_sheet.csv not found: {self.master_sheet_csv}")
        self.df = pd.read_csv(self.master_sheet_csv, engine="python")
        if self.df.empty:
            raise ValueError("master_sheet.csv has no rows")

    def _get_row(self, case_id: str) -> pd.Series:
        mask = self.df["dicom_id"] == case_id
        if not mask.any():
            raise ValueError(f"dicom_id {case_id} not found in {self.master_sheet_csv}")
        return self.df.loc[mask].iloc[0]

    def diagnoses(self, case_id: str) -> Tuple[Optional[str], List[str]]:
        row = self._get_row(case_id)
        dx_cols = [c for c in row.index if c.startswith("dx") and not c.endswith("_icd")]
        diags: List[str] = []
        for col in sorted(dx_cols, key=lambda n: int(n[2:]) if n[2:].isdigit() else 0):
            v = row[col]
            if isinstance(v, str) and v.strip():
                diags.append(v.strip())
        return (diags[0] if diags else None), diags

# -------------------------- boxes --------------------------
@dataclass(frozen=True)
class BoxRow:
    x1: int; y1: int; x2: int; y2: int
    cls_id: int; cls_name: str

# ========================== DATASET ==========================
class EGDCXRDataset(Dataset):
    """
    Multimodal dataset (gaze + image + seg + boxes + transcript + labels) with ROI prompt text.

    LABELS (fixed):
      • Expect `classes` to name columns in master_sheet.csv (e.g., ['Normal','CHF','pneumonia'])
      • Per case, build multi-hot tensor in exactly that order.
      • Keep single_index when sum == 1 else -1 (for any legacy use).
    """
    def __init__(
        self,
        root: Path,
        seg_path: Path,
        transcripts_path: Optional[Path] = None,
        *,
        dicom_root: Optional[Path] = None,
        max_fixations: Optional[int] = None,
        case_ids: Optional[Sequence[str]] = None,
        classes: Sequence[str] = ("Normal", "CHF", "pneumonia"),  # match reference ordering
        drop_unlabelled: bool = False,  # keep rows where sum(labels)==0 to mirror reference
    ):
        self.root = Path(root).expanduser()
        if not self.root.exists(): raise FileNotFoundError(f"Gaze dataset root not found: {self.root}")

        self.fixations_csv     = self.root / "fixations.csv"
        self.master_sheet_csv  = self.root / "master_sheet.csv"
        self.bounding_boxes_csv= self.root / "bounding_boxes.csv"

        self.seg_path = Path(seg_path).expanduser()
        self.transcripts_path = Path(transcripts_path or seg_path).expanduser()
        self.dicom_root = Path(dicom_root).expanduser() if dicom_root else None

        for p in [self.fixations_csv, self.master_sheet_csv, self.seg_path, self.transcripts_path]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing path: {p}")
        if self.bounding_boxes_csv.exists():
            pass
        else:
            Logger.error("bounding_boxes.csv not found — continuing with empty boxes.")
        if self.dicom_root and not self.dicom_root.exists():
            raise FileNotFoundError(f"DICOM root not found: {self.dicom_root}")

        self.max_fix = max_fixations
        # exact order matters for label vector!
        self.single_classes = tuple(str(c).strip() for c in classes if str(c).strip())
        if not self.single_classes:
            raise ValueError("`classes` must contain at least one non-empty name.")
        self.drop_unlabelled = drop_unlabelled

        # load tables
        self.ms_df = pd.read_csv(self.master_sheet_csv, engine="python")
        if self.ms_df.empty: raise ValueError("master_sheet.csv empty")
        self.fx_df = pd.read_csv(self.fixations_csv, engine="python")
        self.bb_df = pd.read_csv(self.bounding_boxes_csv, engine="python") if self.bounding_boxes_csv.exists() else pd.DataFrame()

        self.transcripts_mode = "csv" if self.transcripts_path.is_file() else "directory"
        self.tr_df = pd.read_csv(self.transcripts_path, engine="python") if self.transcripts_mode == "csv" else None

        self.seg_mode = "directory" if self.seg_path.is_dir() else "arrays"
        if self.seg_mode not in {"directory", "arrays"}:
            raise ValueError("Unsupported segmentation storage format.")

        self.region_names: List[str] = self._discover_region_names() if self.seg_mode == "directory" else []
        self.box_label_to_idx = self._build_box_label_mapping()
        self.box_class_names = [n for n, _ in sorted(self.box_label_to_idx.items(), key=lambda kv: kv[1])]
        self.num_box_classes = len(self.box_class_names)
        self.num_segments: Optional[int] = len(self.region_names) if self.region_names else None

        # case intersection: require all modalities present (as before)
        fx_cases = set(self.fx_df["DICOM_ID"].dropna().astype(str))
        ms_cases = set(self.ms_df["dicom_id"].dropna().astype(str))
        transcript_cases = self._discover_transcript_case_ids()
        seg_cases = self._discover_segmentation_case_ids()
        base_cases = sorted(ms_cases & fx_cases & transcript_cases & seg_cases)

        if case_ids is not None:
            requested = set(case_ids)
            missing = requested - set(base_cases)
            if missing:
                Logger.error(f"{len(missing)} requested IDs missing modalities; skipping.")
            base_cases = [cid for cid in base_cases if cid in requested]

        if not base_cases:
            raise ValueError("No cases with all required modalities.")

        # diagnoses helper (for optional metadata)
        self.dx_helper = LabelProcessor(self.master_sheet_csv)

        # --------- FIXED LABEL LOGIC (multi-hot, reference-compatible) ---------
        # Build multi-hot labels in the exact order of self.single_classes
        self._labels_multihot: Dict[str, torch.Tensor] = {}
        filtered_cases: List[str] = []

        for cid in base_cases:
            row = self.ms_df[self.ms_df["dicom_id"] == cid]
            if row.empty: continue
            r = row.iloc[0]
            vec = []
            for cls in self.single_classes:
                v = r.get(cls, 0)
                try:
                    iv = int(v)
                except Exception:
                    # Strict like reference: non-numeric treated as 0
                    iv = 0
                iv = 1 if iv == 1 else 0
                vec.append(iv)
            vec_t = torch.tensor(vec, dtype=torch.float32)

            if self.drop_unlabelled and vec_t.sum().item() == 0:
                # optionally drop samples with no positive labels
                continue

            self._labels_multihot[cid] = vec_t
            filtered_cases.append(cid)

        if not filtered_cases:
            raise ValueError("No cases with valid multi-hot labels (after drop_unlabelled filtering).")

        self.case_ids = filtered_cases

        # class_counts for info/weighting: sum of multi-hot across cases
        if len(self.case_ids) > 0:
            mat = torch.stack([self._labels_multihot[cid] for cid in self.case_ids], dim=0)  # [N,C]
            self._class_counts = mat.sum(dim=0)  # [C]
        else:
            self._class_counts = torch.zeros(len(self.single_classes), dtype=torch.float32)

        Logger.info(
            f"EGDCXRDataset ready: {len(self.case_ids)} cases | "
            f"classes={list(self.single_classes)} | "
            f"class_counts={self._class_counts.tolist()} | "
            f"regions={len(self.region_names) if self.region_names else 'n/a'} | "
            f"bbox_classes={len(self.box_label_to_idx)}"
        )

    # ----- dataset props -----
    def __len__(self) -> int:                 return len(self.case_ids)
    @property
    def class_names(self) -> Tuple[str, ...]: return self.single_classes
    @property
    def class_counts(self) -> torch.Tensor:   return self._class_counts.clone()

    def class_weights(self) -> torch.Tensor:
        # inverse-frequency normalized
        counts = self.class_counts.to(torch.float32).clamp_min(1.0)
        total = counts.sum()
        w = total / counts
        return w / w.mean().clamp_min(1e-6)

    def sample_weights(self) -> torch.Tensor:
        # for BCE setups, per-sample weighting is typically unnecessary; return ones
        return torch.ones(len(self.case_ids), dtype=torch.float32)

    # ----- discovery helpers -----
    def _discover_region_names(self) -> List[str]:
        names: set[str] = set()
        if not self.seg_path.is_dir(): return []
        for case_dir in self.seg_path.iterdir():
            if not case_dir.is_dir(): continue
            for png in case_dir.glob("*.png"): names.add(png.stem)
        return sorted(list(names))

    def _discover_transcript_case_ids(self) -> set[str]:
        if self.transcripts_mode == "csv":
            return set(self.tr_df["dicom_id"].dropna().astype(str))
        cases: set[str] = set()
        for p in self.transcripts_path.iterdir():
            if p.is_dir() and (p / "transcript.json").exists():
                cases.add(p.name)
        return cases

    def _discover_segmentation_case_ids(self) -> set[str]:
        cases: set[str] = set()
        if self.seg_mode == "directory":
            for p in self.seg_path.iterdir():
                if p.is_dir() and any(p.glob("*.png")):
                    cases.add(p.name)
        else:
            for f in self.seg_path.glob("*_segs.npz"): cases.add(f.stem.replace("_segs", ""))
            for f in self.seg_path.glob("*_segs.npy"): cases.add(f.stem.replace("_segs", ""))
        return cases

    def _build_box_label_mapping(self) -> Dict[str, int]:
        if self.bb_df.empty: return {}
        names = sorted(self.bb_df["bbox_name"].dropna().astype(str).unique())
        return {n: i for i, n in enumerate(names)}

    # ----- modality loaders -----
    def _load_seg_masks(self, dicom_id: str) -> np.ndarray:
        if self.seg_mode == "arrays":
            npz = self.seg_path / f"{dicom_id}_segs.npz"
            npy = self.seg_path / f"{dicom_id}_segs.npy"
            if   npz.exists(): arr = np.load(npz)["masks"]
            elif npy.exists(): arr = np.load(npy)
            else: raise FileNotFoundError(f"Segmentation array not found for {dicom_id}")
            return (arr > 0.5).astype(np.uint8)

        case_dir = self.seg_path / dicom_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Segmentation directory not found for {dicom_id}")

        region_names = self.region_names or sorted([p.stem for p in case_dir.glob("*.png")])
        if not region_names: raise ValueError(f"No segmentation PNGs for {dicom_id}")

        ref = None
        for n in region_names:
            p = case_dir / f"{n}.png"
            if p.exists():
                ref = imageio.imread(p); break
        if ref is None: raise ValueError(f"No segmentation PNGs for {dicom_id}")

        h, w = ref.shape[:2]
        masks = np.zeros((len(region_names), h, w), dtype=np.uint8)
        for i, n in enumerate(region_names):
            p = case_dir / f"{n}.png"
            if not p.exists(): continue
            img = imageio.imread(p)
            mask = img.max(axis=2) > 0 if img.ndim == 3 else img > 0
            masks[i] = mask.astype(np.uint8)
        bg = (masks.sum(axis=0, keepdims=True) == 0).astype(np.uint8)
        return np.concatenate([masks, bg], axis=0)

    def _load_dicom_image(self, dicom_id: str) -> Optional[np.ndarray]:
        if self.dicom_root is None: return None
        path = self.dicom_root / f"{dicom_id}.dcm"
        if not path.exists(): return None
        try:
            if HAS_PYDICOM:
                ds = pydicom.dcmread(str(path))
                arr = ds.pixel_array.astype(np.float32)
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                arr = arr * slope + intercept

                def _win(tag: str) -> Optional[float]:
                    v = getattr(ds, tag, None)
                    if v is None: return None
                    try:
                        return float(v[0]) if isinstance(v, (list, tuple)) else float(v)
                    except Exception:
                        return None
                center = _win("WindowCenter"); width = _win("WindowWidth")
                if center is not None and width is not None and width > 0:
                    lo = center - width/2.0; hi = center + width/2.0
                    arr = np.clip(arr, lo, hi); arr = (arr - lo) / max((hi - lo), 1e-6)
                else:
                    arr -= arr.min(); mx = arr.max();  arr = arr / mx if mx > 0 else arr
            else:
                arr = imageio.imread(path)
                if arr.ndim == 3: arr = arr[..., 0]
                arr = arr.astype(np.float32)
                arr -= arr.min(); mx = arr.max();  arr = arr / mx if mx > 0 else arr
        except Exception:
            return None
        if arr.ndim == 3: arr = arr[..., 0]
        return arr.astype(np.float32)

    def _load_boxes(self, dicom_id: str) -> List[BoxRow]:
        if self.bb_df.empty: return []
        rows = self.bb_df[self.bb_df["dicom_id"] == dicom_id]
        out: List[BoxRow] = []
        for _, r in rows.iterrows():
            name = str(r["bbox_name"])
            cid = self.box_label_to_idx.get(name, -1)
            out.append(BoxRow(
                x1=int(round(float(r["x1"]))), y1=int(round(float(r["y1"]))),
                x2=int(round(float(r["x2"]))), y2=int(round(float(r["y2"]))),
                cls_id=cid, cls_name=name,
            ))
        return out

    def _load_transcript(self, dicom_id: str) -> Dict[str, Any]:
        if self.transcripts_mode == "csv":
            rows = self.tr_df[self.tr_df["dicom_id"] == dicom_id]
            if rows.empty: return {"text": "", "segments": []}
            txt = str(rows.iloc[0].get("transcript", "") or "")
            return {"text": txt, "segments": []}
        case_dir = self.transcripts_path / dicom_id
        tf = case_dir / "transcript.json"
        if tf.exists():
            data = json.loads(tf.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                txt = str(data.get("transcript") or data.get("full_text") or "").strip()
                segs = data.get("segments") or []
                return {"text": txt, "segments": segs}
        return {"text": "", "segments": []}

    def _load_fixations(self, dicom_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        df = self.fx_df[self.fx_df["DICOM_ID"] == dicom_id].copy()
        if df.empty: raise ValueError(f"No fixations for {dicom_id}")
        df = df[
            df[X_COLUMN].between(0.0, 1.0) &
            df[Y_COLUMN].between(0.0, 1.0) &
            df[DURATION_COLUMN].notna() &
            (df[DURATION_COLUMN] > 0)
        ].copy()
        if df.empty: raise ValueError(f"Invalid fixations for {dicom_id}")
        if "CNT" in df.columns:
            df.sort_values(by=[TIME_COLUMN, "CNT"], inplace=True, kind="mergesort")
        else:
            df.sort_values(by=[TIME_COLUMN], inplace=True, kind="mergesort")
        if self.max_fix is not None:
            df = df.iloc[: self.max_fix]
        xy_norm = df[[X_COLUMN, Y_COLUMN]].to_numpy(dtype=np.float32)
        dwell   = df[DURATION_COLUMN].to_numpy(dtype=np.float32)   # seconds
        times   = df[TIME_COLUMN].to_numpy(dtype=np.float32)
        if not np.isfinite(xy_norm).all() or not np.isfinite(dwell).all() or not np.isfinite(times).all():
            mask = np.isfinite(xy_norm).all(axis=1) & np.isfinite(dwell) & np.isfinite(times)
            xy_norm = xy_norm[mask]; dwell = dwell[mask]; times = times[mask]
        return times, xy_norm, dwell

    # ----- sample assembly -----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dicom_id = self.case_ids[idx]

        # segmentation
        seg_masks_np = self._load_seg_masks(dicom_id)
        if self.num_segments is None:
            self.num_segments = seg_masks_np.shape[0] - 1 if seg_masks_np.shape[0] > 1 else seg_masks_np.shape[0]
            if not self.region_names:
                self.region_names = [f"segment_{i}" for i in range(self.num_segments)]
        num_segments = self.num_segments or 0
        segments_np = seg_masks_np[:num_segments] if num_segments > 0 else np.zeros((0, *seg_masks_np.shape[1:]), dtype=np.uint8)

        # other modalities
        boxes = self._load_boxes(dicom_id)
        transcript_payload = self._load_transcript(dicom_id)
        final_dx, diagnoses = self.dx_helper.diagnoses(dicom_id)
        times_sec, xy_norm, dwell = self._load_fixations(dicom_id)

        # gaze → pixel
        height, width = seg_masks_np.shape[1:]
        if xy_norm.size == 0:
            xy_px = np.zeros((0, 2), dtype=np.float32)
        else:
            xy_px = np.stack([xy_norm[:, 0]*(width-1), xy_norm[:, 1]*(height-1)], axis=1).astype(np.float32)

        # ROI hits (per fixation)
        T = xy_px.shape[0]
        seg_hits = np.zeros((T, num_segments), dtype=np.float32)
        if num_segments > 0 and T > 0:
            xs = np.clip(np.round(xy_px[:, 0]).astype(int), 0, width-1)
            ys = np.clip(np.round(xy_px[:, 1]).astype(int), 0, height-1)
            hits = segments_np[:, ys, xs] > 0
            seg_hits = hits.T.astype(np.float32)

        num_box_classes = max(0, self.num_box_classes)
        box_hits = np.zeros((T, num_box_classes), dtype=np.float32)
        if num_box_classes > 0 and T > 0:
            xs_int = np.clip(np.round(xy_px[:, 0]).astype(int), 0, width-1)
            ys_int = np.clip(np.round(xy_px[:, 1]).astype(int), 0, height-1)
            for t in range(T):
                x, y = int(xs_int[t]), int(ys_int[t])
                for box in boxes:
                    cid = box.cls_id
                    if 0 <= cid < num_box_classes and box.x1 <= x < box.x2 and box.y1 <= y < box.y2:
                        box_hits[t, cid] = 1.0

        # image (224,224)
        img_arr = self._load_dicom_image(dicom_id)
        if img_arr is None:
            image_tensor = torch.zeros(1, 224, 224, dtype=torch.float32)
        else:
            img_tensor = torch.from_numpy(img_arr).unsqueeze(0).float()
            image_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(224,224), mode="bilinear", align_corners=False).squeeze(0)

        # presence vectors (global)
        segment_names = self.region_names or [f"segment_{i}" for i in range(num_segments)]
        box_class_names = self.box_class_names or [f"class_{i}" for i in range(self.num_box_classes)]

        if num_segments > 0:
            seg_presence_np = (segments_np.reshape(num_segments, -1).sum(axis=1) > 0).astype(np.float32)
            segment_presence = torch.from_numpy(seg_presence_np)
        else:
            segment_presence = torch.zeros(0, dtype=torch.float32)

        box_presence = torch.zeros(self.num_box_classes, dtype=torch.float32)
        for b in boxes:
            if 0 <= b.cls_id < self.num_box_classes:
                box_presence[b.cls_id] = 1.0

        # per-fixation ROI prompt strings
        roi_texts: List[str] = []
        for t in range(T):
            seg_idxs = np.where(seg_hits[t] > 0.5)[0].tolist()
            box_idxs = np.where(box_hits[t] > 0.5)[0].tolist()
            seg_parts = [segment_names[i].replace("_"," ") for i in seg_idxs] if seg_idxs else []
            box_parts = [box_class_names[i].replace("_"," ") for i in box_idxs] if box_idxs else []
            parts: List[str] = []
            if seg_parts: parts.append("examining " + "; ".join(seg_parts))
            if box_parts: parts.append("finding " + "; ".join(box_parts))
            if not parts: parts = ["examining chest"]
            roi_texts.append("; ".join(parts))

        # --------- labels (multi-hot) ----------
        multihot = self._labels_multihot[dicom_id]                  # [C]
        num_pos = int(multihot.sum().item())
        if num_pos == 1:
            single_index = int(torch.nonzero(multihot, as_tuple=False)[0].item())
            single_name  = self.single_classes[single_index]
        else:
            single_index = -1
            single_name  = "Ambiguous" if num_pos > 1 else "Unknown"

        sample = {
            "dicom_id": dicom_id,
            "image": image_tensor,  # [1,224,224]
            "fixations": {
                "xy": torch.from_numpy(xy_px.astype(np.float32)),
                "xy_norm": torch.from_numpy(xy_norm.astype(np.float32)),
                "time": torch.from_numpy(times_sec.astype(np.float32)),
                "dwell": torch.from_numpy(dwell.astype(np.float32)),
                "seg_hits": torch.from_numpy(seg_hits).float(),
                "box_hits": torch.from_numpy(box_hits).float(),
                "roi_texts": roi_texts,
            },
            "segment_presence": segment_presence.float(),
            "box_presence": box_presence.float(),
            "transcript": transcript_payload,
            # label package (reference-compatible)
            "labels": {
                "multihot": multihot.clone(),                    # [C] float {0,1}
                "single_index": torch.tensor(single_index, dtype=torch.long),
                "single_name": single_name,
                "single_class_names": self.single_classes,
                # free-text diagnoses from dx* if present
                "final_diagnosis": final_dx,
                "diagnoses": diagnoses,
            },
            "boxes": boxes,
            "meta": {
                "segment_names": segment_names,
                "box_class_names": box_class_names,
                "segmentation_height": int(height),
                "segmentation_width": int(width),
                "image_height": int(image_tensor.shape[-2]),
                "image_width": int(image_tensor.shape[-1]),
            },
        }
        return sample

# ========================== COLLATE ==========================
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    dicom_ids = [it["dicom_id"] for it in batch]
    lengths = torch.tensor([it["fixations"]["xy"].shape[0] for it in batch], dtype=torch.long)

    xy      = pad_sequence([it["fixations"]["xy"]       for it in batch], batch_first=True)
    xy_norm = pad_sequence([it["fixations"]["xy_norm"]  for it in batch], batch_first=True)
    dwell   = pad_sequence([it["fixations"]["dwell"]    for it in batch], batch_first=True)
    times   = pad_sequence([it["fixations"]["time"]     for it in batch], batch_first=True)
    seg_h   = pad_sequence([it["fixations"]["seg_hits"] for it in batch], batch_first=True)
    box_h   = pad_sequence([it["fixations"]["box_hits"] for it in batch], batch_first=True)

    roi_texts = [it["fixations"]["roi_texts"] for it in batch]  # list[list[str]], ragged

    images   = torch.stack([it["image"] for it in batch], dim=0)
    seg_pres = torch.stack([it["segment_presence"] for it in batch], dim=0)
    box_pres = torch.stack([it["box_presence"] for it in batch], dim=0)
    transcripts = [it["transcript"] for it in batch]

    # ---- NEW: multi-label tensor [B,C] exactly like reference EyegazeDataset ----
    labels_multihot = torch.stack([it["labels"]["multihot"] for it in batch], dim=0)  # float {0,1}
    single_idx = torch.stack([it["labels"]["single_index"] for it in batch], dim=0)
    single_nam = [it["labels"]["single_name"] for it in batch]
    class_names = list(batch[0]["labels"]["single_class_names"])

    labels_dict = {
        "multihot": labels_multihot,               # use with BCEWithLogitsLoss
        "single_index": single_idx,                # -1 if ambiguous/unknown
        "single_name": single_nam,
        "single_class_names": class_names,
        "final_diagnosis": [it["labels"]["final_diagnosis"] for it in batch],
        "diagnoses": [it["labels"]["diagnoses"] for it in batch],
    }

    meta = {
        "segment_names": batch[0]["meta"]["segment_names"],
        "box_class_names": batch[0]["meta"]["box_class_names"],
        "segmentation_height": batch[0]["meta"].get("segmentation_height"),
        "segmentation_width": batch[0]["meta"].get("segmentation_width"),
        "image_height": batch[0]["meta"].get("image_height"),
        "image_width": batch[0]["meta"].get("image_width"),
    }

    return {
        "dicom_ids": dicom_ids,
        "images": images,
        "segment_presence": seg_pres,
        "box_presence": box_pres,
        "fixations": {
            "xy": xy,
            "xy_norm": xy_norm,
            "dwell": dwell,
            "time": times,
            "seg_hits": seg_h,
            "box_hits": box_h,
            "roi_texts": roi_texts,
            "lengths": lengths,
        },
        "transcripts": transcripts,
        "labels": labels_dict,
        "meta": meta,
        "boxes": [it["boxes"] for it in batch],
    }

# ========================== DATALOADER ==========================
def create_dataloader(
    dataset: EGDCXRDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    sampler=None,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )


# ========================== SPLIT HELPERS (FOLD SUPPORT) ==========================
def _read_ids_file(path: Path) -> List[str]:
    ids: List[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            ids.append(s)
    return ids


def load_fold_ids(splits_root: Path, fold: int = 1) -> Dict[str, List[str]]:
    """
    Load train/val/test ID lists for a given fold.

    Expect structure: <splits_root>/fold{fold}/{train_ids.txt,val_ids.txt,test_ids.txt}
    """
    fold_dir = Path(splits_root) / f"fold{fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
    ids = {
        "train": _read_ids_file(fold_dir / "train_ids.txt"),
        "val": _read_ids_file(fold_dir / "val_ids.txt"),
        "test": _read_ids_file(fold_dir / "test_ids.txt"),
    }
    return ids


def build_datasets_for_fold(
    *,
    root: Path,
    seg_path: Path,
    transcripts_path: Optional[Path],
    dicom_root: Optional[Path],
    splits_root: Path,
    fold: int = 1,
    classes: Sequence[str] = ("Normal", "CHF", "pneumonia"),
    max_fixations: Optional[int] = None,
) -> Tuple[EGDCXRDataset, EGDCXRDataset, EGDCXRDataset]:
    """
    Convenience: construct (train, val, test) EGDCXRDataset instances for a chosen fold.
    """
    split_ids = load_fold_ids(Path(splits_root), fold)
    def _mk(ids: List[str]) -> EGDCXRDataset:
        return EGDCXRDataset(
            root=root,
            seg_path=seg_path,
            transcripts_path=transcripts_path,
            dicom_root=dicom_root,
            max_fixations=max_fixations,
            case_ids=ids,
            classes=classes,
        )
    return _mk(split_ids["train"]), _mk(split_ids["val"]), _mk(split_ids["test"])
