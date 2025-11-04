#!/usr/bin/env python3
"""
Generate train/val/test K-fold splits for EGD-CXR case IDs using the rewritten dataset.

- Optionally holds out a fixed test set by --test ratio (stratified if --stratify).
- Builds K folds on the remaining (train+val) IDs: for each fold i,
  val = fold_i, train = union(other folds), test = fixed holdout (may be empty if --test=0).
- Writes:
    <output_dir>/fold1/{train_ids.txt,val_ids.txt,test_ids.txt}
    ...
    <output_dir>/foldK/{...}
    <output_dir>/summary.json

Examples:
  # 5-fold CV, no external test (test files empty), stratified
  python create_splits.py --config-path configs/data_egd_cxr_single_label.yaml \
      --output-dir configs/splits --folds 5 --test 0.0 --seed 42 --stratify --print-summary

  # 5-fold CV with 20% fixed test holdout, stratified
  python create_splits.py --config-path configs/data_egd_cxr_single_label.yaml \
      --output-dir configs/splits --folds 5 --test 0.2 --seed 42 --stratify --print-summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader  # noqa: E402
from egd_cxr_dataset.split import SplitConfig  # noqa: E402  # (kept for compatibility if you import elsewhere)

DEFAULT_CONFIG = ROOT / "configs" / "data_egd-cxr.yaml"
DEFAULT_OUTPUT = ROOT / "configs" / "splits"


# ------------------------- CLI -------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic K-fold train/val/test splits for EGD-CXR IDs."
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG,
                        help=f"Path to dataset configuration YAML (default: {DEFAULT_CONFIG}).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Directory where split files will be written (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of folds K for cross-validation (default: 5).")
    parser.add_argument("--train", type=float, default=0.7,
                        help="(Unused in K-fold mode except to compute holdout if you wish).")
    parser.add_argument("--val", type=float, default=0.1,
                        help="(Unused in K-fold mode; val is 1/K of train+val pool).")
    parser.add_argument("--test", type=float, default=0.2,
                        help="Proportion of IDs assigned to a fixed test holdout (default: 0.2). "
                             "Set 0.0 to skip external test.")
    parser.add_argument("--seed", type=int, default=17,
                        help="Random seed used for shuffling IDs (default: 17).")
    parser.add_argument("--print-summary", action="store_true",
                        help="Print JSON summary of counts and file paths after writing splits.")
    parser.add_argument("--stratify", action="store_true",
                        help="Stratify by class to balance distributions.")
    return parser.parse_args()


# --------------------- Dataset loading ---------------------

def load_dataset_with_classification(config_path: Path) -> Tuple[Dict[str, Dict[str, int]], List[str], Tuple[str, ...]]:
    """Read labels from master_sheet.csv and return (labels_by_id, case_ids, class_names)."""
    config_loader = ConfigLoader(config_path)

    # Prefer new schema: input_path.*, but support legacy schema: path.* from existing configs
    gaze_raw = config_loader.get("input_path", "gaze_raw")
    if gaze_raw is None:
        gaze_raw = config_loader.get("path", "raw")
    root = Path(gaze_raw)


    seg_dir = config_loader.get("input_path", "segmentation_dir")
    if seg_dir is None:
        # best-effort fallbacks: try sampling_data, else use a non-existing default under root
        seg_dir = config_loader.get("path", "sampling_data", default=root / "segmentation")
    seg_path = Path(seg_dir)

    transcripts_dir = config_loader.get("input_path", "transcripts_dir")
    if transcripts_dir is None:
        transcripts_dir = config_loader.get("path", "transcript", default=seg_dir)
    transcripts_path = Path(transcripts_dir) if transcripts_dir is not None else Path(seg_dir)

    dicom_dir = config_loader.get("input_path", "dicom_raw")
    if dicom_dir is None:
        # legacy key typo: dcom_raw
        dicom_dir = config_loader.get("path", "dcom_raw")
    dicom_root = Path(dicom_dir) if dicom_dir is not None else None

    master_sheet_csv = root / "master_sheet.csv"
    if not master_sheet_csv.exists():
        raise FileNotFoundError(f"master_sheet.csv not found at {master_sheet_csv}")

    df = pd.read_csv(master_sheet_csv, engine="python")
    if df.empty:
        raise ValueError(f"No rows found in {master_sheet_csv}")

    class_names: Tuple[str, ...] = ("Normal", "CHF", "pneumonia")
    missing_cols = [c for c in class_names if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing label columns in master_sheet.csv: {missing_cols}")

    labels_by_id: Dict[str, Dict[str, int]] = {}
    case_ids: List[str] = []
    for _, row in df.iterrows():
        dicom_id = str(row.get("dicom_id", "")).strip()
        if not dicom_id:
            continue
        case_ids.append(dicom_id)
        entry: Dict[str, int] = {}
        for cls in class_names:
            v = row.get(cls, 0)
            try:
                iv = int(v)
            except Exception:
                iv = 0
            entry[cls] = 1 if iv == 1 else 0
        labels_by_id[dicom_id] = entry

    return labels_by_id, case_ids, class_names


# ---------------------- Split helpers ----------------------

def _class_to_ids(case_ids: List[str], labels_by_id: Dict[str, Dict[str, int]], class_names: Tuple[str, ...]) -> Dict[str, List[str]]:
    """Map class name -> list of dicom_id using multi-hot labels (may appear in multiple classes)."""
    class_to_ids: Dict[str, List[str]] = {name: [] for name in class_names}
    seen_per_class = {k: set() for k in class_names}
    for dicom_id in case_ids:
        entry = labels_by_id.get(dicom_id, {})
        for cls_name in class_names:
            is_pos = bool(int(entry.get(cls_name, 0)))
            if is_pos and dicom_id not in seen_per_class[cls_name]:
                seen_per_class[cls_name].add(dicom_id)
                class_to_ids[cls_name].append(dicom_id)
    return class_to_ids


def _pick_test_holdout(ids_or_class_ids, test_ratio: float, seed: int, stratify: bool) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Return (test_ids, pool_after_test_by_class)

    - If stratify: ids_or_class_ids is dict[class -> ids]. We select per-class test and return remaining per-class pools.
    - If not stratify: ids_or_class_ids is list of ids. We select global test and return remaining under a single key "_ALL".
    """
    np.random.seed(seed)
    if stratify:
        test_ids: List[str] = []
        remaining_by_class: Dict[str, List[str]] = {}
        for cls, ids in ids_or_class_ids.items():
            ids = ids.copy()
            np.random.shuffle(ids)
            n_test = int(len(ids) * test_ratio)
            cls_test = ids[:n_test]
            cls_rem = ids[n_test:]
            test_ids.extend(cls_test)
            remaining_by_class[cls] = cls_rem
        return test_ids, remaining_by_class
    else:
        all_ids = ids_or_class_ids.copy()
        np.random.shuffle(all_ids)
        total = len(all_ids)
        n_test = int(total * test_ratio)
        test_ids = all_ids[:n_test]
        rem_ids = all_ids[n_test:]
        return test_ids, {"_ALL": rem_ids}


def _array_split_shuffled(ids: List[str], k: int, seed: int) -> List[List[str]]:
    """Shuffle then split a list into k nearly equal chunks."""
    rng = np.random.RandomState(seed)
    ids = ids.copy()
    rng.shuffle(ids)
    return [lst.tolist() for lst in np.array_split(np.array(ids, dtype=object), k)]


def _stratified_k_chunks(class_to_ids: Dict[str, List[str]], k: int, seed: int) -> List[List[str]]:
    """Return k chunks; each chunk concatenates class-wise chunks for stratified K-fold."""
    per_class_chunks: Dict[str, List[List[str]]] = {}
    for cls, ids in class_to_ids.items():
        per_class_chunks[cls] = _array_split_shuffled(ids, k, seed + hash(cls) % 100000)  # per-class shuffle
    # stitch chunks across classes
    chunks: List[List[str]] = []
    for i in range(k):
        chunk_i: List[str] = []
        for cls in per_class_chunks:
            chunk_i.extend(per_class_chunks[cls][i])
        # Shuffle inside chunk to avoid class clustering
        rng = np.random.RandomState(seed + i)
        rng.shuffle(chunk_i)
        chunks.append(chunk_i)
    return chunks


def create_kfold_splits(
    case_ids: List[str],
    class_names: Tuple[str, ...],
    labels_by_id: Dict[str, Dict[str, int]],
    k: int,
    seed: int,
    test_ratio: float,
    stratify: bool,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns: { f"fold{i}": {"train": [...], "val": [...], "test": [...]} }
    - test is a fixed holdout (may be empty if test_ratio==0)
    - val is the i-th chunk; train is the rest, all excluding test.
    """
    # 1) Build class mapping (if needed)
    if stratify:
        class_ids = _class_to_ids(case_ids, labels_by_id, class_names)
        # remove any duplicates across classes (shouldn't happen if single label), but guard anyway
        # (no-op if clean)
    else:
        class_ids = {"_ALL": list(dict.fromkeys(case_ids))}  # unique preserve order

    # 2) Carve out test holdout
    test_ids, remaining_by_class = _pick_test_holdout(
        class_ids if stratify else class_ids["_ALL"],
        test_ratio=test_ratio,
        seed=seed,
        stratify=stratify,
    )

    # 3) Build K chunks on remaining
    if stratify:
        chunks = _stratified_k_chunks(remaining_by_class, k, seed + 123)
    else:
        chunks = _array_split_shuffled(remaining_by_class["_ALL"], k, seed + 123)

    # 4) Assemble per-fold splits
    folds: Dict[str, Dict[str, List[str]]] = {}
    test_ids = list(dict.fromkeys(test_ids))  # dedupe, preserve order
    for i in range(k):
        fold_name = f"fold{i+1}"
        val_ids = chunks[i]
        train_ids = []
        for j in range(k):
            if j == i:
                continue
            train_ids.extend(chunks[j])

        # Safety: ensure disjointness and no leakage with test
        test_set = set(test_ids)
        train_ids = [x for x in train_ids if x not in test_set]
        val_ids = [x for x in val_ids if x not in test_set]

        folds[fold_name] = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,  # same across folds (can be empty)
        }
    return folds


# ---------------------- Writing & Reporting ----------------------

def write_split_files_per_fold(folds: Dict[str, Dict[str, List[str]]], output_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Write files under <output_dir>/foldN/{train_ids.txt,val_ids.txt,test_ids.txt}."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Dict[str, Path]] = {}
    for fold_name, split in folds.items():
        fold_dir = output_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        written[fold_name] = {}
        for split_name in ["train", "val", "test"]:
            p = fold_dir / f"{split_name}_ids.txt"
            payload = "\n".join(split[split_name])
            if payload:
                payload += "\n"
            p.write_text(payload, encoding="utf-8")
            written[fold_name][split_name] = p
    return written


def _count_classes_for_ids(ids: List[str], labels_by_id: Dict[str, Dict[str, int]], class_names: Tuple[str, ...]) -> Dict[str, int]:
    counts = {name: 0 for name in class_names}
    for dicom_id in ids:
        entry = labels_by_id.get(dicom_id, {})
        for name in class_names:
            counts[name] += int(entry.get(name, 0))
    return counts


def print_and_build_summary(class_names: Tuple[str, ...], labels_by_id: Dict[str, Dict[str, int]], folds: Dict[str, Dict[str, List[str]]],
                            written_paths: Dict[str, Dict[str, Path]]) -> Dict[str, dict]:
    """Print stats and return a JSON-ready summary across folds."""
    print("\n" + "=" * 70)
    print("K-FOLD SPLIT STATISTICS")
    print("=" * 70)

    overall = {name: 0 for name in class_names}
    grand_total = 0

    all_ids = sorted({cid for f in folds.values() for lst in f.values() for cid in lst})
    dataset_counts = _count_classes_for_ids(all_ids, labels_by_id, class_names)
    summary: Dict[str, dict] = {"folds": {}, "dataset_class_distribution": dataset_counts}

    for fold_name, split in folds.items():
        print(f"\n{fold_name.upper()}:")
        fold_summary = {"splits": {}}
        for split_name in ["train", "val", "test"]:
            ids = split[split_name]
            counts = _count_classes_for_ids(ids, labels_by_id, class_names)
            total = len(ids)
            # Compact per-class summary in dataset order
            per_class_str = " | ".join([f"{name} {counts.get(name, 0):4d}" for name in class_names])
            print(f"  {split_name:5s}: {total:4d} cases | {per_class_str}")
            fold_summary["splits"][split_name] = {
                "count": total,
                "class_counts": counts,
                "files": str(written_paths[fold_name][split_name]),
            }
            for name in class_names:
                overall[name] += counts.get(name, 0)
            grand_total += total
        summary["folds"][fold_name] = fold_summary

    print("\n" + "-" * 70)
    print(f"OVERALL (sum across all folds/splits): total entries written = {grand_total}")
    print("  " + " | ".join([f"{name}={overall.get(name, 0)}" for name in class_names]))
    print("-" * 70)

    summary["overall_written"] = {
        "total": grand_total,
        "class_counts": overall,
    }
    return summary


# ---------------------------- Main ----------------------------

def main() -> None:
    args = parse_args()

    if args.folds < 1:
        raise ValueError("--folds must be >= 1")
    if args.test < 0 or args.test >= 1:
        raise ValueError("--test must be in [0, 1). Use 0 for no external test.")

    print("Loading dataset and extracting valid case IDs...")
    labels_by_id, case_ids, class_names = load_dataset_with_classification(args.config_path)
    print(f"Found {len(case_ids)} cases in master_sheet.csv")

    # Resolve default output from config if not overridden
    output_dir = args.output_dir
    if args.output_dir == DEFAULT_OUTPUT:
        configured_dir = ConfigLoader(args.config_path).get("split_files", "dir", default=None)
        if configured_dir is not None:
            output_dir = Path(configured_dir)
            if not output_dir.is_absolute():
                output_dir = ROOT / output_dir
    output_dir = Path(output_dir)

    # Build K-fold splits (with optional test holdout)
    folds = create_kfold_splits(
        case_ids=case_ids,
        class_names=class_names,
        labels_by_id=labels_by_id,
        k=args.folds,
        seed=args.seed,
        test_ratio=args.test,
        stratify=bool(args.stratify),
    )

    # Write per-fold files
    written_paths = write_split_files_per_fold(folds, output_dir)

    # Print detailed stats and build summary
    summary = print_and_build_summary(class_names, labels_by_id, folds, written_paths)

    # Persist compact JSON summary
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to: {summary_path}")

    if args.print_summary:
        print("\nJSON Summary:")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
