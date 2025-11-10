#!/usr/bin/env python3
"""
Create filtered CSV files (train/val/test) from master_sheet.csv based on split IDs.
These CSV files match the format expected by the original EyegazeDataset.
"""
from pathlib import Path
import pandas as pd

def create_filtered_csvs(
    master_csv: Path,
    split_dir: Path,
    output_dir: Path,
    classes: list = ["CHF", "pneumonia", "Normal"]
):
    """
    Filter master_sheet.csv by split IDs and create train/val/test CSV files.
    
    Expected columns in master_sheet.csv:
    - dicom_id: DICOM file identifier
    - patient_id: Patient ID
    - path: Relative path to DICOM file (e.g., 'p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.dcm')
    - CHF, pneumonia, Normal: Binary labels (0/1)
    """
    # Load master sheet
    master = pd.read_csv(master_csv)
    print(f"Loaded master_sheet.csv with {len(master)} entries")
    print(f"Columns: {master.columns.tolist()}")
    
    # Verify required columns exist
    required_cols = ['dicom_id', 'patient_id', 'path'] + classes
    missing = [col for col in required_cols if col not in master.columns]
    if missing:
        raise ValueError(f"Missing columns in master_sheet.csv: {missing}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        split_file = split_dir / f"{split_name}_ids.txt"
        if not split_file.exists():
            print(f"Warning: {split_file} not found, skipping {split_name}")
            continue
        
        # Load split IDs
        with open(split_file, 'r') as f:
            split_ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Filter master sheet
        filtered = master[master['dicom_id'].isin(split_ids)].copy()
        
        # Ensure all class labels are present
        for cls in classes:
            if cls not in filtered.columns:
                print(f"Warning: Class {cls} not in master_sheet.csv")
                filtered[cls] = 0
        
        # Save filtered CSV
        output_csv = output_dir / f"{split_name}.csv"
        filtered.to_csv(output_csv, index=False)
        print(f"✓ Created {split_name}.csv with {len(filtered)} entries")
        
        # Print label distribution
        label_counts = filtered[classes].sum()
        print(f"  Label distribution: {dict(label_counts)}")
    
    print(f"\n✓ All CSV files created in: {output_dir}")

if __name__ == '__main__':
    # Paths
    gaze_root = Path("/home/qtnguy50/gaze-01-baseline/datasets/gaze_data/egd_cxr/transcripts")
    master_csv = gaze_root / "master_sheet.csv"
    
    split_dir = Path("/home/qtnguy50/gaze-01-baseline/configs/splits/fold1")
    output_dir = Path("/home/qtnguy50/gaze-01-baseline/datasets/filtered_csvs/fold1")
    
    classes = ["CHF", "pneumonia", "Normal"]
    
    print(f"Master CSV: {master_csv}")
    print(f"Split dir: {split_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Classes: {classes}\n")
    
    create_filtered_csvs(
        master_csv=master_csv,
        split_dir=split_dir,
        output_dir=output_dir,
        classes=classes
    )

