#!/usr/bin/env python3
"""
Wrapper to call original eye-gaze-dataset preprocessing script.
Generates temporal and static heatmap PNG files from fixations.csv
"""
import sys
import os
from pathlib import Path
import pandas as pd
import multiprocessing

# Add original preprocessing script to path
ROOT = Path(__file__).resolve().parent
PREPROCESS_DIR = ROOT / "externals/eye-gaze-dataset/DataProcessing/DataPostProcessing"
sys.path.insert(0, str(PREPROCESS_DIR))

# Import original preprocessing functions (but we'll call with modified paths)
from create_heatmap_images_and_or_videos import (
    process_eye_gaze_table, 
    concatenate_session_tables
)

def process_fixations_custom(master_csv, fixations_csv, dicom_root, experiment_name, video=False):
    """Modified version of process_fixations that accepts custom paths."""
    import create_heatmap_images_and_or_videos as heatmap_module
    
    # Override the global path variable in the original module
    heatmap_module.original_folder_images = str(dicom_root)
    
    print('--------> FIXATIONS <--------')
    
    cases = pd.read_csv(master_csv)
    table = pd.read_csv(fixations_csv)
    
    sessions = table.groupby(['SESSION_ID'])
    
    try:
        os.makedirs(experiment_name, exist_ok=True)
    except OSError as exc:
        print(exc, ' Proceeding...')
        pass
    
    print(f"Processing {len(sessions)} sessions with {len(table)} fixations total...")
    
    p = multiprocessing.Pool(processes=min(len(sessions), multiprocessing.cpu_count()))
    objects = []
    for session in sessions:
        df = session[1].copy().reset_index(drop=True)
        objects.append((df, experiment_name, cases))
    eye_gaze_session_tables = p.starmap(process_eye_gaze_table, [i for i in objects])
    p.close()
    p.join()
    
    final_table = concatenate_session_tables(eye_gaze_session_tables)
    
    # Save experiment consolidated table
    final_table.to_csv(experiment_name + '.csv', index=False)
    print(f"✓ Saved consolidated fixations to: {experiment_name}.csv")
    
    # We don't need videos for training
    return final_table

if __name__ == '__main__':
    # Our dataset paths
    gaze_root = Path("/home/qtnguy50/gaze-01-baseline/datasets/gaze_data/egd_cxr/transcripts")
    dicom_root = Path("/home/qtnguy50/gaze-01-baseline/datasets/dicom_raw")
    
    master_sheet = gaze_root / "master_sheet.csv"
    fixations_csv = gaze_root / "fixations.csv"
    
    print(f"Master sheet: {master_sheet}")
    print(f"Fixations CSV: {fixations_csv}")
    print(f"DICOM root: {dicom_root}")
    
    # Verify files exist
    if not master_sheet.exists():
        raise FileNotFoundError(f"master_sheet.csv not found: {master_sheet}")
    if not fixations_csv.exists():
        raise FileNotFoundError(f"fixations.csv not found: {fixations_csv}")
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root not found: {dicom_root}")
    
    # Output directory for heatmaps
    output_dir = ROOT / "datasets/heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating heatmaps using original eye-gaze-dataset script")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Call modified preprocessing function with our paths
    experiment_name = str(output_dir / "fixation_heatmaps")
    
    final_table = process_fixations_custom(
        master_csv=master_sheet,
        fixations_csv=fixations_csv,
        dicom_root=dicom_root,
        experiment_name=experiment_name,
        video=False
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Heatmap generation complete!")
    print(f"Output: {output_dir}/fixation_heatmaps/")
    print(f"Total images processed: {final_table['DICOM_ID'].nunique()}")
    print(f"{'='*60}\n")
    
    print("\nNext steps:")
    print("1. Update configs to include heatmaps_path")
    print("2. Modify training scripts to load pre-generated heatmaps")
    print("3. Run training with new pipeline")

