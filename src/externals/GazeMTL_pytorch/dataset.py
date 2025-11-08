"""
Modern PyTorch dataset implementation for gaze-based medical image classification.
Compatible with existing data loading utilities.
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pydicom

# Add parent directory to path to import utils
gazemtl_dir = os.path.join(os.path.dirname(__file__), '..', 'GazeMTL')
sys.path.append(gazemtl_dir)

# Change to GazeMTL directory temporarily to load file markers (utils.py uses relative paths)
original_cwd = os.getcwd()
try:
    os.chdir(gazemtl_dir)
    from utils import (
        load_file_markers,
        load_helper_task_labels,
        load_weak_labels,
        standardize_label,
    )
finally:
    os.chdir(original_cwd)

# Task configuration dictionaries
NUM_GAZE_DIMS_DICT = {
    "none": 0,
    "loc": 9,
    "time": 1,
    "diffusivity": 1,
}

NUM_CLASSES_DICT = {"cxr": 2, "mets": 2, "cxr2": 2}

HELPER_OUTPUT_DIM_DICT = {"loc": 9, "time": 2, "diffusivity": 2}


class GazeMTLDataset(Dataset):
    """
    Dataset for multi-task learning with gaze data.
    
    Args:
        source: Dataset source name (cxr, cxr2, mets)
        task: Task type (gaze_mtl, weak_gaze, etc.)
        gaze_mtl_task: Helper task name(s), e.g., "diffusivity" or "loc_time"
        data_dir: Directory containing image data
        split_type: Data split (train, val, test)
        transform: Image transforms to apply
        train_scale: Fraction of training data to use
        val_scale: Fraction of data to use for validation
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        source: str,
        task: str,
        gaze_mtl_task: str,
        data_dir: str,
        split_type: str,
        transform=None,
        train_scale: float = 1.0,
        val_scale: float = 0.2,
        seed: int = 0,
        gaze_raw_dir: str = None,
    ):
        self.split_type = split_type
        self.transform = transform
        self.data_dir = data_dir
        self.gaze_raw_dir = gaze_raw_dir  # Path to 1.0.0 folder for PhysioNet structure
        self.source = source
        self.task = task
        
        # Load file markers (image paths and labels)
        # Need to change to GazeMTL directory because utils.py uses relative paths
        original_cwd = os.getcwd()
        try:
            os.chdir(gazemtl_dir)
            self.file_markers = load_file_markers(
                source, split_type, train_scale, val_scale, seed
            )
        finally:
            os.chdir(original_cwd)
        print(f"{len(self.file_markers)} files in {split_type} split...")
        
        # Load helper task labels if using gaze_mtl
        # Need to change to GazeMTL directory because utils.py uses relative paths
        helper_task_labels_dict = {}
        if gaze_mtl_task:
            original_cwd = os.getcwd()
            try:
                os.chdir(gazemtl_dir)
                helper_task_labels_dict = load_helper_task_labels(source, gaze_mtl_task)
            finally:
                os.chdir(original_cwd)
        
        # Parse helper tasks
        helper_tasks = gaze_mtl_task.split("_") if gaze_mtl_task else []
        self.num_helper_tasks = len(helper_tasks)
        self.num_gaze_dims = [NUM_GAZE_DIMS_DICT[task] for task in helper_tasks]
        self.num_classes = NUM_CLASSES_DICT[source]
        
        # Prepare data
        self.image_ids = []
        self.target_labels = []
        self.helper_labels = [[] for _ in range(self.num_helper_tasks)]
        
        for img_path, label in self.file_markers:
            img_id = img_path.split("/")[-1]
            img_name = img_id.split(".")[0]
            
            if source == "cxr":
                img_id = img_path
                img_name = img_path
            if source == "mets":
                img_name = img_id
            
            self.image_ids.append(img_id)
            
            # Standardize label (0 = negative, 1 = positive)
            label = standardize_label(label, source)
            self.target_labels.append(label)
            
            # Get helper task labels
            if self.task == "gaze_mtl" and img_name in helper_task_labels_dict:
                gaze_labels = helper_task_labels_dict[img_name]
            else:
                gaze_labels = []
                for i in range(self.num_helper_tasks):
                    if self.num_gaze_dims[i] == 1:
                        gaze_labels.append(0)
                    else:
                        gaze_labels.append([0] * self.num_gaze_dims[i])
            
            for i in range(self.num_helper_tasks):
                self.helper_labels[i].append(gaze_labels[i])
        
        # Convert to tensors
        self.target_labels = torch.tensor(self.target_labels, dtype=torch.long)
        for i in range(self.num_helper_tasks):
            self.helper_labels[i] = torch.tensor(
                np.array(self.helper_labels[i]), dtype=torch.long
            )
    
    def __len__(self):
        return len(self.file_markers)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns:
            image: Transformed image tensor [3, H, W]
            target: Target label (int)
            helper_labels: List of helper task labels
            image_id: Image identifier
        """
        img_id = self.image_ids[idx]
        target_label = self.target_labels[idx]
        
        # Construct image path
        # Support both old structure and PhysioNet 1.0.0 structure
        if self.source == "mets":
            base_dir = os.path.join(self.data_dir, self.source.upper())
            img_case_id = img_id.split("_")[1]
            img_path = os.path.join(base_dir, "Mets_" + img_case_id)
            img_path = os.path.join(img_path, img_id)
        elif self.source == "cxr":
            base_dir = os.path.join(self.data_dir, self.source.upper())
            img_path = os.path.join(base_dir, "dicom_images", img_id)
        elif self.source == "cxr2":
            # Try PhysioNet 1.0.0 structure first if gaze_raw_dir is provided
            if self.gaze_raw_dir and os.path.exists(self.gaze_raw_dir):
                # In 1.0.0 structure, images might be in subdirectories
                # Try direct path first
                img_path = os.path.join(self.gaze_raw_dir, img_id)
                if not os.path.exists(img_path):
                    # Try in dicom_images subdirectory
                    img_path = os.path.join(self.gaze_raw_dir, "dicom_images", img_id)
                    if not os.path.exists(img_path):
                        # Fall back to data_dir
                        img_path = os.path.join(self.data_dir, img_id)
                else:
                    # Found in gaze_raw_dir, use it
                    pass
            else:
                # Use original structure: directly in data_dir
                img_path = os.path.join(self.data_dir, img_id)
        else:
            base_dir = os.path.join(self.data_dir, self.source.upper())
            img_path = os.path.join(base_dir, img_id)
        
        # Load image
        if "cxr" in self.source:
            try:
                ds = pydicom.dcmread(img_path)
                img = ds.pixel_array
                img = Image.fromarray(np.uint8(img))
            except (ValueError, AttributeError, KeyError) as e:
                import warnings
                try:
                    ds = pydicom.dcmread(img_path, stop_before_pixels=True)
                    rows = getattr(ds, 'Rows', 512)
                    cols = getattr(ds, 'Columns', 512)
                    warnings.warn(
                        f"Error reading pixel data from {img_path}: {str(e)}. "
                        f"Using black image ({rows}x{cols}) as fallback."
                    )
                    img = Image.new('L', (cols, rows), 0)
                except Exception:
                    warnings.warn(
                        f"Error reading DICOM file {img_path}: {str(e)}. "
                        "Using black image (512x512) as fallback."
                    )
                    img = Image.new('L', (512, 512), 0)
        else:
            try:
                img = Image.open(img_path)
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Error reading image file {img_path}: {str(e)}. "
                    "Using black image as fallback."
                )
                img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Ensure image has correct shape: [C, H, W] where C is 3
        if len(img.shape) != 3:
            while len(img.shape) < 3:
                img = img.unsqueeze(0)
        
        if img.shape[0] == 1:
            # Convert grayscale to RGB by replicating the channel
            img = torch.cat([img, img, img], dim=0)
        
        # Ensure final shape is [3, H, W]
        assert len(img.shape) == 3 and img.shape[0] == 3, \
            f"Expected image shape [3, H, W], got {img.shape}"
        
        # Prepare helper labels
        helper_labels = {}
        for i in range(self.num_helper_tasks):
            helper_labels[f'helper_task_{i}'] = self.helper_labels[i][idx]
        
        return {
            'image': img,
            'target': target_label,
            'image_id': img_id,
            **helper_labels
        }

