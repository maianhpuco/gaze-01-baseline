#!/usr/bin/env python3
"""
Temporal Model Training - Adapted from eye-gaze-dataset/Experiments/main.py
Reproduces the exact training steps with EGDCXRDataset
"""
from __future__ import annotations

import os
import sys
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torchvision import transforms
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import gaze_utils after sys.path is set
from gaze_utils.heatmap_utils import (
    create_static_heatmap,
    create_temporal_heatmaps,
)

# Import dataset
try:
    from egd_cxr_dataset.datasets.edg_cxr import EGDCXRDataset
except ModuleNotFoundError:
    DATASETS_DIR = SRC / "egd_cxr_dataset" / "datasets"
    if str(DATASETS_DIR) not in sys.path:
        sys.path.insert(0, str(DATASETS_DIR))
    from edg_cxr import EGDCXRDataset

# Import temporal model components from externals
sys.path.insert(0, str(SRC / "externals/eye-gaze-dataset/Experiments"))
from models.eyegaze_model import EyegazeModel
from utils.utils import cyclical_lr, train_teacher_network, test_eyegaze_network, load_model

plt.rcParams['figure.figsize'] = [10, 10]

logging.basicConfig(stream=sys.stdout, format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', 
                    datefmt='%H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger('temporal')
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('PIL').setLevel(logging.INFO)


class TemporalDatasetAdapter(Dataset):
    """Adapt EGDCXRDataset to temporal model format with heatmap sequences"""

    def __init__(
        self,
        base: EGDCXRDataset,
        image_transform,
        heatmap_temporal_transform,
        heatmap_static_transform,
        *,
        cache_dir: Path | None = None,
        split: str = "train",
        max_seq: int = 10,
    ):
        self.base = base
        self.image_transform = image_transform
        self.heatmap_temporal_transform = heatmap_temporal_transform
        self.heatmap_static_transform = heatmap_static_transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split = split
        self.max_seq = max_seq

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]

        img1 = sample["image"]
        img3 = img1.repeat(3, 1, 1)
        pil_img = transforms.ToPILImage()(img3)
        img_tensor = self.image_transform(pil_img)

        fx = sample["fixations"]
        xy = fx["xy"]
        dwell = fx["dwell"]
        dicom_id = sample["dicom_id"]

        temporal_np = None
        static_np = None
        if self.cache_dir:
            temporal_path = self.cache_dir / "temporal" / self.split / f"{dicom_id}.npz"
            static_path = self.cache_dir / "static" / self.split / f"{dicom_id}.npy"
            if temporal_path.exists():
                temporal_np = np.load(temporal_path)["heatmaps"]
            if static_path.exists():
                static_np = np.load(static_path)

        if temporal_np is None:
            temporal_np = create_temporal_heatmaps(xy, dwell, max_seq=self.max_seq)
        if static_np is None:
            static_np = create_static_heatmap(xy, dwell)

        temporal_heatmaps = [
            self.heatmap_temporal_transform(
                Image.fromarray((frame * 255).astype(np.uint8))
            )
            for frame in temporal_np
        ]
        temporal_tensor = torch.stack(temporal_heatmaps)  # [seq_len, 3, 224, 224]

        static_heatmap_pil = Image.fromarray((static_np * 255).astype(np.uint8))
        static_tensor = self.heatmap_static_transform(static_heatmap_pil)  # [1, 224, 224]

        y_idx = int(sample["labels"]["single_index"].item())
        if y_idx < 0:
            y_idx = 0

        # Return class index (for CrossEntropyLoss) instead of one-hot (for BCEWithLogitsLoss)
        label = torch.tensor(y_idx, dtype=torch.long)

        return img_tensor, label, idx, temporal_tensor, static_tensor


def collate_fn_temporal(batch):
    """
    Custom collate function for temporal data with dynamic padding.
    Sorts sequences by length (descending) for efficient RNN processing.
    """
    # Sort by temporal sequence length (descending) for efficient RNN batching
    batch.sort(key=lambda x: x[3].shape[0], reverse=True)
    images, labels, indices, temporal_hms, static_hms = zip(*batch)
    
    images = torch.stack(images)
    labels = torch.stack(labels)
    indices = torch.tensor(indices)
    static_hms = torch.stack(static_hms)
    
    # Use pad_sequence for efficient dynamic padding of temporal sequences
    if isinstance(temporal_hms[0], torch.Tensor):
        temporal_hms = torch.nn.utils.rnn.pad_sequence(temporal_hms, batch_first=True)
    else:
        temporal_hms = torch.stack(temporal_hms)
    
    return images, labels, indices, temporal_hms, static_hms


def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch Temporal Model with Eye Gaze')
    
    # Data
    parser.add_argument('--config', type=str, default='configs/train_gazegnn.yaml', help='Config path')
    parser.add_argument('--output_dir', type=str, default='results/temporal', help='Output directory')
    parser.add_argument('--class_names', type=list, default=['CHF', 'pneumonia', 'Normal'], help='Label names')
    parser.add_argument('--num_workers', type=int, default=None, help='number of workers')
    parser.add_argument('--resize', type=int, default=None, help='Resizing images')
    parser.add_argument('--max_fixations', type=int, default=None, help='Max temporal sequence length')
    parser.add_argument('--cache_dir', type=str, default='cache/heatmaps', help='Directory to read/write cached heatmaps')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate (default: from config or 3e-4)')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay (default: from config or 1e-4)')
    parser.add_argument('--scheduler', action='store_true', help='[USE] scheduler (cyclical LR, default: True)')
    parser.add_argument('--no_scheduler', dest='scheduler', action='store_false', help='Disable scheduler (use ReduceLROnPlateau instead)')
    parser.add_argument('--step_size', type=int, default=5, help='scheduler step size')
    
    # Temporal Model Specific
    parser.add_argument('--model_type', default='temporal', choices=['baseline', 'temporal'], help='model choice')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden size for image model')
    parser.add_argument('--emb_dim', type=int, default=64, help='cnn embedding size for heatmap model')
    parser.add_argument('--hidden_hm', nargs='+', type=int, default=[256, 128], help='hidden size for heatmap model')
    parser.add_argument('--num_layers_hm', type=int, default=1, help='num layers for heatmap model')
    parser.add_argument('--cell', type=str, default='lstm', choices=['lstm', 'gru'], help='LSTM or GRU')
    parser.add_argument('--brnn_hm', default=True, action='store_true', help='bidirectional for heatmap model')
    parser.add_argument('--attention', default=True, action='store_true', help='[USE] attention')
    
    # Misc
    parser.add_argument('--gpus', type=str, default='0,1', help='Which gpus to use')
    parser.add_argument('--test', default=False, action='store_true', help='[USE] flag for testing only')
    parser.add_argument('--testdir', type=str, default=None, help='model to test')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    return parser


def load_data(args):
    """Load data using EGDCXRDataset"""
    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Paths
    root = Path(cfg['input_path']['gaze_raw'])
    seg = Path(cfg['input_path']['segmentation_dir'])
    transcripts = Path(cfg['input_path'].get('transcripts_dir', seg))
    dicom_root = Path(cfg['input_path']['dicom_raw'])
    
    # Training configuration overrides
    train_cfg = cfg.get('train', {}) if isinstance(cfg, dict) else {}
    if args.batch_size is None:
        args.batch_size = int(train_cfg.get('batch_size', 32))
    if args.num_workers is None:
        args.num_workers = int(train_cfg.get('num_workers', 16))
    if args.epochs is None:
        args.epochs = int(train_cfg.get('epochs', 10))
    if args.max_fixations is None:
        args.max_fixations = int(train_cfg.get('max_fixations', 10))
    if args.resize is None:
        args.resize = int(train_cfg.get('resize', 224))
    
    if args.weight_decay is None:
        args.weight_decay = float(train_cfg.get('weight_decay', 1e-4))
    if 'scheduler' in train_cfg:
        args.scheduler = bool(train_cfg.get('scheduler'))
    else:
        args.scheduler = True
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    
    # Splits
    split_cfg = cfg.get('split_files', {}) if cfg else {}
    default_split_dir = ROOT / "configs" / "splits" / "fold1"
    split_dir_path = split_cfg.get('dir', default_split_dir)
    split_dir = Path(split_dir_path)
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    if not split_dir.exists():
        logger.warning(f"Split directory {split_dir} not found. Falling back to {default_split_dir}.")
        split_dir = default_split_dir
    
    def read_ids(name):
        file_path = split_dir / f"{name}_ids.txt"
        if not file_path.exists():
            raise FileNotFoundError(f"Expected split file not found: {file_path}")
        return [
            line.strip()
            for line in file_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    
    train_ids = read_ids("train")
    val_ids = read_ids("val")
    test_ids = read_ids("test")
    
    # Classes
    class_names = train_cfg.get('classes', args.class_names)
    if not class_names:
        class_names = args.class_names
    args.class_names = class_names
    
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    seq = iaa.Sequential([iaa.Resize((args.resize, args.resize))])

    def _apply_seq(img):
        arr = np.array(img)
        arr = seq(image=arr)
        return Image.fromarray(arr)

    image_transform = transforms.Compose([
        transforms.Lambda(_apply_seq),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    heatmap_temporal_transform = transforms.Compose([
        transforms.Resize([args.resize, args.resize]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    heatmap_static_transform = transforms.Compose([
        transforms.Resize([args.resize, args.resize]),
        transforms.ToTensor()
    ])
    
    # Create base datasets
    train_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts, 
                               dicom_root=dicom_root, max_fixations=args.max_fixations,
                               case_ids=train_ids, classes=args.class_names)
    valid_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts,
                               dicom_root=dicom_root, max_fixations=args.max_fixations,
                               case_ids=val_ids, classes=args.class_names)
    test_base = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=transcripts,
                              dicom_root=dicom_root, max_fixations=args.max_fixations,
                              case_ids=test_ids, classes=args.class_names)
    
    # Wrap with adapter
    train_dataset = TemporalDatasetAdapter(
        train_base,
        image_transform,
        heatmap_temporal_transform,
        heatmap_static_transform,
        cache_dir=cache_dir,
        split="train",
        max_seq=args.max_fixations,
    )
    valid_dataset = TemporalDatasetAdapter(
        valid_base,
        image_transform,
        heatmap_temporal_transform,
        heatmap_static_transform,
        cache_dir=cache_dir,
        split="val",
        max_seq=args.max_fixations,
    )
    test_dataset = TemporalDatasetAdapter(
        test_base,
        image_transform,
        heatmap_temporal_transform,
        heatmap_static_transform,
        cache_dir=cache_dir,
        split="test",
        max_seq=args.max_fixations,
    )
    
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                          num_workers=args.num_workers, collate_fn=collate_fn_temporal, 
                          drop_last=True, pin_memory=True)
    valid_dl = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_fn_temporal, 
                          drop_last=True, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate_fn_temporal,
                         pin_memory=True)
    
    return train_dl, valid_dl, test_dl


def run_experiment(args, train_dl, valid_dl, output_model_path):
    """Train the model"""
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    model = EyegazeModel(args.model_type, len(args.class_names), dropout=args.dropout,
                        emb_dim=args.emb_dim, hidden_dim=args.emb_dim, hidden_hm=args.hidden_hm,
                        attention=args.attention, cell=args.cell, brnn_hm=args.brnn_hm,
                        num_layers_hm=args.num_layers_hm).to(args.device)
    
    logger.info(model)
    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    logger.info(f'Number of parameters: {total_params}')
    
    if len(args.gpus.split(',')) > 1:
        logger.info(f"Using {len(args.gpus.split(','))} GPUs!")
        device_ids = [int(i) for i in args.gpus.split(',')]
        model = nn.DataParallel(model, device_ids=device_ids)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler:
        clr = cyclical_lr(step_sz=args.step_size, min_lr=args.lr, max_lr=1, mode='triangular2')
        exp_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, [clr])
    else:
        exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_teacher_network(model, criterion, optimizer, exp_lr_scheduler, train_dl, valid_dl,
                         output_model_path, args.epochs, viz=None, env_name=None, is_schedule=args.scheduler)
    
    logger.info(f'Model saved at {output_model_path}')
    return model


if __name__ == '__main__':
    args = make_parser().parse_args()
    
    # Set seeds
    random.seed(args.rseed)
    np.random.seed(args.rseed)
    torch.manual_seed(args.rseed)
    
    cuda = torch.cuda.is_available() and args.gpus != '-1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if cuda:
        torch.cuda.manual_seed(args.rseed)
        torch.cuda.manual_seed_all(args.rseed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    torch.cuda.set_device("cuda:" + args.gpus.split(',')[0])
    args.device = torch.device("cuda:" + args.gpus.split(',')[0]) if cuda else torch.device('cpu')
    
    logger.info(f"Using device: {args.device}")
    if cuda:
        logger.info(torch.cuda.get_device_name(args.device))
    
    # Create output path
    timestamp = str(datetime.now()).replace(" ", "").split('.')[0]
    comment_variable = f'{args.model_type}_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}_{timestamp}'
    output_model_path = os.path.join(args.output_dir, comment_variable)
    
    # Load data (this sets scheduler default and other config values)
    train_dl, valid_dl, test_dl = load_data(args)
    
    # Log arguments AFTER load_data so scheduler default is applied
    logger.info(f"[Arguments]: {args}")
    
    if not args.test:  # Training
        logger.info('---- TRAINING ----')
        run_experiment(args, train_dl, valid_dl, output_model_path)
    
    # Testing
    logger.info('---- TESTING ----')
    model_dir = args.testdir if args.testdir else output_model_path
    best_metrics = None
    best_model_name = ''

    candidate_models = []
    best_weights_path = os.path.join(model_dir, "best_weights.pth")
    if os.path.exists(best_weights_path):
        candidate_models.append("best_weights.pth")
    else:
        candidate_models.extend([f'Epoch_{i}.pth' for i in range(args.epochs)])

    for model_name in candidate_models:
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            logger.info(f"Skipping missing model: {model_path}")
            continue

        model = EyegazeModel(args.model_type, len(args.class_names), dropout=args.dropout,
                            emb_dim=args.emb_dim, hidden_dim=args.emb_dim, hidden_hm=args.hidden_hm,
                            attention=args.attention, cell=args.cell, brnn_hm=args.brnn_hm,
                            num_layers_hm=args.num_layers_hm).to(args.device)
        
        # Load model first before wrapping in DataParallel
        loaded_model = load_model(model_name, model_dir, model)
        if loaded_model is False:
            continue
        
        # Wrap in DataParallel after loading if using multiple GPUs
        if len(args.gpus.split(',')) > 1:
            device_ids = [int(i) for i in args.gpus.split(',')]
            model = nn.DataParallel(loaded_model, device_ids=device_ids)
        else:
            model = loaded_model
        
        model = model.to(args.device)

        metrics = test_eyegaze_network(model, test_dl, args.class_names, model_dir, model_name)
        if not metrics:
            continue

        logger.info(
            f"{model_name} metrics: AUC={metrics.get('macro_auc')}, "
            f"Precision={metrics.get('macro_precision')}, "
            f"Recall={metrics.get('macro_recall')}, "
            f"SubsetAcc={metrics.get('subset_accuracy')}, "
            f"MulticlassAcc={metrics.get('multiclass_accuracy')}, "
            f"Loss={metrics.get('loss')}, "
            f"IoU={metrics.get('macro_iou')}"
        )

        macro_auc = metrics.get('macro_auc', 0.0) or 0.0
        if best_metrics is None or macro_auc >= best_metrics.get('macro_auc', -np.inf):
            best_model_name = model_name
            best_metrics = metrics
    
    if best_metrics:
        logger.info(
            f"Best model: {best_model_name} | "
            f"AUC={best_metrics.get('macro_auc')}, "
            f"Precision={best_metrics.get('macro_precision')}, "
            f"Recall={best_metrics.get('macro_recall')}, "
            f"SubsetAcc={best_metrics.get('subset_accuracy')}, "
            f"MulticlassAcc={best_metrics.get('multiclass_accuracy')}, "
            f"Loss={best_metrics.get('loss')}, "
            f"IoU={best_metrics.get('macro_iou')}"
        )
        
        logger.info("[FINAL TEST] Per-class metrics:")
        per_class_auc = best_metrics.get("per_class_auc", [])
        per_class_precision = best_metrics.get("per_class_precision", [])
        per_class_recall = best_metrics.get("per_class_recall", [])
        per_class_iou = best_metrics.get("per_class_iou", [])
        
        for i, cls_name in enumerate(args.class_names):
            if i < len(per_class_auc) and i < len(per_class_precision) and i < len(per_class_recall) and i < len(per_class_iou):
                logger.info(f"  {cls_name}: AUC={per_class_auc[i]:.4f}, Precision={per_class_precision[i]:.4f}, "
                           f"Recall={per_class_recall[i]:.4f}, IoU={per_class_iou[i]:.4f}")
    else:
        logger.info("No valid temporal checkpoints evaluated.")

