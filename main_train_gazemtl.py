"""
Main training script for GazeMTL (Multi-Task Learning with Gaze data).
This script provides a unified entry point for training GazeMTL models.
"""
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GazeMTL model for medical image classification with gaze data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file option
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file (overrides other arguments)'
    )
    
    # Data arguments
    parser.add_argument(
        '--source',
        type=str,
        default='cxr2',
        choices=['cxr', 'cxr2', 'mets'],
        help='Dataset source'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/ubuntu/hung.nh2/gaze/dicom/dicom_raw',
        help='Directory containing DICOM image data'
    )
    parser.add_argument(
        '--gaze_raw_dir',
        type=str,
        default=None,
        help='Path to PhysioNet 1.0.0 folder (gaze_raw) for proper data structure'
    )
    parser.add_argument(
        '--train_scale',
        type=float,
        default=1.0,
        help='Fraction of training data to use'
    )
    parser.add_argument(
        '--val_scale',
        type=float,
        default=0.2,
        help='Fraction of data for validation'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of data loading workers'
    )
    
    # Model arguments
    parser.add_argument(
        '--gaze_mtl_task',
        type=str,
        default='diffusivity',
        help='Helper task name(s), e.g., "diffusivity", "loc", "time", or "loc_time"'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use pretrained ResNet50 weights'
    )
    parser.add_argument(
        '--no_pretrained',
        dest='pretrained',
        action='store_false',
        help='Do not use pretrained weights'
    )
    parser.add_argument(
        '--task_weights',
        type=str,
        default='1.0',
        help='Task weights as comma-separated values (target,helper1,helper2,...)'
    )
    
    # Training arguments
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=15,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate'
    )
    parser.add_argument(
        '--l2',
        type=float,
        default=0.01,
        help='L2 regularization (weight decay)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'adamw'],
        help='Optimizer type'
    )
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default='cosine_annealing',
        choices=['cosine_annealing', 'step', 'plateau', 'none'],
        help='Learning rate scheduler'
    )
    
    # Other arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed'
    )
    parser.add_argument(
        '--log_path',
        type=str,
        default=None,
        help='Directory to save logs and checkpoints (auto-generated if not specified)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='src/externals/GazeMTL_pytorch/logs',
        help='Base directory for output logs'
    )
    
    return parser.parse_args()


def get_default_log_path(args):
    """Generate default log path if not specified."""
    if args.log_path:
        return args.log_path
    
    # Auto-generate log path
    log_path = os.path.join(
        args.output_dir,
        'gaze_mtl',
        args.source,
        args.gaze_mtl_task,
        f'seed_{args.seed}'
    )
    return log_path


def main():
    """Main training function."""
    args = parse_args()
    
    # If config file is provided, use it
    if args.config:
        # Resolve config path before changing directories
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config_path))
        
        # Use train_from_config.py
        # Change to GazeMTL_pytorch directory for proper imports
        gazemtl_dir = Path(__file__).parent / 'src' / 'externals' / 'GazeMTL_pytorch'
        original_cwd = os.getcwd()
        try:
            os.chdir(gazemtl_dir)
            sys.path.insert(0, str(gazemtl_dir))
            
            from train_from_config import main as train_from_config
            # Temporarily replace sys.argv with absolute config path
            original_argv = sys.argv
            sys.argv = ['train_from_config.py', '--config', config_path]
            try:
                train_from_config()
            finally:
                sys.argv = original_argv
        finally:
            os.chdir(original_cwd)
        return
    
    # Otherwise, use direct training
    log_path = get_default_log_path(args)
    
    # Change to GazeMTL_pytorch directory
    gazemtl_dir = Path(__file__).parent / 'src' / 'externals' / 'GazeMTL_pytorch'
    original_cwd = os.getcwd()
    
    try:
        os.chdir(gazemtl_dir)
        sys.path.insert(0, str(gazemtl_dir))
        
        # Import training components directly
        from train import Trainer
        from model import MultiTaskModel
        from dataset import GazeMTLDataset, HELPER_OUTPUT_DIM_DICT, NUM_CLASSES_DICT
        import torch
        from torch.utils.data import DataLoader
        from torch.optim import Adam, AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
        import numpy as np
        
        # Add parent directory to path for transforms
        gazemtl_parent_dir = str(gazemtl_dir.parent / 'GazeMTL')
        sys.path.append(gazemtl_parent_dir)
        from transforms import get_data_transforms
        
        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # Setup device
        device = torch.device(args.device)
        print(f"Using device: {device}")
        
        # Parse task weights
        task_weights_list = [float(w) for w in args.task_weights.split(',')]
        task_weights = {'target': task_weights_list[0]}
        helper_tasks = args.gaze_mtl_task.split('_') if args.gaze_mtl_task else []
        for i, weight in enumerate(task_weights_list[1:]):
            if i < len(helper_tasks):
                task_weights[f'helper_task_{i}'] = weight
        
        # Get helper output dimensions
        helper_output_dims = [HELPER_OUTPUT_DIM_DICT[task] for task in helper_tasks]
        num_classes = NUM_CLASSES_DICT[args.source]
        
        # Create datasets
        transforms = get_data_transforms(args.source, normalization_type="train_images")
        train_dataset = GazeMTLDataset(
            source=args.source,
            task='gaze_mtl',
            gaze_mtl_task=args.gaze_mtl_task,
            data_dir=args.data_dir,
            split_type='train',
            transform=transforms['train'],
            train_scale=args.train_scale,
            val_scale=args.val_scale,
            seed=args.seed,
            gaze_raw_dir=args.gaze_raw_dir,
        )
        val_dataset = GazeMTLDataset(
            source=args.source,
            task='gaze_mtl',
            gaze_mtl_task=args.gaze_mtl_task,
            data_dir=args.data_dir,
            split_type='val',
            transform=transforms['val'],
            train_scale=args.train_scale,
            val_scale=args.val_scale,
            seed=args.seed,
            gaze_raw_dir=args.gaze_raw_dir,
        )
        test_dataset = GazeMTLDataset(
            source=args.source,
            task='gaze_mtl',
            gaze_mtl_task=args.gaze_mtl_task,
            data_dir=args.data_dir,
            split_type='test',
            transform=transforms['test'],
            train_scale=args.train_scale,
            val_scale=args.val_scale,
            seed=args.seed,
            gaze_raw_dir=args.gaze_raw_dir,
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        # Create model
        model = MultiTaskModel(
            num_classes=num_classes,
            helper_output_dims=helper_output_dims,
            pretrained=args.pretrained,
            freeze_backbone=False,
        )
        
        # Create optimizer
        if args.optimizer == 'adam':
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
        
        # Create scheduler
        scheduler = None
        if args.lr_scheduler == 'cosine_annealing':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        elif args.lr_scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=args.n_epochs // 3, gamma=0.1)
        elif args.lr_scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            log_dir=log_path,
            task_weights=task_weights,
            checkpoint_metric='target/val/accuracy',
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)
            print(f"Resumed from checkpoint: {args.resume}")
        
        # Train
        trainer.train(args.n_epochs)
            
    finally:
        os.chdir(original_cwd)


if __name__ == '__main__':
    main()

