"""
Train GazeMTL model from YAML configuration file.
"""
import argparse
import yaml
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))
from train import main as train_main

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def config_to_args(config):
    """Convert config dictionary to argparse.Namespace."""
    args = argparse.Namespace()
    
    # Data configuration
    data = config.get('data', {})
    args.source = data.get('source', 'cxr2')
    args.train_scale = data.get('train_scale', 1.0)
    args.val_scale = data.get('val_scale', 0.2)
    args.batch_size = data.get('batch_size', 32)
    args.num_workers = data.get('num_workers', 8)
    
    # Model configuration
    model = config.get('model', {})
    args.pretrained = model.get('pretrained', True)
    args.gaze_mtl_task = model.get('gaze_mtl_task', 'diffusivity')
    args.task_weights = model.get('task_weights', '1.0')
    
    # Training configuration
    train = config.get('train', {})
    args.n_epochs = train.get('n_epochs', 15)
    args.lr = train.get('lr', 0.0001)
    args.l2 = train.get('l2', 0.01)
    args.optimizer = train.get('optimizer', 'adam')
    args.lr_scheduler = train.get('lr_scheduler', 'cosine_annealing')
    args.seed = train.get('seed', 0)
    
    # Output configuration
    output = config.get('output_path', {})
    base_log_dir = output.get('log_dir', 'logs/gaze_mtl')
    args.log_path = os.path.join(
        base_log_dir,
        args.source,
        args.gaze_mtl_task,
        f'seed_{args.seed}'
    )
    
    # Options
    options = config.get('options', {})
    args.device = options.get('device', 'cuda')
    args.resume = options.get('resume', None)
    
    # Input paths
    input_path = config.get('input_path', {})
    args.data_dir = input_path.get('dicom_raw', '/home/ubuntu/hung.nh2/gaze/dicom/dicom_raw')
    args.gaze_raw_dir = input_path.get('gaze_raw', None)  # Path to 1.0.0 folder
    
    return args

def main():
    parser = argparse.ArgumentParser(description='Train GazeMTL from YAML config')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--override',
        nargs='*',
        help='Override config values: key=value (e.g., train.n_epochs=20)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.override:
        for override in args.override:
            key, value = override.split('=', 1)
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string
            d[keys[-1]] = value
    
    # Convert to args
    train_args = config_to_args(config)
    
    # Run training with the parsed args
    try:
        # Import training components directly
        from train import Trainer
        from model import MultiTaskModel
        from dataset import GazeMTLDataset, HELPER_OUTPUT_DIM_DICT, NUM_CLASSES_DICT
        import torch
        from torch.utils.data import DataLoader
        from torch.optim import Adam, AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
        import numpy as np
        import os
        
        # Add parent directory to path
        gazemtl_dir = os.path.join(os.path.dirname(__file__), '..', 'GazeMTL')
        sys.path.append(gazemtl_dir)
        from transforms import get_data_transforms
        
        # Set random seeds
        torch.manual_seed(train_args.seed)
        np.random.seed(train_args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_args.seed)
        
        # Setup device
        device = torch.device(train_args.device)
        print(f"Using device: {device}")
        
        # Parse task weights
        task_weights_list = [float(w) for w in train_args.task_weights.split(',')]
        task_weights = {'target': task_weights_list[0]}
        helper_tasks = train_args.gaze_mtl_task.split('_') if train_args.gaze_mtl_task else []
        for i, weight in enumerate(task_weights_list[1:]):
            if i < len(helper_tasks):
                task_weights[f'helper_task_{i}'] = weight
        
        # Get helper output dimensions
        helper_output_dims = [HELPER_OUTPUT_DIM_DICT[task] for task in helper_tasks]
        num_classes = NUM_CLASSES_DICT[train_args.source]
        
        # Create datasets
        transforms = get_data_transforms(train_args.source, normalization_type="train_images")
        train_dataset = GazeMTLDataset(
            source=train_args.source,
            task='gaze_mtl',
            gaze_mtl_task=train_args.gaze_mtl_task,
            data_dir=train_args.data_dir,
            split_type='train',
            transform=transforms['train'],
            train_scale=train_args.train_scale,
            val_scale=train_args.val_scale,
            seed=train_args.seed,
            gaze_raw_dir=getattr(train_args, 'gaze_raw_dir', None),
        )
        val_dataset = GazeMTLDataset(
            source=train_args.source,
            task='gaze_mtl',
            gaze_mtl_task=train_args.gaze_mtl_task,
            data_dir=train_args.data_dir,
            split_type='val',
            transform=transforms['val'],
            train_scale=train_args.train_scale,
            val_scale=train_args.val_scale,
            seed=train_args.seed,
            gaze_raw_dir=getattr(train_args, 'gaze_raw_dir', None),
        )
        test_dataset = GazeMTLDataset(
            source=train_args.source,
            task='gaze_mtl',
            gaze_mtl_task=train_args.gaze_mtl_task,
            data_dir=train_args.data_dir,
            split_type='test',
            transform=transforms['test'],
            train_scale=train_args.train_scale,
            val_scale=train_args.val_scale,
            seed=train_args.seed,
            gaze_raw_dir=getattr(train_args, 'gaze_raw_dir', None),
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=True,
            num_workers=train_args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_args.batch_size,
            shuffle=False,
            num_workers=train_args.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=train_args.batch_size,
            shuffle=False,
            num_workers=train_args.num_workers,
            pin_memory=True,
        )
        
        # Create model
        model = MultiTaskModel(
            num_classes=num_classes,
            helper_output_dims=helper_output_dims,
            pretrained=train_args.pretrained,
            freeze_backbone=False,
        )
        
        # Create optimizer
        if train_args.optimizer == 'adam':
            optimizer = Adam(model.parameters(), lr=train_args.lr, weight_decay=train_args.l2)
        else:
            optimizer = AdamW(model.parameters(), lr=train_args.lr, weight_decay=train_args.l2)
        
        # Create scheduler
        scheduler = None
        if train_args.lr_scheduler == 'cosine_annealing':
            scheduler = CosineAnnealingLR(optimizer, T_max=train_args.n_epochs)
        elif train_args.lr_scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=train_args.n_epochs // 3, gamma=0.1)
        elif train_args.lr_scheduler == 'plateau':
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
            log_dir=train_args.log_path,
            task_weights=task_weights,
            checkpoint_metric='target/val/accuracy',
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if train_args.resume:
            start_epoch = trainer.load_checkpoint(train_args.resume)
            print(f"Resumed from epoch {start_epoch}")
        
        # Train
        trainer.train(train_args.n_epochs)
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == '__main__':
    main()

