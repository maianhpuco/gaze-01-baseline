"""
Main training script for GazeMTL (Multi-Task Learning with Gaze data).
This script provides a unified entry point for training GazeMTL models.
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, roc_curve, auc

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


def compute_per_class_metrics(model, test_loader, device):
    """Compute per-class metrics (AUC, Precision, Recall, IoU) for 3 classes: CHF, pneumonia, Normal."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(images)
            target_logits = outputs['target']
            
            # Get probabilities
            probs = F.softmax(target_logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
    
    # Concatenate all predictions and labels
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Detect actual number of classes from model output and labels
    num_model_classes = all_probs.shape[1]
    unique_labels = np.unique(all_labels)
    num_label_classes = len(unique_labels)
    
    # Debug: Check shapes and unique labels
    print(f"Debug: all_probs shape: {all_probs.shape}, all_labels shape: {all_labels.shape}")
    print(f"Debug: Model outputs {num_model_classes} classes, labels have {num_label_classes} unique values: {unique_labels}")
    print(f"Debug: Label range: [{all_labels.min()}, {all_labels.max()}]")
    
    # Always remap labels to contiguous range [0, 1, 2, ...] if needed
    # This ensures compatibility with sklearn metrics
    if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
        print(f"Warning: Labels not in contiguous range. Remapping {unique_labels} to [0, {len(unique_labels)-1}]")
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
        all_labels = np.array([label_mapping[label] for label in all_labels])
        print(f"Debug: After remapping, unique labels: {np.unique(all_labels)}")
    
    class_names = ['CHF', 'pneumonia', 'Normal']
    num_classes = max(num_model_classes, num_label_classes)
    
    # Get predictions
    preds = all_probs.argmax(axis=1)
    
    # Overall metrics - handle case where model outputs don't match labels
    # sklearn's roc_auc_score requires labels to match the number of probability columns
    if num_model_classes != num_label_classes:
        print(f"Warning: Model outputs {num_model_classes} classes but labels have {num_label_classes} classes")
        # Compute AUC for each class separately and average (one-vs-rest)
        class_aucs = []
        for class_idx in range(min(num_model_classes, num_label_classes)):
            y_true_binary = (all_labels == class_idx).astype(int)
            if len(np.unique(y_true_binary)) > 1 and class_idx < all_probs.shape[1]:
                try:
                    fpr, tpr, _ = roc_curve(y_true_binary, all_probs[:, class_idx])
                    class_aucs.append(auc(fpr, tpr))
                except Exception as e:
                    print(f"Warning: Could not compute AUC for class {class_idx}: {e}")
        overall_auc = np.mean(class_aucs) if class_aucs else 0.0
    else:
        # Standard multiclass case - but need to ensure labels are in [0, num_classes-1]
        # Check if all labels are in valid range
        if all_labels.max() < num_model_classes and all_labels.min() >= 0:
            try:
                overall_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"Warning: roc_auc_score failed with multi_class='ovr': {e}")
                # Fallback to per-class AUC computation
                class_aucs = []
                for class_idx in range(num_model_classes):
                    y_true_binary = (all_labels == class_idx).astype(int)
                    if len(np.unique(y_true_binary)) > 1:
                        try:
                            fpr, tpr, _ = roc_curve(y_true_binary, all_probs[:, class_idx])
                            class_aucs.append(auc(fpr, tpr))
                        except:
                            pass
                overall_auc = np.mean(class_aucs) if class_aucs else 0.0
        else:
            # Labels out of range, use per-class computation
            print(f"Warning: Labels out of range for multiclass AUC, using per-class computation")
            class_aucs = []
            for class_idx in range(min(num_model_classes, int(all_labels.max()) + 1)):
                y_true_binary = (all_labels == class_idx).astype(int)
                if len(np.unique(y_true_binary)) > 1 and class_idx < all_probs.shape[1]:
                    try:
                        fpr, tpr, _ = roc_curve(y_true_binary, all_probs[:, class_idx])
                        class_aucs.append(auc(fpr, tpr))
                    except:
                        pass
            overall_auc = np.mean(class_aucs) if class_aucs else 0.0
    overall_precision = precision_score(all_labels, preds, average='macro', zero_division=0)
    overall_recall = recall_score(all_labels, preds, average='macro', zero_division=0)
    overall_acc = accuracy_score(all_labels, preds)
    
    per_class_metrics = {
        'overall': {
            'AUC': overall_auc,
            'Precision': overall_precision,
            'Recall': overall_recall,
            'Acc': overall_acc
        }
    }
    
    # Per-class metrics
    for class_idx, class_name in enumerate(class_names):
        # Skip if class_idx is beyond model output or labels
        if class_idx >= num_model_classes:
            print(f"Warning: Skipping {class_name} (class_idx={class_idx}) - model only outputs {num_model_classes} classes")
            per_class_metrics[class_name] = {
                'AUC': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'IoU': 0.0
            }
            continue
        
        # Binarize labels for this class
        y_true_binary = (all_labels == class_idx).astype(int)
        y_pred_binary = (preds == class_idx).astype(int)
        
        # AUC (one-vs-rest)
        try:
            if len(np.unique(y_true_binary)) > 1 and class_idx < all_probs.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_binary, all_probs[:, class_idx])
                class_auc = auc(fpr, tpr)
            else:
                class_auc = float('nan')
        except Exception as e:
            print(f"Warning: Could not compute AUC for {class_name}: {e}")
            class_auc = float('nan')
        
        # Precision, Recall
        tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # IoU (Intersection over Union)
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0.0
        
        per_class_metrics[class_name] = {
            'AUC': class_auc if not np.isnan(class_auc) else 0.0,
            'Precision': precision,
            'Recall': recall,
            'IoU': iou
        }
    
    return per_class_metrics


def print_per_class_metrics(per_class_metrics):
    """Print per-class metrics in the desired format."""
    # Print overall metrics
    if 'overall' in per_class_metrics:
        overall = per_class_metrics['overall']
        print(f"[FINAL TEST] AUC={overall['AUC']}, Precision={overall['Precision']}, "
              f"Recall={overall['Recall']}, Acc={overall['Acc']}")
    
    # Print per-class metrics
    print("\n[FINAL TEST] Per-class metrics:")
    class_names = ['CHF', 'pneumonia', 'Normal']
    for class_name in class_names:
        if class_name in per_class_metrics:
            metrics = per_class_metrics[class_name]
            print(f"  {class_name}: AUC={metrics['AUC']:.4f}, "
                  f"Precision={metrics['Precision']:.4f}, "
                  f"Recall={metrics['Recall']:.4f}, "
                  f"IoU={metrics['IoU']:.4f}")


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
        
        # Compute and print per-class metrics on test set
        print("\nComputing per-class metrics on test set...")
        per_class_metrics = compute_per_class_metrics(model, test_loader, device)
        print_per_class_metrics(per_class_metrics)
            
    finally:
        os.chdir(original_cwd)


if __name__ == '__main__':
    main()

