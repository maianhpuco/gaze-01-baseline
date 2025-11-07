"""
Modern PyTorch training script for gaze-based multi-task learning.
Uses latest PyTorch features and best practices.
"""
import argparse
import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import torchmetrics
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
gazemtl_dir = os.path.join(os.path.dirname(__file__), '..', 'GazeMTL')
sys.path.append(gazemtl_dir)
from transforms import get_data_transforms

from model import MultiTaskModel
from dataset import GazeMTLDataset, HELPER_OUTPUT_DIM_DICT, NUM_CLASSES_DICT


class Trainer:
    """
    Trainer class for multi-task learning with modern PyTorch practices.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        log_dir: str,
        task_weights: Dict[str, float] = None,
        checkpoint_metric: str = 'target/val/accuracy',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.task_weights = task_weights or {'target': 1.0}
        self.checkpoint_metric = checkpoint_metric
        self.best_metric = -float('inf')
        self.best_epoch = 0
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Initialize metrics
        self.metrics = {}
        # Target task metrics (binary classification)
        for split in ['train', 'val', 'test']:
            self.metrics[f'target/{split}'] = {
                'accuracy': torchmetrics.Accuracy(task='binary', num_classes=2).to(device),
                'auc': torchmetrics.AUROC(task='binary', num_classes=2).to(device),
            }
        
        # Helper task metrics (dimension depends on task)
        for i in range(model.num_helper_tasks):
            num_classes = model.helper_output_dims[i] if i < len(model.helper_output_dims) else 2
            task_type = 'binary' if num_classes == 2 else 'multiclass'
            for split in ['train', 'val', 'test']:
                self.metrics[f'helper_task_{i}/{split}'] = {
                    'accuracy': torchmetrics.Accuracy(task=task_type, num_classes=num_classes).to(device),
                }
                if num_classes == 2:
                    self.metrics[f'helper_task_{i}/{split}']['auc'] = torchmetrics.AUROC(task='binary', num_classes=2).to(device)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        # Initialize task_losses for all possible tasks (target + helper tasks)
        task_losses = {'target': 0.0}
        for i in range(self.model.num_helper_tasks):
            task_losses[f'helper_task_{i}'] = 0.0
        
        # Reset metrics
        for metric_dict in self.metrics.values():
            for metric in metric_dict.values():
                metric.reset()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute losses
            losses = {}
            total_batch_loss = 0.0
            
            # Target task loss
            target_logits = outputs['target']
            target_loss = F.cross_entropy(target_logits, targets)
            losses['target'] = target_loss
            total_batch_loss += self.task_weights.get('target', 1.0) * target_loss
            
            # Helper task losses
            for i in range(self.model.num_helper_tasks):
                task_name = f'helper_task_{i}'
                if task_name in batch and task_name in outputs:
                    helper_targets = batch[task_name].to(self.device)
                    helper_logits = outputs[task_name]
                    helper_loss = F.cross_entropy(helper_logits, helper_targets)
                    losses[task_name] = helper_loss
                    total_batch_loss += self.task_weights.get(task_name, 1.0) * helper_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                # Target task metrics
                probs = F.softmax(target_logits, dim=1)
                preds = target_logits.argmax(dim=1)  # Get predicted class indices
                self.metrics['target/train']['accuracy'].update(preds, targets)
                self.metrics['target/train']['auc'].update(probs[:, 1], targets)
                
                # Helper task metrics
                for i in range(self.model.num_helper_tasks):
                    task_name = f'helper_task_{i}'
                    if task_name in batch and task_name in outputs:
                        helper_targets = batch[task_name].to(self.device)
                        helper_logits = outputs[task_name]
                        helper_probs = F.softmax(helper_logits, dim=1)
                        helper_preds = helper_logits.argmax(dim=1)  # Get predicted class indices
                        self.metrics[f'{task_name}/train']['accuracy'].update(helper_preds, helper_targets)
                        if f'{task_name}/train' in self.metrics and 'auc' in self.metrics[f'{task_name}/train']:
                            self.metrics[f'{task_name}/train']['auc'].update(helper_probs[:, 1], helper_targets)
            
            total_loss += total_batch_loss.item()
            for task, loss in losses.items():
                task_losses[task] += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'target_loss': f'{target_loss.item():.4f}'
            })
        
        # Compute epoch metrics
        epoch_metrics = {
            'loss': total_loss / len(self.train_loader),
        }
        for task, loss in task_losses.items():
            epoch_metrics[f'{task}/loss'] = loss / len(self.train_loader)
        
        # Get metric values
        for split in ['train']:
            # Target task metrics
            acc = self.metrics['target/train']['accuracy'].compute().item()
            auc = self.metrics['target/train']['auc'].compute().item()
            epoch_metrics['target/train/accuracy'] = acc
            epoch_metrics['target/train/auc'] = auc
            
            # Helper task metrics
            for i in range(self.model.num_helper_tasks):
                task_name = f'helper_task_{i}'
                if f'{task_name}/train' in self.metrics:
                    acc = self.metrics[f'{task_name}/train']['accuracy'].compute().item()
                    epoch_metrics[f'{task_name}/train/accuracy'] = acc
                    if 'auc' in self.metrics[f'{task_name}/train']:
                        auc = self.metrics[f'{task_name}/train']['auc'].compute().item()
                        epoch_metrics[f'{task_name}/train/auc'] = auc
        
        # Log to TensorBoard
        for key, value in epoch_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, split: str = 'val', epoch: int = 0) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        loader = self.val_loader if split == 'val' else self.test_loader
        
        total_loss = 0.0
        # Initialize task_losses for all possible tasks (target + helper tasks)
        task_losses = {'target': 0.0}
        for i in range(self.model.num_helper_tasks):
            task_losses[f'helper_task_{i}'] = 0.0
        
        # Reset metrics
        for metric_dict in self.metrics.values():
            for metric in metric_dict.values():
                metric.reset()
        
        all_preds = {task: [] for task in ['target'] + [f'helper_task_{i}' for i in range(self.model.num_helper_tasks)]}
        all_targets = {task: [] for task in ['target'] + [f'helper_task_{i}' for i in range(self.model.num_helper_tasks)]}
        
        pbar = tqdm(loader, desc=f'[{split.upper()}]')
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute losses
            losses = {}
            total_batch_loss = 0.0
            
            # Target task
            target_logits = outputs['target']
            target_loss = F.cross_entropy(target_logits, targets)
            losses['target'] = target_loss
            total_batch_loss += self.task_weights.get('target', 1.0) * target_loss
            
            # Helper tasks
            for i in range(self.model.num_helper_tasks):
                task_name = f'helper_task_{i}'
                if task_name in batch and task_name in outputs:
                    helper_targets = batch[task_name].to(self.device)
                    helper_logits = outputs[task_name]
                    helper_loss = F.cross_entropy(helper_logits, helper_targets)
                    losses[task_name] = helper_loss
                    total_batch_loss += self.task_weights.get(task_name, 1.0) * helper_loss
            
            # Update metrics
            # Target task metrics
            probs = F.softmax(target_logits, dim=1)
            preds = target_logits.argmax(dim=1)  # Get predicted class indices
            self.metrics[f'target/{split}']['accuracy'].update(preds, targets)
            self.metrics[f'target/{split}']['auc'].update(probs[:, 1], targets)
            
            # Helper task metrics
            for i in range(self.model.num_helper_tasks):
                task_name = f'helper_task_{i}'
                if task_name in batch and task_name in outputs:
                    helper_targets = batch[task_name].to(self.device)
                    helper_logits = outputs[task_name]
                    helper_probs = F.softmax(helper_logits, dim=1)
                    helper_preds = helper_logits.argmax(dim=1)  # Get predicted class indices
                    if f'{task_name}/{split}' in self.metrics:
                        self.metrics[f'{task_name}/{split}']['accuracy'].update(helper_preds, helper_targets)
                        if 'auc' in self.metrics[f'{task_name}/{split}']:
                            self.metrics[f'{task_name}/{split}']['auc'].update(helper_probs[:, 1], helper_targets)
            
            # Store predictions
            all_preds['target'].append(probs.cpu())
            all_targets['target'].append(targets.cpu())
            
            total_loss += total_batch_loss.item()
            for task, loss in losses.items():
                task_losses[task] += loss.item()
        
        # Compute metrics
        epoch_metrics = {
            'loss': total_loss / len(loader),
        }
        for task, loss in task_losses.items():
            epoch_metrics[f'{task}/loss'] = loss / len(loader)
        
        # Get metric values
        # Target task metrics
        acc = self.metrics[f'target/{split}']['accuracy'].compute().item()
        auc = self.metrics[f'target/{split}']['auc'].compute().item()
        epoch_metrics[f'target/{split}/accuracy'] = acc
        epoch_metrics[f'target/{split}/auc'] = auc
        
        # Helper task metrics
        for i in range(self.model.num_helper_tasks):
            task_name = f'helper_task_{i}'
            if f'{task_name}/{split}' in self.metrics:
                acc = self.metrics[f'{task_name}/{split}']['accuracy'].compute().item()
                epoch_metrics[f'{task_name}/{split}/accuracy'] = acc
                if 'auc' in self.metrics[f'{task_name}/{split}']:
                    auc = self.metrics[f'{task_name}/{split}']['auc'].compute().item()
                    epoch_metrics[f'{task_name}/{split}/auc'] = auc
        
        return epoch_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'best_metric': self.best_metric,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.log_dir / 'checkpoint_latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.log_dir / 'checkpoint_best.pth')
            self.best_epoch = epoch
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
    
    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Task weights: {self.task_weights}")
        
        history = {
            'train': [],
            'val': [],
        }
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            history['train'].append(train_metrics)
            
            # Validate
            val_metrics = self.validate('val', epoch)
            history['val'].append(val_metrics)
            
            # Log validation metrics to TensorBoard
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Check if best model
            metric_name = self.checkpoint_metric.replace('/', '_')
            current_metric = val_metrics.get(metric_name, val_metrics.get('target/val/accuracy', 0.0))
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
            
            # Save checkpoint
            self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics.get('target/train/accuracy', 0):.4f}, "
                  f"AUC: {train_metrics.get('target/train/auc', 0):.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics.get('target/val/accuracy', 0):.4f}, "
                  f"AUC: {val_metrics.get('target/val/auc', 0):.4f}")
            if is_best:
                print(f"★ New best model! ({self.checkpoint_metric} = {current_metric:.4f})")
        
        # Save training history
        with open(self.log_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nTraining complete! Best {self.checkpoint_metric} = {self.best_metric:.4f} at epoch {self.best_epoch+1}")
        
        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_metrics = self.validate('test', num_epochs - 1)
        print(f"Test - Loss: {test_metrics['loss']:.4f}, "
              f"Acc: {test_metrics.get('target/test/accuracy', 0):.4f}, "
              f"AUC: {test_metrics.get('target/test/auc', 0):.4f}")
        
        # Log test metrics to TensorBoard
        for key, value in test_metrics.items():
            self.writer.add_scalar(f'Test/{key}', value, num_epochs - 1)
        
        # Close TensorBoard writer
        self.writer.close()
        
        return history


def main():
    parser = argparse.ArgumentParser(description='Train multi-task gaze model with PyTorch')
    
    # Data arguments
    parser.add_argument('--source', type=str, default='cxr2', choices=['cxr', 'cxr2', 'mets'],
                        help='Dataset source')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing DICOM image data')
    parser.add_argument('--gaze_raw_dir', type=str, default=None,
                        help='Path to PhysioNet 1.0.0 folder (gaze_raw) for proper data structure')
    parser.add_argument('--train_scale', type=float, default=1.0,
                        help='Fraction of training data to use')
    parser.add_argument('--val_scale', type=float, default=0.2,
                        help='Fraction of data for validation')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--gaze_mtl_task', type=str, default='diffusivity',
                        help='Helper task name(s), e.g., "diffusivity" or "loc_time"')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained ResNet50 weights')
    parser.add_argument('--task_weights', type=str, default='1.0',
                        help='Task weights as comma-separated values (target,helper1,helper2,...)')
    
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--l2', type=float, default=0.01,
                        help='L2 regularization (weight decay)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--lr_scheduler', type=str, default='cosine_annealing',
                        choices=['cosine_annealing', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--log_path', type=str, required=True,
                        help='Directory to save logs and checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
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
        gaze_raw_dir=getattr(args, 'gaze_raw_dir', None),
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
        gaze_raw_dir=getattr(args, 'gaze_raw_dir', None),
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
        gaze_raw_dir=getattr(args, 'gaze_raw_dir', None),
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
        log_dir=args.log_path,
        task_weights=task_weights,
        checkpoint_metric='target/val/accuracy',
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Train
    trainer.train(args.n_epochs)


if __name__ == '__main__':
    main()

