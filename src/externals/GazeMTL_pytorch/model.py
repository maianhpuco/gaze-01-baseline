"""
Modern PyTorch implementation of multi-task learning model for gaze-based medical image classification.
Uses ResNet50 backbone with task-specific heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Optional


class MultiTaskModel(nn.Module):
    """
    Multi-task learning model with shared ResNet50 encoder and task-specific heads.
    
    Args:
        num_classes: Number of classes for target task
        helper_output_dims: List of output dimensions for helper tasks
        pretrained: Whether to use pretrained ResNet50 weights
        freeze_backbone: Whether to freeze the backbone during training
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        helper_output_dims: Optional[List[int]] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.helper_output_dims = helper_output_dims or []
        self.num_helper_tasks = len(self.helper_output_dims)
        
        # Load pretrained ResNet50 and remove final FC layer
        backbone = models.resnet50(weights='DEFAULT' if pretrained else None)
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.num_features = 2048
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Task-specific classification heads
        self.target_head = nn.Linear(self.num_features, num_classes)
        
        # Helper task heads
        self.helper_heads = nn.ModuleList([
            nn.Linear(self.num_features, dim) 
            for dim in self.helper_output_dims
        ])
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input images of shape [batch_size, 3, H, W]
            
        Returns:
            Dictionary with keys:
                - 'target': logits for target task [batch_size, num_classes]
                - 'helper_task_0', 'helper_task_1', ...: logits for helper tasks
        """
        # Extract features using shared backbone
        features = self.backbone(x)
        # Flatten spatial dimensions: [batch_size, 2048, 1, 1] -> [batch_size, 2048]
        features = features.view(features.size(0), -1)
        
        # Get predictions for each task
        outputs = {
            'target': self.target_head(features)
        }
        
        # Add helper task outputs
        for i, head in enumerate(self.helper_heads):
            outputs[f'helper_task_{i}'] = head(features)
        
        return outputs
    
    def get_target_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions for target task."""
        outputs = self.forward(x)
        return F.softmax(outputs['target'], dim=1)
    
    def get_target_preds(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions for target task."""
        outputs = self.forward(x)
        return outputs['target'].argmax(dim=1)

