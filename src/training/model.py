"""
PyTorch model definition for image classification
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class DandelionGrassClassifier(nn.Module):
    """
    CNN model for binary classification (Dandelion vs Grass)
    Based on ResNet18 with custom head
    """
    
    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        """
        Initialize model
        
        Args:
            pretrained: Use pretrained ImageNet weights
            num_classes: Number of output classes (2 for binary)
        """
        super(DandelionGrassClassifier, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Get number of features from last layer
        num_features = self.backbone.fc.in_features
        
        # Replace final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)


def create_model(pretrained: bool = True) -> DandelionGrassClassifier:
    """
    Create and return a model instance
    
    Args:
        pretrained: Use pretrained weights
        
    Returns:
        Model instance
    """
    model = DandelionGrassClassifier(pretrained=pretrained)
    return model


def load_model(model_path: str, device: Optional[torch.device] = None) -> DandelionGrassClassifier:
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def save_model(model: DandelionGrassClassifier, save_path: str):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        save_path: Path to save checkpoint
    """
    torch.save(model.state_dict(), save_path)
