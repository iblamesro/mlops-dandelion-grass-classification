"""
Unit tests for model
"""

import pytest
import torch
from src.training.model import DandelionGrassClassifier, create_model


def test_model_creation():
    """Test model can be created"""
    model = create_model(pretrained=False)
    assert model is not None
    assert isinstance(model, DandelionGrassClassifier)


def test_model_forward_pass():
    """Test model forward pass"""
    model = create_model(pretrained=False)
    model.eval()
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, 2)


def test_model_output_range():
    """Test model output can be converted to probabilities"""
    model = create_model(pretrained=False)
    model.eval()
    
    x = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(x)
        probabilities = torch.softmax(output, dim=1)
    
    # Check probabilities sum to 1
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0), atol=1e-6)
    
    # Check probabilities are in [0, 1]
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()


def test_model_different_batch_sizes():
    """Test model works with different batch sizes"""
    model = create_model(pretrained=False)
    model.eval()
    
    batch_sizes = [1, 2, 4, 8, 16]
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 2)
