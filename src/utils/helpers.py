"""
Utility functions for the MLOps project
"""

import os
import io
from pathlib import Path
from typing import Optional, Tuple
import hashlib

import requests
from PIL import Image
import numpy as np
import torch
from loguru import logger


def download_image(url: str, timeout: int = 10) -> Optional[Image.Image]:
    """
    Download an image from a URL
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
        
    Returns:
        PIL Image or None if download fails
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        return image
        
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None


def validate_image(image: Image.Image, min_size: Tuple[int, int] = (32, 32)) -> bool:
    """
    Validate an image meets minimum requirements
    
    Args:
        image: PIL Image
        min_size: Minimum (width, height)
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check if image can be loaded
        image.verify()
        
        # Reload image after verify
        image = image.copy()
        
        # Check dimensions
        if image.size[0] < min_size[0] or image.size[1] < min_size[1]:
            return False
        
        # Check mode (convert if necessary)
        if image.mode not in ['RGB', 'L']:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False


def preprocess_image(
    image: Image.Image, 
    target_size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image
        target_size: Target (width, height)
        
    Returns:
        Preprocessed PIL Image
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size, Image.LANCZOS)
    
    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to PyTorch tensor
    
    Args:
        image: PIL Image
        
    Returns:
        PyTorch tensor
    """
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Transpose to (C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Convert to tensor
    tensor = torch.from_numpy(img_array)
    
    return tensor


def calculate_image_hash(image: Image.Image) -> str:
    """
    Calculate hash of an image for deduplication
    
    Args:
        image: PIL Image
        
    Returns:
        MD5 hash string
    """
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_hash = hashlib.md5(img_bytes.getvalue()).hexdigest()
    return img_hash


def get_model_path(model_name: str, version: Optional[str] = None) -> Path:
    """
    Get path for a model file
    
    Args:
        model_name: Name of the model
        version: Optional version string
        
    Returns:
        Path to model file
    """
    from src.config import settings
    
    if version:
        filename = f"{model_name}_{version}.pth"
    else:
        filename = f"{model_name}_latest.pth"
    
    return settings.MODELS_DIR / filename


def setup_logger(log_file: Optional[str] = None):
    """
    Setup loguru logger
    
    Args:
        log_file: Optional log file path
    """
    from src.config import settings
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler if specified
    if log_file:
        log_path = settings.LOGS_DIR / log_file
        logger.add(
            log_path,
            rotation="500 MB",
            retention="10 days",
            level="DEBUG"
        )


import sys
