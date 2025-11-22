"""
Utility functions for Pix2Pix Manga Colorization

This module provides helper functions for weight initialization, image processing,
saving, reproducibility, and device management.
"""

import random
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image as tv_save_image


def init_weights(model: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """
    Initialize network weights using the specified initialization method.
    
    This function applies weight initialization recursively to all submodules.
    Supports normal, xavier, kaiming, and orthogonal initialization.
    
    Args:
        model: PyTorch model to initialize
        init_type: Type of initialization ("normal", "xavier", "kaiming", "orthogonal")
        init_gain: Gain factor for initialization (default: 0.02)
    
    Usage:
        model.apply(lambda m: init_weights(m, init_type="normal", init_gain=0.02))
    """
    classname = model.__class__.__name__
    
    # Initialize Conv2d and ConvTranspose2d layers
    if hasattr(model, 'weight') and (classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1):
        if init_type == "normal":
            nn.init.normal_(model.weight.data, 0.0, init_gain)
        elif init_type == "xavier":
            nn.init.xavier_normal_(model.weight.data, gain=init_gain)
        elif init_type == "kaiming":
            nn.init.kaiming_normal_(model.weight.data, a=0, mode='fan_in')
        elif init_type == "orthogonal":
            nn.init.orthogonal_(model.weight.data, gain=init_gain)
        else:
            raise NotImplementedError(f"Initialization type '{init_type}' is not implemented")
        
        # Set bias to zero if it exists
        if hasattr(model, 'bias') and model.bias is not None:
            nn.init.constant_(model.bias.data, 0.0)
    
    # Initialize BatchNorm and InstanceNorm layers
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        if hasattr(model, 'weight') and model.weight is not None:
            nn.init.normal_(model.weight.data, 1.0, init_gain)
        if hasattr(model, 'bias') and model.bias is not None:
            nn.init.constant_(model.bias.data, 0.0)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize tensor from [-1, 1] range to [0, 1] range.
    
    This function is used to convert model outputs or normalized inputs
    back to a displayable format.
    
    Args:
        tensor: Input tensor in range [-1, 1]
    
    Returns:
        Denormalized tensor in range [0, 1] (clipped to avoid floating point errors)
    """
    # Denormalize: (x + 1) / 2
    denorm = (tensor + 1.0) / 2.0
    
    # Clip to [0, 1] to avoid floating point errors
    denorm = torch.clamp(denorm, 0.0, 1.0)
    
    return denorm


def save_image(tensor: torch.Tensor, filepath: Union[str, Path], nrow: int = 8) -> None:
    """
    Save tensor image(s) to disk.
    
    Automatically handles denormalization and creates parent directories.
    Supports both single images [C, H, W] and batches [B, C, H, W].
    
    Args:
        tensor: Image tensor of shape [C, H, W] or [B, C, H, W]
        filepath: Path where the image should be saved
        nrow: Number of images per row in grid (for batch saving, default: 8)
    """
    filepath = Path(filepath)
    
    # Create parent directories if they don't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Denormalize tensor from [-1, 1] to [0, 1]
    tensor = denormalize(tensor)
    
    # Save using torchvision
    tv_save_image(tensor, str(filepath), nrow=nrow, padding=2, normalize=False)


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across random, numpy, and PyTorch.
    
    This function ensures that experiments can be reproduced by setting
    seeds for all random number generators used in the project.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic cuDNN behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the appropriate device (CUDA if available, else CPU).
    
    Returns:
        torch.device object representing the computation device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

