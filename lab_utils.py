"""
LAB Color Space Utilities for Manga Colorization

Converts between RGB and LAB color spaces for training and inference.
LAB separates luminance (L) from color (A, B channels), which is more
natural for colorization tasks.
"""

import warnings
import torch
import numpy as np
from typing import Tuple


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor to LAB color space.
    
    Args:
        rgb: RGB tensor [B, 3, H, W] in range [-1, 1] (normalized)
    
    Returns:
        LAB tensor [B, 3, H, W] where:
        - L: [0, 100] (luminance)
        - A: [-127, 127] (green-red axis)
        - B: [-127, 127] (blue-yellow axis)
    """
    # Convert from [-1, 1] to [0, 1]
    rgb = (rgb + 1.0) / 2.0
    rgb = torch.clamp(rgb, 0.0, 1.0)
    
    # Convert to numpy for skimage (detach to avoid gradient issues)
    rgb_np = rgb.permute(0, 2, 3, 1).cpu().detach().numpy()  # [B, H, W, 3]
    # Convert to uint8 to match training (yes, this causes quantization, but model was trained with it!)
    rgb_np = (rgb_np * 255).astype(np.uint8)
    
    # Convert RGB to LAB using skimage (suppress warnings about clipped values)
    from skimage import color
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*negative Z values.*')
        lab_np = color.rgb2lab(rgb_np)  # L: [0, 100], A: [-127, 127], B: [-127, 127]
    
    # Convert back to tensor  
    lab = torch.from_numpy(lab_np).permute(0, 3, 1, 2).float()  # [B, 3, H, W]
    
    return lab.to(rgb.device)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """
    Convert LAB tensor to RGB color space.
    
    Args:
        lab: LAB tensor [B, 3, H, W] where L: [0, 100], A/B: [-127, 127]
    
    Returns:
        RGB tensor [B, 3, H, W] in range [-1, 1] (normalized)
    """
    # Convert to numpy (detach to avoid gradient issues)
    lab_np = lab.permute(0, 2, 3, 1).cpu().detach().numpy()  # [B, H, W, 3]
    
    # Convert LAB to RGB using skimage (suppress warnings about clipped values)
    from skimage import color
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*negative Z values.*')
        rgb_np = color.lab2rgb(lab_np)  # [0, 1]
    
    # Clamp to valid range
    rgb_np = np.clip(rgb_np, 0.0, 1.0)
    
    # Convert to tensor and normalize to [-1, 1]
    rgb = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()  # [B, 3, H, W]
    rgb = rgb * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    
    return rgb.to(lab.device)


def normalize_lab(lab: torch.Tensor) -> torch.Tensor:
    """
    Normalize LAB tensor to [-1, 1] range for training.
    
    Args:
        lab: LAB tensor [B, 3, H, W] where L: [0, 100], A/B: [-127, 127]
    
    Returns:
        Normalized LAB tensor [B, 3, H, W] in range [-1, 1]
    """
    # Normalize L: [0, 100] -> [-1, 1]
    # Normalize A, B: [-127, 127] -> [-1, 1]
    normalized = lab.clone()
    normalized[:, 0:1, :, :] = (lab[:, 0:1, :, :] / 50.0) - 1.0  # L: [0, 100] -> [-1, 1]
    normalized[:, 1:3, :, :] = lab[:, 1:3, :, :] / 127.0  # A, B: [-127, 127] -> [-1, 1]
    
    return normalized


def denormalize_lab(lab_normalized: torch.Tensor) -> torch.Tensor:
    """
    Denormalize LAB tensor from [-1, 1] back to original range.
    
    Args:
        lab_normalized: Normalized LAB tensor [B, 3, H, W] in range [-1, 1]
    
    Returns:
        LAB tensor [B, 3, H, W] where L: [0, 100], A/B: [-127, 127]
    """
    lab = lab_normalized.clone()
    lab[:, 0:1, :, :] = (lab_normalized[:, 0:1, :, :] + 1.0) * 50.0  # L: [-1, 1] -> [0, 100]
    lab[:, 1:3, :, :] = lab_normalized[:, 1:3, :, :] * 127.0  # A, B: [-1, 1] -> [-127, 127]
    
    return lab


def extract_l_channel(lab: torch.Tensor) -> torch.Tensor:
    """
    Extract L (luminance) channel from LAB tensor.
    
    Args:
        lab: LAB tensor [B, 3, H, W]
    
    Returns:
        L channel [B, 1, H, W]
    """
    return lab[:, 0:1, :, :]


def extract_ab_channels(lab: torch.Tensor) -> torch.Tensor:
    """
    Extract AB (color) channels from LAB tensor.
    
    Args:
        lab: LAB tensor [B, 3, H, W]
    
    Returns:
        AB channels [B, 2, H, W]
    """
    return lab[:, 1:3, :, :]


def combine_l_ab(l: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """
    Combine L and AB channels into LAB tensor.
    
    Args:
        l: L channel [B, 1, H, W]
        ab: AB channels [B, 2, H, W]
    
    Returns:
        LAB tensor [B, 3, H, W]
    """
    return torch.cat([l, ab], dim=1)

