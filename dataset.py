"""
Dataset classes for Pix2Pix Manga Colorization

This module provides two dataset classes:
1. PairedImageDataset: For paired sketch-color image pairs
2. ColorToEdgeDataset: For color-only images with synthetic edge detection
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2


class PairedImageDataset(Dataset):
    """
    Dataset for paired sketch-color images (e.g., Anime Sketch Colorization Pair).
    
    Loads matching sketch and color images from separate directories,
    applies synchronized augmentations, and returns normalized tensors.
    
    Args:
        sketch_dir: Directory containing sketch/line-art images
        color_dir: Directory containing corresponding color images
        augment: Whether to apply data augmentation (default: True)
    """
    
    def __init__(self, sketch_dir: str, color_dir: str, augment: bool = True):
        self.sketch_dir = Path(sketch_dir)
        self.color_dir = Path(color_dir)
        self.augment = augment
        
        # Validate directories exist
        if not self.sketch_dir.exists():
            raise ValueError(f"Sketch directory not found: {sketch_dir}")
        if not self.color_dir.exists():
            raise ValueError(f"Color directory not found: {color_dir}")
        
        # Get and sort filenames
        sketch_files = sorted([f for f in self.sketch_dir.iterdir() 
                              if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        color_files = sorted([f for f in self.color_dir.iterdir() 
                             if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        if len(sketch_files) == 0:
            raise ValueError(f"No images found in sketch directory: {sketch_dir}")
        if len(color_files) == 0:
            raise ValueError(f"No images found in color directory: {color_dir}")
        
        # Validate pairing
        if len(sketch_files) != len(color_files):
            raise ValueError(
                f"Mismatched paired files: {len(sketch_files)} sketches vs "
                f"{len(color_files)} color images"
            )
        
        # Check that basenames match
        sketch_basenames = {f.stem for f in sketch_files}
        color_basenames = {f.stem for f in color_files}
        
        if sketch_basenames != color_basenames:
            missing_in_color = sketch_basenames - color_basenames
            missing_in_sketch = color_basenames - sketch_basenames
            error_msg = "Mismatched paired files:\n"
            if missing_in_color:
                error_msg += f"  Missing in color dir: {list(missing_in_color)[:5]}\n"
            if missing_in_sketch:
                error_msg += f"  Missing in sketch dir: {list(missing_in_sketch)[:5]}"
            raise ValueError(error_msg)
        
        # Store paired file paths
        self.pairs = list(zip(sketch_files, color_files))
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess a paired sketch-color image pair.
        
        Args:
            idx: Index of the pair to load
        
        Returns:
            Dictionary with keys:
                "A": Sketch tensor [1, 256, 256] normalized to [-1, 1]
                "B": Color tensor [3, 256, 256] normalized to [-1, 1]
        """
        sketch_path, color_path = self.pairs[idx]
        
        # Load images
        sketch_img = Image.open(sketch_path).convert('L')  # Grayscale
        color_img = Image.open(color_path).convert('RGB')  # RGB
        
        # Apply synchronized augmentations
        if self.augment:
            sketch_img, color_img = self._apply_augmentations(sketch_img, color_img)
        
        # Resize to 256x256
        sketch_img = TF.resize(sketch_img, (256, 256), interpolation=TF.InterpolationMode.BILINEAR)
        color_img = TF.resize(color_img, (256, 256), interpolation=TF.InterpolationMode.BILINEAR)
        
        # Convert to tensors
        sketch_tensor = TF.to_tensor(sketch_img)  # [1, 256, 256] in [0, 1]
        color_tensor = TF.to_tensor(color_img)    # [3, 256, 256] in [0, 1]
        
        # Normalize to [-1, 1]
        sketch_tensor = TF.normalize(sketch_tensor, mean=[0.5], std=[0.5])
        color_tensor = TF.normalize(color_tensor, mean=[0.5] * 3, std=[0.5] * 3)
        
        return {
            "A": sketch_tensor,  # [1, 256, 256]
            "B": color_tensor    # [3, 256, 256]
        }
    
    def _apply_augmentations(self, sketch_img: Image.Image, color_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Apply synchronized spatial augmentations to both images.
        
        Args:
            sketch_img: Sketch PIL Image
            color_img: Color PIL Image
        
        Returns:
            Tuple of (augmented_sketch, augmented_color)
        """
        # Random horizontal flip
        if random.random() > 0.5:
            sketch_img = TF.hflip(sketch_img)
            color_img = TF.hflip(color_img)
        
        # Random rotation (±5 degrees)
        angle = random.uniform(-5.0, 5.0)
        sketch_img = TF.rotate(sketch_img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=255)
        color_img = TF.rotate(color_img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=255)
        
        return sketch_img, color_img


class ColorToEdgeDataset(Dataset):
    """
    Dataset for color-only images with synthetic edge detection (e.g., Manga Dataset Images).
    
    Loads color images, generates line-art via Canny edge detection,
    applies synchronized augmentations, and returns normalized tensors.
    
    Args:
        color_dir: Directory containing color images
        augment: Whether to apply data augmentation (default: True)
        canny_low: Lower threshold for Canny edge detection (default: 50)
        canny_high: Upper threshold for Canny edge detection (default: 150)
        blur_kernel: Gaussian blur kernel size (0 to disable, default: 3)
        invert_edges: Whether to invert edges to black lines on white (default: True)
    """
    
    def __init__(
        self,
        color_dir: str,
        augment: bool = True,
        canny_low: int = 50,
        canny_high: int = 150,
        blur_kernel: int = 3,
        invert_edges: bool = True
    ):
        self.color_dir = Path(color_dir)
        self.augment = augment
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_kernel = blur_kernel
        self.invert_edges = invert_edges
        
        # Validate directory exists
        if not self.color_dir.exists():
            raise ValueError(f"Color directory not found: {color_dir}")
        
        # Get and sort filenames
        color_files = sorted([f for f in self.color_dir.iterdir() 
                             if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        if len(color_files) == 0:
            raise ValueError(f"No images found in color directory: {color_dir}")
        
        self.color_files = color_files
    
    def __len__(self) -> int:
        return len(self.color_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load color image, generate edge map, and preprocess both.
        
        Args:
            idx: Index of the image to load
        
        Returns:
            Dictionary with keys:
                "A": Edge/sketch tensor [1, 256, 256] normalized to [-1, 1]
                "B": Color tensor [3, 256, 256] normalized to [-1, 1]
        """
        color_path = self.color_files[idx]
        
        # Load color image
        color_img = Image.open(color_path).convert('RGB')
        
        # Generate edge map from color image
        sketch_img = self._generate_edge_map(color_img)
        
        # Apply synchronized augmentations
        if self.augment:
            sketch_img, color_img = self._apply_augmentations(sketch_img, color_img)
        
        # Resize to 256x256
        sketch_img = TF.resize(sketch_img, (256, 256), interpolation=TF.InterpolationMode.BILINEAR)
        color_img = TF.resize(color_img, (256, 256), interpolation=TF.InterpolationMode.BILINEAR)
        
        # Convert to tensors
        sketch_tensor = TF.to_tensor(sketch_img)  # [1, 256, 256] in [0, 1]
        color_tensor = TF.to_tensor(color_img)    # [3, 256, 256] in [0, 1]
        
        # Normalize to [-1, 1]
        sketch_tensor = TF.normalize(sketch_tensor, mean=[0.5], std=[0.5])
        color_tensor = TF.normalize(color_tensor, mean=[0.5] * 3, std=[0.5] * 3)
        
        return {
            "A": sketch_tensor,  # [1, 256, 256]
            "B": color_tensor    # [3, 256, 256]
        }
    
    def _generate_edge_map(self, color_img: Image.Image) -> Image.Image:
        """
        Generate edge map from color image using Canny edge detection.
        
        Args:
            color_img: RGB PIL Image
        
        Returns:
            Grayscale PIL Image with edge map
        """
        # Convert PIL to numpy array
        color_np = np.array(color_img)
        
        # Convert RGB to grayscale
        if len(color_np.shape) == 3:
            gray_np = cv2.cvtColor(color_np, cv2.COLOR_RGB2GRAY)
        else:
            gray_np = color_np
        
        # Apply Gaussian blur (optional)
        if self.blur_kernel > 0:
            gray_np = cv2.GaussianBlur(gray_np, (self.blur_kernel, self.blur_kernel), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray_np, self.canny_low, self.canny_high)
        
        # Invert edges (black lines on white background)
        if self.invert_edges:
            edges = 255 - edges
        
        # Convert back to PIL Image
        return Image.fromarray(edges, mode='L')
    
    def _apply_augmentations(self, sketch_img: Image.Image, color_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Apply synchronized spatial augmentations to both images.
        
        Args:
            sketch_img: Sketch/edge PIL Image
            color_img: Color PIL Image
        
        Returns:
            Tuple of (augmented_sketch, augmented_color)
        """
        # Random horizontal flip
        if random.random() > 0.5:
            sketch_img = TF.hflip(sketch_img)
            color_img = TF.hflip(color_img)
        
        # Random rotation (±5 degrees)
        angle = random.uniform(-5.0, 5.0)
        # Fill with white for sketch (inverted edges), white for color
        sketch_fill = 255 if self.invert_edges else 0
        sketch_img = TF.rotate(sketch_img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=sketch_fill)
        color_img = TF.rotate(color_img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=255)
        
        return sketch_img, color_img

