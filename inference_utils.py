"""
Shared inference utilities for Pix2Pix manga colorization.

Provides high-resolution tiling support so we can colorize large
images without downscaling them to 256x256, avoiding the low-res
artifacts that come from resizing everything before inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image
import torch

from preprocessing import SKETCH_TRANSFORM


@dataclass(frozen=True)
class TilingConfig:
    tile_size: int = 256
    overlap: int = 128  # Increased significantly for better blending
    blend_mode: str = "feather"  # "feather" or "average"
    use_color_consistency: bool = False  # Disabled - use simpler approach

    @property
    def stride(self) -> int:
        stride = self.tile_size - self.overlap
        if stride <= 0:
            raise ValueError("Overlap must be smaller than tile_size")
        return stride


def _compute_starts(full: int, tile: int, stride: int) -> List[int]:
    if full <= tile:
        return [0]

    starts = list(range(0, full - tile, stride))
    starts.append(full - tile)
    return starts


def _patch_to_tensor(patch_np: np.ndarray, device: torch.device) -> torch.Tensor:
    patch_img = Image.fromarray(patch_np.astype(np.uint8), mode="L")
    tensor = SKETCH_TRANSFORM(patch_img)
    return tensor.unsqueeze(0).to(device)


def _build_grayscale_color_map(reference_gray: np.ndarray, reference_colors: np.ndarray, bins: int = 256) -> dict:
    """
    Build a mapping from grayscale values to colors using the reference tile.
    
    This creates a lookup table that ensures consistent colors for similar grayscale values.
    Uses vectorized operations for speed.
    """
    # Flatten arrays
    gray_flat = reference_gray.flatten().astype(np.int32)
    color_flat = reference_colors.reshape(-1, 3)
    
    # Clip grayscale values
    gray_flat = np.clip(gray_flat, 0, bins - 1)
    
    # Build mapping: for each grayscale value, compute average color
    gray_map = {}
    
    for bin_val in range(bins):
        # Find all pixels with this grayscale value
        mask = gray_flat == bin_val
        if np.any(mask):
            gray_map[bin_val] = np.mean(color_flat[mask], axis=0)
        else:
            gray_map[bin_val] = None
    
    # Fill in missing bins by finding nearest valid bin
    valid_bins = np.array([b for b, c in gray_map.items() if c is not None])
    valid_colors = np.array([gray_map[b] for b in valid_bins])
    
    for bin_val in range(bins):
        if gray_map[bin_val] is None:
            # Find nearest valid bin
            distances = np.abs(valid_bins - bin_val)
            nearest_idx = np.argmin(distances)
            gray_map[bin_val] = valid_colors[nearest_idx]
    
    return gray_map


def _apply_tile_color_consistency(tile_predictions: dict, gray_image: np.ndarray) -> dict:
    """
    Apply color consistency across tiles by detecting uniform regions
    and forcing them to use consistent colors.
    
    Strategy: Find uniform grayscale regions across the image and
    average their colors across all tiles to eliminate prismatic effects.
    """
    # First, detect uniform regions in the full image
    h, w = gray_image.shape
    uniform_mask = np.zeros((h, w), dtype=bool)
    
    # Simple uniform region detection: low local variance
    kernel_size = 20
    half_k = kernel_size // 2
    
    for y in range(half_k, h - half_k, 5):  # Sample every 5 pixels
        for x in range(half_k, w - half_k, 5):
            window = gray_image[y - half_k:y + half_k, x - half_k:x + half_k]
            if np.std(window) < 15:  # Uniform region
                uniform_mask[y - half_k:y + half_k, x - half_k:x + half_k] = True
    
    # For uniform regions, compute average color across all overlapping tiles
    uniform_colors = {}
    
    for (y, x), (pred_np, gray_patch) in tile_predictions.items():
        tile_y_end = min(y + pred_np.shape[0], h)
        tile_x_end = min(x + pred_np.shape[1], w)
        
        tile_uniform = uniform_mask[y:tile_y_end, x:tile_x_end]
        
        if np.any(tile_uniform):
            # Get colors in uniform regions
            pred_cropped = pred_np[:tile_y_end-y, :tile_x_end-x]
            uniform_colors_tile = pred_cropped[tile_uniform]
            
            # Store for averaging
            for i, (uy, ux) in enumerate(np.argwhere(tile_uniform)):
                global_y = y + uy
                global_x = x + ux
                if (global_y, global_x) not in uniform_colors:
                    uniform_colors[(global_y, global_x)] = []
                uniform_colors[(global_y, global_x)].append(uniform_colors_tile[i])
    
    # Average colors for each uniform pixel
    avg_uniform_colors = {}
    for pos, colors in uniform_colors.items():
        avg_uniform_colors[pos] = np.mean(colors, axis=0)
    
    # Apply averaged colors to uniform regions in each tile
    corrected_tiles = {}
    for (y, x), (pred_np, gray_patch) in tile_predictions.items():
        corrected = pred_np.copy()
        tile_y_end = min(y + pred_np.shape[0], h)
        tile_x_end = min(x + pred_np.shape[1], w)
        
        tile_uniform = uniform_mask[y:tile_y_end, x:tile_x_end]
        
        if np.any(tile_uniform):
            for uy, ux in np.argwhere(tile_uniform):
                global_y = y + uy
                global_x = x + ux
                if (global_y, global_x) in avg_uniform_colors:
                    # Blend: 80% averaged color, 20% original
                    corrected[uy, ux] = pred_np[uy, ux] * 0.2 + avg_uniform_colors[(global_y, global_x)] * 0.8
        
        corrected_tiles[(y, x)] = (np.clip(corrected, 0.0, 1.0), gray_patch)
    
    return corrected_tiles


def _create_feather_mask(tile_size: int, overlap: int) -> np.ndarray:
    """
    Create a feathering mask for smooth blending at tile edges.
    
    Returns a mask where:
    - Center regions have weight 1.0
    - Edge regions (overlap/2 pixels) fade from low weight (edge) to 1.0 (center)
    - This ensures smooth blending in overlapping regions
    """
    mask = np.ones((tile_size, tile_size), dtype=np.float32)
    feather_size = overlap // 2
    
    if feather_size == 0:
        return mask
    
    # Create distance-based weights: low at edges, 1.0 at center
    # Horizontal feathering (left and right edges)
    for i in range(feather_size):
        # Distance from edge: i=0 is at edge, i=feather_size-1 is at center
        # Weight increases linearly from ~0.1 at edge to 1.0 at center
        weight = 0.1 + 0.9 * (i + 1) / feather_size
        mask[:, i] = np.minimum(mask[:, i], weight)
        mask[:, tile_size - 1 - i] = np.minimum(mask[:, tile_size - 1 - i], weight)
    
    # Vertical feathering (top and bottom edges)
    for i in range(feather_size):
        weight = 0.1 + 0.9 * (i + 1) / feather_size
        mask[i, :] = np.minimum(mask[i, :], weight)
        mask[tile_size - 1 - i, :] = np.minimum(mask[tile_size - 1 - i, :], weight)
    
    # Corner handling: use minimum of horizontal and vertical distances
    for i in range(feather_size):
        for j in range(feather_size):
            h_weight = 0.1 + 0.9 * (i + 1) / feather_size
            v_weight = 0.1 + 0.9 * (j + 1) / feather_size
            corner_weight = min(h_weight, v_weight)
            mask[i, j] = corner_weight
            mask[i, tile_size - 1 - j] = corner_weight
            mask[tile_size - 1 - i, j] = corner_weight
            mask[tile_size - 1 - i, tile_size - 1 - j] = corner_weight
    
    return mask


@torch.no_grad()
def colorize_with_tiling(
    gray_image: Image.Image,
    netG: torch.nn.Module,
    device: torch.device,
    config: TilingConfig | None = None,
) -> torch.Tensor:
    """
    Colorize a grayscale PIL image using overlapping tiles with feathering for seamless blending.

    Returns a tensor in [-1, 1] of shape [1, 3, H, W] without
    resizing the original image.
    """
    if config is None:
        config = TilingConfig()

    gray_np = np.array(gray_image.convert("L"))
    orig_h, orig_w = gray_np.shape

    pad_h = (config.tile_size - orig_h % config.tile_size) % config.tile_size
    pad_w = (config.tile_size - orig_w % config.tile_size) % config.tile_size

    padded = np.pad(
        gray_np,
        pad_width=((0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=255,
    )

    padded_h, padded_w = padded.shape
    stride = config.stride

    # Create feather mask if using feather blending
    if config.blend_mode == "feather":
        feather_mask = _create_feather_mask(config.tile_size, config.overlap)
    else:
        feather_mask = np.ones((config.tile_size, config.tile_size), dtype=np.float32)

    accum = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
    weights = np.zeros((padded_h, padded_w), dtype=np.float32)

    y_starts = _compute_starts(padded_h, config.tile_size, stride)
    x_starts = _compute_starts(padded_w, config.tile_size, stride)

    # Process tiles with simple averaging - increased overlap helps reduce artifacts
    for y in y_starts:
        for x in x_starts:
            patch = padded[y : y + config.tile_size, x : x + config.tile_size]
            tensor = _patch_to_tensor(patch, device)
            pred = netG(tensor)
            pred = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)

            pred_np = (
                pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            )  # [H, W, 3]

            # Apply feather mask for weighted blending
            mask_3d = feather_mask[:, :, np.newaxis]  # [H, W, 1]
            weighted_pred = pred_np * mask_3d

            accum[y : y + config.tile_size, x : x + config.tile_size] += weighted_pred
            weights[y : y + config.tile_size, x : x + config.tile_size] += feather_mask

    # Normalize by weights (avoid division by zero)
    weights = np.maximum(weights, 1e-8)
    merged = accum / weights[:, :, np.newaxis]
    merged = merged[:orig_h, :orig_w]

    # Simple post-processing: apply gentle blur to reduce color inconsistencies
    # This is much simpler and faster than complex consistency algorithms
    try:
        from scipy import ndimage
        # Apply very gentle Gaussian blur to smooth out color inconsistencies
        merged = ndimage.gaussian_filter(merged, sigma=1.0, axes=(0, 1))
    except ImportError:
        # If scipy not available, use simple box blur with numpy
        # Simple 3x3 box blur
        h, w = merged.shape[:2]
        blurred = merged.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                blurred[y, x] = np.mean(merged[y-1:y+2, x-1:x+2], axis=(0, 1))
        merged = blurred * 0.3 + merged * 0.7  # Blend

    output = torch.from_numpy(merged).permute(2, 0, 1).unsqueeze(0)
    output = torch.clamp(output * 2.0 - 1.0, -1.0, 1.0)
    return output.to(device)


def _smooth_uniform_regions(color_img: np.ndarray, gray_img: np.ndarray, threshold: float = 30.0) -> np.ndarray:
    """
    Smooth color inconsistencies in uniform grayscale regions.
    
    This helps reduce the prismatic effect where uniform backgrounds
    get different colors in different tiles.
    
    Simplified version for better performance - applies gentle blur to entire image
    in areas with low grayscale variance.
    """
    h, w = gray_img.shape
    
    # Simplified: use a smaller kernel and sample every few pixels for speed
    sample_step = 3  # Process every 3rd pixel for variance calculation
    kernel_size = 9
    half_k = kernel_size // 2
    
    gray_float = gray_img.astype(np.float32)
    
    # Calculate variance at sampled points
    variance_map = np.full((h, w), 100.0, dtype=np.float32)  # Default: not uniform
    
    for y in range(half_k, h - half_k, sample_step):
        for x in range(half_k, w - half_k, sample_step):
            window = gray_float[y - half_k:y + half_k + 1, x - half_k:x + half_k + 1]
            var_val = np.var(window)
            # Fill surrounding area with this variance value
            y_end = min(y + sample_step, h - half_k)
            x_end = min(x + sample_step, w - half_k)
            variance_map[y:y_end, x:x_end] = var_val
    
    # Create mask for uniform regions
    uniform_mask = variance_map < threshold
    uniform_mask = uniform_mask[:, :, np.newaxis]  # [H, W, 1]
    
    # Simple box blur for all channels (faster)
    blur_size = 7
    blur_half = blur_size // 2
    
    # Pad and blur
    color_padded = np.pad(color_img, ((blur_half, blur_half), (blur_half, blur_half), (0, 0)), mode='edge')
    color_smooth = np.zeros_like(color_img)
    
    # Vectorized blur using stride tricks
    for y in range(0, h, 2):  # Process every 2nd row for speed
        for x in range(0, w, 2):  # Process every 2nd column for speed
            window = color_padded[y:y+blur_size, x:x+blur_size]
            color_smooth[y, x] = np.mean(window, axis=(0, 1))
            # Fill nearby pixels
            if y + 1 < h:
                color_smooth[y+1, x] = color_smooth[y, x]
            if x + 1 < w:
                color_smooth[y, x+1] = color_smooth[y, x]
            if y + 1 < h and x + 1 < w:
                color_smooth[y+1, x+1] = color_smooth[y, x]
    
    # Fill remaining pixels
    for y in range(h):
        for x in range(w):
            if color_smooth[y, x].sum() == 0:
                # Use nearest processed pixel
                y_src = (y // 2) * 2
                x_src = (x // 2) * 2
                if y_src < h and x_src < w:
                    color_smooth[y, x] = color_smooth[y_src, x_src]
    
    # Blend: use smoothed colors in uniform regions
    blend_factor = 0.3  # Reduced for subtler effect
    result = color_img * (1.0 - uniform_mask * blend_factor) + color_smooth * (uniform_mask * blend_factor)
    
    return np.clip(result, 0.0, 1.0)

