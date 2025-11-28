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
    overlap: int = 64

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


@torch.no_grad()
def colorize_with_tiling(
    gray_image: Image.Image,
    netG: torch.nn.Module,
    device: torch.device,
    config: TilingConfig | None = None,
) -> torch.Tensor:
    """
    Colorize a grayscale PIL image using overlapping tiles.

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

    accum = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
    counts = np.zeros((padded_h, padded_w, 3), dtype=np.float32)

    y_starts = _compute_starts(padded_h, config.tile_size, stride)
    x_starts = _compute_starts(padded_w, config.tile_size, stride)

    for y in y_starts:
        for x in x_starts:
            patch = padded[y : y + config.tile_size, x : x + config.tile_size]
            tensor = _patch_to_tensor(patch, device)
            pred = netG(tensor)
            pred = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)

            pred_np = (
                pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            )  # [H, W, 3]

            accum[y : y + config.tile_size, x : x + config.tile_size] += pred_np
            counts[y : y + config.tile_size, x : x + config.tile_size] += 1.0

    counts[counts == 0] = 1.0
    merged = accum / counts
    merged = merged[:orig_h, :orig_w]

    output = torch.from_numpy(merged).permute(2, 0, 1).unsqueeze(0)
    output = torch.clamp(output * 2.0 - 1.0, -1.0, 1.0)
    return output.to(device)

