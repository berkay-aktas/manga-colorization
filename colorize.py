"""
Inference script for Pix2Pix Manga Colorization

This script loads a trained Generator checkpoint and colorizes black-and-white
manga sketch/line-art images. Supports both single image and batch folder processing.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from inference_utils import TilingConfig, colorize_with_tiling
from networks import UNetGenerator
from preprocessing import SKETCH_TRANSFORM
from utils import get_device, save_image


def load_model(
    checkpoint_path: str,
    in_channels: int = 1,
    out_channels: int = 3,
    device: torch.device = None,
) -> UNetGenerator:
    """
    Load trained Generator model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pth)
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 3)
        device: Device to load model on (default: auto-detect)
    
    Returns:
        Loaded and initialized Generator model in eval mode
    """
    if device is None:
        device = get_device()
    
    # Initialize model
    netG = UNetGenerator(in_channels=in_channels, out_channels=out_channels, ngf=64)
    netG = netG.to(device)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    state = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "netG_state_dict" in state:
        netG.load_state_dict(state["netG_state_dict"])
    else:
        # Assume the state dict is the model state directly
        netG.load_state_dict(state)
    
    # Set to evaluation mode
    netG.eval()
    
    print(f"Loaded model from: {checkpoint_path}")
    print(f"Model parameters: {sum(p.numel() for p in netG.parameters()):,}")
    
    return netG


def preprocess_pil_image(img: Image.Image) -> torch.Tensor:
    """
    Preprocess input image to match training format.
    """
    if img.mode != 'L':
        img = img.convert('L')
    
    tensor = SKETCH_TRANSFORM(img)
    tensor = tensor.unsqueeze(0)
    
    return tensor


def preprocess_image(image_path: str) -> torch.Tensor:
    """Backward-compatible wrapper that loads the image then preprocesses it."""
    img = Image.open(image_path)
    return preprocess_pil_image(img)


def infer_single_image(
    netG: UNetGenerator,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Run inference on a single preprocessed image.
    
    Args:
        netG: Generator model
        input_tensor: Preprocessed input tensor [1, 1, 256, 256]
        device: Device to run inference on
    
    Returns:
        Colorized output tensor [1, 3, 256, 256] in range [-1, 1]
    """
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        fake_B = netG(input_tensor)
    
    return fake_B


def process_single_image(
    netG: UNetGenerator,
    input_path: str,
    output_dir: str,
    device: torch.device,
    tiling_config: TilingConfig,
    enable_tiling: bool,
) -> None:
    """
    Process a single input image and save colorized output.
    
    Args:
        netG: Generator model
        input_path: Path to input image
        output_dir: Directory to save output
        device: Device to run inference on
    """
    img = Image.open(input_path).convert("L")

    if enable_tiling and (
        img.width > tiling_config.tile_size or img.height > tiling_config.tile_size
    ):
        output_tensor = colorize_with_tiling(
            gray_image=img,
            netG=netG,
            device=device,
            config=tiling_config,
        )
    else:
        input_tensor = preprocess_pil_image(img)
        output_tensor = infer_single_image(netG, input_tensor, device)
    
    # Generate output filename
    input_basename = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{input_basename}_colorized.png")
    
    # Save colorized image
    save_image(output_tensor, output_path)
    
    print(f"Saved colorized image to: {output_path}")


def get_image_files(input_path: str) -> List[str]:
    """
    Get list of image files from input path (file or directory).
    
    Args:
        input_path: Path to image file or directory
    
    Returns:
        List of image file paths
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            return [str(input_path)]
        else:
            raise ValueError(f"Unsupported image format: {input_path.suffix}")
    
    elif input_path.is_dir():
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if len(image_files) == 0:
            raise ValueError(f"No image files found in directory: {input_path}")
        
        return sorted([str(f) for f in image_files])
    
    else:
        raise ValueError(f"Invalid input path: {input_path}")


def process_folder(
    netG: UNetGenerator,
    input_path: str,
    output_dir: str,
    device: torch.device,
    tiling_config: TilingConfig,
    enable_tiling: bool,
) -> None:
    """
    Process all images in a folder.
    
    Args:
        netG: Generator model
        input_path: Path to input directory
        output_dir: Directory to save outputs
        device: Device to run inference on
    """
    image_files = get_image_files(input_path)
    
    print(f"Found {len(image_files)} image(s) to process")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing [{i}/{len(image_files)}]: {Path(image_path).name}")
        process_single_image(
            netG=netG,
            input_path=image_path,
            output_dir=output_dir,
            device=device,
            tiling_config=tiling_config,
            enable_tiling=enable_tiling,
        )


def main() -> None:
    """Main function for inference script."""
    parser = argparse.ArgumentParser(
        description="Colorize manga sketch/line-art images using trained Pix2Pix model"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image file or directory containing images"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="inference_outputs",
        help="Directory to save colorized outputs (default: inference_outputs)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Generator checkpoint file (.pth)"
    )

    parser.add_argument(
        "--tile_size",
        type=int,
        default=256,
        help="Tile size for high-resolution inference (default: 256)",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Tile overlap/padding to reduce seams (default: 64)",
    )

    parser.add_argument(
        "--disable_tiling",
        action="store_true",
        help="Force classic 256x256 resizing instead of tiling",
    )
    
    parser.add_argument(
        "--in_channels",
        type=int,
        default=1,
        help="Number of input channels (default: 1)"
    )
    
    parser.add_argument(
        "--out_channels",
        type=int,
        default=3,
        help="Number of output channels (default: 3)"
    )
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    # Load model
    print("\n" + "=" * 60)
    print("Loading model...")
    print("=" * 60)
    netG = load_model(
        checkpoint_path=args.checkpoint,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        device=device,
    )
    
    # Process input
    print("\n" + "=" * 60)
    print("Processing images...")
    print("=" * 60)
    
    input_path = Path(args.input)
    
    tiling_config = TilingConfig(tile_size=args.tile_size, overlap=args.overlap)
    enable_tiling = not args.disable_tiling

    if input_path.is_file():
        process_single_image(
            netG=netG,
            input_path=args.input,
            output_dir=args.output,
            device=device,
            tiling_config=tiling_config,
            enable_tiling=enable_tiling,
        )
    elif input_path.is_dir():
        process_folder(
            netG=netG,
            input_path=args.input,
            output_dir=args.output,
            device=device,
            tiling_config=tiling_config,
            enable_tiling=enable_tiling,
        )
    else:
        raise ValueError(f"Invalid input path: {args.input}")
    
    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
    