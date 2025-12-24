"""
Create Black & White versions from Colored Manga Images for Training

This script processes a directory of colored manga images and creates
paired B/W versions for training. It generates high-quality grayscale
conversions that preserve line art details.
"""

import os
from pathlib import Path
from typing import List, Tuple
import argparse

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_to_grayscale(color_img: Image.Image, method: str = "weighted") -> Image.Image:
    """
    Convert colored image to grayscale using different methods.
    
    Args:
        color_img: PIL RGB Image
        method: Conversion method - "weighted" (default), "luminance", or "simple"
    
    Returns:
        PIL Grayscale Image
    """
    if color_img.mode != 'RGB':
        color_img = color_img.convert('RGB')
    
    color_np = np.array(color_img)
    
    if method == "weighted":
        # Weighted grayscale (preserves perceived brightness better)
        # Standard weights: 0.299*R + 0.587*G + 0.114*B
        gray = cv2.cvtColor(color_np, cv2.COLOR_RGB2GRAY)
    elif method == "luminance":
        # Luminance-based (better for manga)
        gray = np.dot(color_np[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:  # simple
        # Simple average
        gray = np.mean(color_np, axis=2).astype(np.uint8)
    
    return Image.fromarray(gray, mode='L')


def enhance_line_art(gray_img: Image.Image, 
                    contrast: float = 1.2,
                    brightness: float = 1.0,
                    sharpen: bool = True) -> Image.Image:
    """
    Enhance grayscale image to better match manga line art style.
    
    Args:
        gray_img: PIL Grayscale Image
        contrast: Contrast multiplier (1.0 = no change, >1.0 = higher contrast)
        brightness: Brightness multiplier (1.0 = no change)
        sharpen: Whether to apply sharpening
    
    Returns:
        Enhanced PIL Grayscale Image
    """
    from PIL import ImageEnhance, ImageFilter
    
    # Adjust contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(gray_img)
        gray_img = enhancer.enhance(contrast)
    
    # Adjust brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(gray_img)
        gray_img = enhancer.enhance(brightness)
    
    # Sharpen to enhance line edges
    if sharpen:
        gray_img = gray_img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    return gray_img


def remove_color_artifacts(gray_img: Image.Image) -> Image.Image:
    """
    Remove any remaining color artifacts and ensure clean B/W output.
    
    Args:
        gray_img: PIL Grayscale Image
    
    Returns:
        Cleaned PIL Grayscale Image
    """
    gray_np = np.array(gray_img)
    
    # Apply slight Gaussian blur to smooth any artifacts
    blurred = cv2.GaussianBlur(gray_np, (3, 3), 0)
    
    # Enhance contrast to make lines clearer
    _, enhanced = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Blend original and enhanced (70% original, 30% enhanced)
    result = cv2.addWeighted(gray_np, 0.7, enhanced, 0.3, 0)
    
    return Image.fromarray(result, mode='L')


def process_image(input_path: Path, 
                 output_path: Path,
                 method: str = "weighted",
                 enhance: bool = True,
                 contrast: float = 1.2,
                 brightness: float = 1.0) -> bool:
    """
    Process a single image: convert to B/W and save.
    
    Args:
        input_path: Path to colored input image
        output_path: Path to save B/W output image
        method: Grayscale conversion method
        enhance: Whether to enhance line art
        contrast: Contrast enhancement factor
        brightness: Brightness enhancement factor
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load colored image
        color_img = Image.open(input_path).convert('RGB')
        
        # Convert to grayscale
        gray_img = convert_to_grayscale(color_img, method=method)
        
        # Enhance if requested
        if enhance:
            gray_img = enhance_line_art(gray_img, contrast=contrast, brightness=brightness)
            gray_img = remove_color_artifacts(gray_img)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save B/W image
        gray_img.save(output_path, quality=95)
        
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def create_paired_dataset(color_dir: str,
                         output_bw_dir: str,
                         method: str = "weighted",
                         enhance: bool = True,
                         contrast: float = 1.2,
                         brightness: float = 1.0,
                         extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')) -> None:
    """
    Create paired B/W dataset from colored manga images.
    
    Args:
        color_dir: Directory containing colored manga images
        output_bw_dir: Directory to save B/W images (will be created)
        method: Grayscale conversion method
        enhance: Whether to enhance line art
        contrast: Contrast enhancement factor
        brightness: Brightness enhancement factor
        extensions: Image file extensions to process
    """
    color_path = Path(color_dir)
    bw_path = Path(output_bw_dir)
    
    if not color_path.exists():
        raise ValueError(f"Color directory not found: {color_dir}")
    
    # Find all image files (recursively)
    image_files = []
    for ext in extensions:
        image_files.extend(color_path.rglob(f"*{ext}"))
        image_files.extend(color_path.rglob(f"*{ext.upper()}"))
    
    # Remove duplicates (Windows is case-insensitive, so .jpg and .JPG find same file)
    image_files = list(set(image_files))
    image_files.sort()  # Sort for consistent processing order
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {color_dir}")
    
    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: {bw_path}")
    print(f"Method: {method}, Enhance: {enhance}")
    print("-" * 60)
    
    # Process each image
    success_count = 0
    failed_count = 0
    failed_files = []  # Track failed files for cleanup
    
    for img_path in tqdm(image_files, desc="Converting to B/W"):
        # Calculate relative path to preserve directory structure
        relative_path = img_path.relative_to(color_path)
        output_img_path = bw_path / relative_path
        
        # Process and save
        if process_image(img_path, output_img_path, method, enhance, contrast, brightness):
            success_count += 1
        else:
            failed_count += 1
            failed_files.append(str(img_path))  # Save failed file path
    
    print("-" * 60)
    print(f"Processing complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"\nB/W images saved to: {bw_path}")
    
    # Save list of failed files for cleanup
    if failed_files:
        failed_list_path = bw_path.parent / "failed_files.txt"
        with open(failed_list_path, 'w', encoding='utf-8') as f:
            for failed_file in failed_files:
                f.write(f"{failed_file}\n")
        print(f"\n⚠️  {len(failed_files)} files failed to convert.")
        print(f"   Failed file list saved to: {failed_list_path}")
        print(f"   Run cleanup script to delete corrupted color images:")
        print(f"   python cleanup_failed_images.py")
    
    print(f"\nYour dataset structure should be:")
    print(f"  {color_dir}/     -> colored images")
    print(f"  {output_bw_dir}/ -> B/W images (paired)")


def main():
    parser = argparse.ArgumentParser(
        description="Create B/W versions from colored manga images for training"
    )
    parser.add_argument(
        "--color_dir",
        type=str,
        required=True,
        help="Directory containing colored manga images"
    )
    parser.add_argument(
        "--output_bw_dir",
        type=str,
        required=True,
        help="Output directory for B/W images"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="weighted",
        choices=["weighted", "luminance", "simple"],
        help="Grayscale conversion method (default: weighted)"
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable line art enhancement"
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.2,
        help="Contrast enhancement factor (default: 1.2)"
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=1.0,
        help="Brightness enhancement factor (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    create_paired_dataset(
        color_dir=args.color_dir,
        output_bw_dir=args.output_bw_dir,
        method=args.method,
        enhance=not args.no_enhance,
        contrast=args.contrast,
        brightness=args.brightness
    )


if __name__ == "__main__":
    main()


