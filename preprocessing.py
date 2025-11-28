"""
Shared preprocessing utilities for training and inference transforms.
"""

from torchvision import transforms


SKETCH_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


COLOR_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)