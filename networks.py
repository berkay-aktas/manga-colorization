"""
Pix2Pix Networks for Manga Colorization

This module contains the Generator (U-Net) and Discriminator (PatchGAN) architectures
for conditional GAN-based manga colorization.
"""

import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    """
    U-Net based Generator for Pix2Pix.
    
    Architecture:
    - Encoder: Downsampling blocks that extract features
    - Decoder: Upsampling blocks with skip connections to preserve details
    - Skip connections between encoder and decoder at corresponding levels
    
    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        out_channels (int): Number of output channels (default: 3 for RGB)
        ngf (int): Number of generator filters in the first conv layer (default: 64)
    """
    
    def __init__(self, in_channels=1, out_channels=3, ngf=64):
        super(UNetGenerator, self).__init__()
        
        # Encoder (downsampling path)
        # Each block: Conv2d -> BatchNorm -> LeakyReLU
        self.encoder1 = self._encoder_block(in_channels, ngf, use_norm=False)  # 256x256 -> 128x128
        self.encoder2 = self._encoder_block(ngf, ngf * 2)  # 128x128 -> 64x64
        self.encoder3 = self._encoder_block(ngf * 2, ngf * 4)  # 64x64 -> 32x32
        self.encoder4 = self._encoder_block(ngf * 4, ngf * 8)  # 32x32 -> 16x16
        self.encoder5 = self._encoder_block(ngf * 8, ngf * 8)  # 16x16 -> 8x8
        self.encoder6 = self._encoder_block(ngf * 8, ngf * 8)  # 8x8 -> 4x4
        self.encoder7 = self._encoder_block(ngf * 8, ngf * 8)  # 4x4 -> 2x2
        self.encoder8 = self._encoder_block(ngf * 8, ngf * 8, use_norm=False)  # 2x2 -> 1x1
        
        # Decoder (upsampling path with skip connections)
        # Each block: ConvTranspose2d -> BatchNorm -> Dropout (optional) -> ReLU
        # Input channels = previous decoder channels + skip connection channels
        self.decoder1 = self._decoder_block(ngf * 8, ngf * 8, use_dropout=True)  # 1x1 -> 2x2
        self.decoder2 = self._decoder_block(ngf * 8 * 2, ngf * 8, use_dropout=True)  # 2x2 -> 4x4 (skip from encoder7)
        self.decoder3 = self._decoder_block(ngf * 8 * 2, ngf * 8, use_dropout=True)  # 4x4 -> 8x8 (skip from encoder6)
        self.decoder4 = self._decoder_block(ngf * 8 * 2, ngf * 8)  # 8x8 -> 16x16 (skip from encoder5)
        self.decoder5 = self._decoder_block(ngf * 8 * 2, ngf * 4)  # 16x16 -> 32x32 (skip from encoder4)
        self.decoder6 = self._decoder_block(ngf * 4 * 2, ngf * 2)  # 32x32 -> 64x64 (skip from encoder3)
        self.decoder7 = self._decoder_block(ngf * 2 * 2, ngf)  # 64x64 -> 128x128 (skip from encoder2)
        self.decoder8 = self._decoder_block(ngf * 2, out_channels, use_norm=False, final=True)  # 128x128 -> 256x256 (skip from encoder1)
    
    def _encoder_block(self, in_channels, out_channels, use_norm=True):
        """Create an encoder block: Conv2d -> BatchNorm -> LeakyReLU"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _decoder_block(self, in_channels, out_channels, use_norm=True, use_dropout=False, final=False):
        """Create a decoder block: ConvTranspose2d -> BatchNorm -> Dropout -> ReLU (or Tanh if final)"""
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        if final:
            layers.append(nn.Tanh())  # Output range [-1, 1]
        else:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through U-Net with skip connections.
        
        Args:
            x: Input tensor of shape [B, in_channels, 256, 256]
        
        Returns:
            Output tensor of shape [B, out_channels, 256, 256] with values in [-1, 1]
        """
        # Encoder path (store outputs for skip connections)
        e1 = self.encoder1(x)  # [B, ngf, 128, 128]
        e2 = self.encoder2(e1)  # [B, ngf*2, 64, 64]
        e3 = self.encoder3(e2)  # [B, ngf*4, 32, 32]
        e4 = self.encoder4(e3)  # [B, ngf*8, 16, 16]
        e5 = self.encoder5(e4)  # [B, ngf*8, 8, 8]
        e6 = self.encoder6(e5)  # [B, ngf*8, 4, 4]
        e7 = self.encoder7(e6)  # [B, ngf*8, 2, 2]
        e8 = self.encoder8(e7)  # [B, ngf*8, 1, 1]
        
        # Decoder path with skip connections
        d1 = self.decoder1(e8)  # [B, ngf*8, 2, 2]
        d2 = self.decoder2(torch.cat([d1, e7], dim=1))  # [B, ngf*8, 4, 4]
        d3 = self.decoder3(torch.cat([d2, e6], dim=1))  # [B, ngf*8, 8, 8]
        d4 = self.decoder4(torch.cat([d3, e5], dim=1))  # [B, ngf*8, 16, 16]
        d5 = self.decoder5(torch.cat([d4, e4], dim=1))  # [B, ngf*4, 32, 32]
        d6 = self.decoder6(torch.cat([d5, e3], dim=1))  # [B, ngf*2, 64, 64]
        d7 = self.decoder7(torch.cat([d6, e2], dim=1))  # [B, ngf, 128, 128]
        d8 = self.decoder8(torch.cat([d7, e1], dim=1))  # [B, out_channels, 256, 256]
        
        return d8


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix.
    
    This discriminator classifies N x N patches (e.g., 70x70) as real or fake,
    rather than the entire image. This allows it to focus on local texture and
    structure, which is beneficial for preserving fine details in manga.
    
    Architecture:
    - Series of Conv2d layers with LeakyReLU and BatchNorm
    - Outputs a feature map where each element represents a patch classification
    
    Args:
        in_channels (int): Number of input channels (default: 4 = 1 grayscale + 3 RGB)
        ndf (int): Number of discriminator filters in the first conv layer (default: 64)
    """
    
    def __init__(self, in_channels=4, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        
        # PatchGAN architecture: 70x70 patches
        # Input: [B, in_channels, 256, 256]
        # Output: [B, 1, 30, 30] (each element classifies a 70x70 patch)
        
        self.model = nn.Sequential(
            # Layer 1: No normalization
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # 256x256 -> 128x128
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),  # 32x32 -> 31x31
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5: Output layer (no normalization, no activation - raw logits)
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1, bias=False),  # 31x31 -> 30x30
        )
    
    def forward(self, x):
        """
        Forward pass through PatchGAN discriminator.
        
        Args:
            x: Concatenated input tensor of shape [B, in_channels, 256, 256]
               where in_channels = grayscale_channels + RGB_channels (typically 1 + 3 = 4)
        
        Returns:
            Feature map of shape [B, 1, H', W'] with real/fake logits per patch
            For 256x256 input, output is [B, 1, 30, 30]
        """
        return self.model(x)

