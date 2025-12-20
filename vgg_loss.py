"""
VGG Perceptual Loss for better colorization quality and generalization.

Uses pretrained VGG19 to extract features and compute perceptual loss
that focuses on high-level features rather than pixel-level differences.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


class VGGPerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss for image generation.
    
    Extracts features from multiple layers of pretrained VGG19 and
    computes L1 loss between real and generated features.
    """
    
    def __init__(self, feature_layers: List[int] = [0, 5, 10, 19, 28], weights: List[float] = None):
        """
        Args:
            feature_layers: List of layer indices to extract features from
            weights: Weights for each layer (default: [0.1, 0.1, 0.1, 0.1, 0.1])
        """
        super(VGGPerceptualLoss, self).__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features
        vgg.eval()
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg
        self.feature_layers = feature_layers
        self.weights = weights if weights else [0.1] * len(feature_layers)
        
        # Normalize input to match ImageNet preprocessing
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input to ImageNet statistics.
        
        Args:
            x: Input tensor in range [-1, 1] (from Tanh output)
        
        Returns:
            Normalized tensor in range [0, 1] for VGG
        """
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        # Normalize with ImageNet stats
        x = (x - self.mean) / self.std
        return x
    
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from multiple VGG layers.
        
        Args:
            x: Input tensor [B, 3, H, W] in range [-1, 1]
        
        Returns:
            List of feature tensors from specified layers
        """
        x = self.normalize(x)
        features = []
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between predicted and target images.
        
        Args:
            pred: Predicted image [B, 3, H, W] in range [-1, 1]
            target: Target image [B, 3, H, W] in range [-1, 1]
        
        Returns:
            Perceptual loss value
        """
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0.0
        for pred_feat, target_feat, weight in zip(pred_features, target_features, self.weights):
            # Normalize by feature map size for stability
            feat_loss = torch.nn.functional.l1_loss(pred_feat, target_feat)
            # Scale by number of elements in feature map
            loss += weight * feat_loss
        
        return loss

