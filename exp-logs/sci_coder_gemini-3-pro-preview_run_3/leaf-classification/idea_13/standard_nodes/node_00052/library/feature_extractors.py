import torch
import torch.nn as nn
import timm
import numpy as np
from library.config import Config


class DualStreamExtractor:
    """
    A dual-stream feature extractor combining Global Geometry (DINOv2) and
    Local Texture (ConvNeXt) representations.
    """

    def __init__(self):
        """
        Initializes the DINOv2 and ConvNeXt models using timm, sets them to
        evaluation mode, and moves them to the appropriate device (GPU/CPU).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Global Geometry Stream: DINOv2 (ViT-Large)
        # num_classes=0 removes the head, returning the pooled features/class token
        self.dino_model = timm.create_model(
            Config.MODEL_DINOV2,
            pretrained=True,
            num_classes=0,
            img_size=Config.IMG_SIZE,
        )
        self.dino_model.to(self.device)
        self.dino_model.eval()

        # Initialize Local Texture Stream: ConvNeXt Large
        self.convnext_model = timm.create_model(
            Config.MODEL_CONVNEXT, pretrained=True, num_classes=0
        )
        self.convnext_model.to(self.device)
        self.convnext_model.eval()

    def extract_batch(self, batch_tensors: torch.Tensor) -> np.ndarray:
        """
        Extracts and concatenates features from both backbones for a batch of images.

        Args:
            batch_tensors (torch.Tensor): Input batch of images with shape
                                          (B, C, H, W).

        Returns:
            np.ndarray: Combined feature vectors with shape
                        (B, DINO_dim + ConvNeXt_dim).
        """
        # Ensure input is on the correct device
        batch_tensors = batch_tensors.to(self.device)

        with torch.no_grad():
            # Extract Global Geometry features
            dino_features = self.dino_model(batch_tensors)

            # Extract Local Texture features
            convnext_features = self.convnext_model(batch_tensors)

            # Concatenate features along the feature dimension (dim=1)
            combined_features = torch.cat([dino_features, convnext_features], dim=1)

        # Move to CPU and convert to numpy
        return combined_features.cpu().numpy()
