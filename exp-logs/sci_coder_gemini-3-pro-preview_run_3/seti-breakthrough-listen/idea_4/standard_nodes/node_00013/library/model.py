import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class SiameseEfficientNet(nn.Module):
    """
    Siamese Network architecture using EfficientNet-B0 backbone.

    This model processes two input streams (On-Target and Off-Target spectrograms)
    sharing the same weights. It explicitly computes the difference between the
    feature representations of the two streams to learn the 'subtraction' logic
    required for identifying technosignatures while rejecting RFI.
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): If True, loads ImageNet weights for the backbone.
        """
        super(SiameseEfficientNet, self).__init__()

        # Load EfficientNet-B0 backbone
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Extract the feature dimension
        # EfficientNet-B0 classifier's input features is the output of the avgpool/flatten
        # Typically 1280 for B0
        self.feature_dim = self.backbone.classifier[1].in_features

        # We only need the feature extractor and pooling parts
        # self.backbone.features handles the convolutions
        # self.backbone.avgpool handles the spatial pooling

        # Define the custom classifier head
        # Input: Concatenation of (v_on, v_off, v_on - v_off) -> 3 * feature_dim
        self.classifier = nn.Linear(self.feature_dim * 3, Config.NUM_CLASSES)

    def forward_features(self, x):
        """
        Extracts flattened features from a single image tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Feature vector of shape (B, feature_dim).
        """
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, on_input, off_input):
        """
        Forward pass of the Siamese Network.

        Args:
            on_input (torch.Tensor): Batch of On-Target images (B, 3, H, W).
            off_input (torch.Tensor): Batch of Off-Target images (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Extract features for both streams using shared weights
        v_on = self.forward_features(on_input)
        v_off = self.forward_features(off_input)

        # Explicitly compute the difference in feature space
        # This helps the model distinguish between signals present in both (RFI)
        # and signals present only in 'on' (Technosignatures).
        v_diff = v_on - v_off

        # Fuse the representations
        # We include v_on and v_off to retain context about absolute intensity/features
        # and v_diff to emphasize the contrast.
        v_fused = torch.cat([v_on, v_off, v_diff], dim=1)

        # Classification
        logits = self.classifier(v_fused)

        return logits
