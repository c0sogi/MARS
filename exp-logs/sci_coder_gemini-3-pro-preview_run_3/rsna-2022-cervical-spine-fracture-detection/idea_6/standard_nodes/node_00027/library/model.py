import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class ResNetBackbone(nn.Module):
    """
    ResNet-18 backbone for feature extraction.
    Initialized with ImageNet weights.
    The fully connected layer is replaced with Identity to return raw feature vectors.
    """

    def __init__(self):
        super(ResNetBackbone, self).__init__()
        # Load pretrained ResNet18
        # Using the weights API compatible with torchvision
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.net = models.resnet18(weights=weights)

        # Replace the fully connected layer with Identity to extract features
        # ResNet18 output before FC (after avgpool) is 512 dimensions
        self.net.fc = nn.Identity()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input batch of images (N, 3, H, W)
        Returns:
            torch.Tensor: Feature vectors (N, 512)
        """
        return self.net(x)


class SequenceContextResNet(nn.Module):
    """
    Sequence-Smoothed MIL Network.

    Architecture:
    1. Backbone: ResNet18 extracts features from each slice independently (2.5D).
    2. Context: Simple 1D Conv block refines features using z-axis context.
       Cite Lesson 00026: Avoid residual connections in context for small data.
       Cite Lesson 00020: Use 1D Conv for sequence smoothing.
    3. Instance Classifier: Predicts fracture probabilities for each slice independently.
    4. Aggregation: Global Max Pooling aggregates slice predictions to study level.
    """

    def __init__(self):
        super(SequenceContextResNet, self).__init__()

        # 1. Backbone
        self.backbone = ResNetBackbone()
        self.feature_dim = 512  # ResNet18 output dim

        # 2. Context Module
        # Simple linear convolution to mix features across depth.
        # No BatchNorm/ReLU/Residual to prevent overfitting to noise.
        self.context_conv = nn.Conv1d(
            self.feature_dim, self.feature_dim, kernel_size=3, padding=1
        )

        # 3. Instance Classifier
        self.classifier = nn.Linear(self.feature_dim, Config.N_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input volume (Batch, Slices, Channels, Height, Width)
                              Channels should be 3 (2.5D input).
        Returns:
            torch.Tensor: Study-level logits for C1-C7 (Batch, 7)
        """
        b, s, c, h, w = x.shape

        # --- Feature Extraction ---
        # Flatten batch and sequence dimensions to pass through 2D backbone
        x_flat = x.view(b * s, c, h, w)
        features = self.backbone(x_flat)  # (B*S, 512)

        # Reshape back to sequence format
        features = features.view(b, s, self.feature_dim)

        # --- Context Modeling ---
        # Conv1d expects (Batch, Channels, Length)
        features = features.permute(0, 2, 1)  # (B, C, S)
        features = self.context_conv(features)
        features = features.permute(0, 2, 1)  # (B, S, C)

        # --- Instance Classification ---
        # Apply classifier to every slice
        # Input: (B, S, 512) -> Output: (B, S, 7)
        slice_logits = self.classifier(features)

        # --- Aggregation ---
        # Global Max Pooling across the slice dimension (MIL)
        study_logits, _ = torch.max(slice_logits, dim=1)  # (B, 7)

        return study_logits
