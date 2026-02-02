import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import NUM_CLASSES


class AttentionPooling2D(nn.Module):
    """
    Applies 2D Attention Pooling to a feature map.
    Computes a spatial attention map and performs a weighted sum of features.
    """

    def __init__(self, in_channels):
        super().__init__()
        # Project channels to 1 to get a raw attention score map
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Feature map of shape (Batch, Channels, Freq, Time)
        Returns:
            pooled: Context vector of shape (Batch, Channels)
        """
        b, c, h, w = x.size()

        # 1. Compute attention scores
        # Shape: (B, 1, H, W)
        attn_logits = self.attn_conv(x)

        # 2. Normalize scores across spatial dimensions (H * W)
        # Shape: (B, 1, H*W)
        attn_logits = attn_logits.view(b, 1, h * w)
        attn_weights = F.softmax(attn_logits, dim=-1)

        # 3. Weighted Aggregation
        # Reshape input to (B, C, H*W)
        x_flat = x.view(b, c, h * w)

        # Perform weighted sum: (B, C, H*W) @ (B, H*W, 1) -> (B, C, 1)
        # Transpose weights to (B, H*W, 1) for batched matrix multiplication
        pooled = torch.bmm(x_flat, attn_weights.transpose(1, 2))

        # Remove last dimension -> (B, C)
        return pooled.squeeze(-1)


class EfficientNetAttn(nn.Module):
    """
    Standard EfficientNet-B0 with Attention Pooling.
    Simpler architecture than Multi-Scale, often sufficient for global classification.
    (Cite solution_lesson_node_00033)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # 1. Load Pretrained Backbone
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights)

        # 2. Adapt First Layer for 1-Channel Input
        first_conv = self.backbone.features[0][0]

        # Sum weights across the channel dimension (3 -> 1)
        with torch.no_grad():
            new_weight = first_conv.weight.sum(dim=1, keepdim=True)

        # Re-initialize the layer
        first_conv.in_channels = 1
        first_conv.weight = nn.Parameter(new_weight)

        # 3. Attention Pooling
        # EfficientNet-B0 final feature map has 1280 channels
        self.pool = AttentionPooling2D(in_channels=1280)

        # 4. Classifier
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input spectrogram of shape (Batch, 1, Freq, Time)
        """
        # Extract features (B, 1280, F/32, T/32)
        x = self.backbone.features(x)

        # Pool (B, 1280)
        x = self.pool(x)

        # Classify
        out = self.classifier(x)

        return out
