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


class MultiScaleEfficientNet(nn.Module):
    """
    EfficientNet-B0 modified for Multi-Scale Feature Aggregation.
    Extracts features from 8x, 16x, and 32x downsampling stages,
    applies attention pooling to each, and concatenates them for classification.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # 1. Load Pretrained Backbone
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)

        # 2. Adapt First Layer for 1-Channel Input
        # EfficientNet's first layer is features[0][0] (Conv2d)
        first_conv = backbone.features[0][0]

        # Sum weights across the channel dimension (3 -> 1)
        with torch.no_grad():
            new_weight = first_conv.weight.sum(dim=1, keepdim=True)

        # Re-initialize the layer with new configuration
        first_conv.in_channels = 1
        first_conv.weight = nn.Parameter(new_weight)

        # 3. Slice Backbone into Hierarchical Stages
        # Based on EfficientNet-B0 architecture:
        # features[0-3]: Downsamples to 8x. Output channels: 40
        # features[4-5]: Downsamples to 16x. Output channels: 112
        # features[6-8]: Downsamples to 32x. Output channels: 1280

        # Note: We use *list(...) to unpack into Sequential to avoid keeping reference to full backbone
        features_list = list(backbone.features.children())

        self.stage1_extractor = nn.Sequential(*features_list[0:4])
        self.stage2_extractor = nn.Sequential(*features_list[4:6])
        self.stage3_extractor = nn.Sequential(*features_list[6:])

        # 4. Attention Heads
        self.pool1 = AttentionPooling2D(in_channels=40)
        self.pool2 = AttentionPooling2D(in_channels=112)
        self.pool3 = AttentionPooling2D(in_channels=1280)

        # 5. Classifier
        # Input dimension is sum of all stage output channels
        concat_dim = 40 + 112 + 1280
        self.classifier = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input spectrogram of shape (Batch, 1, Freq, Time)
        """
        # --- Stage 1 (High Res) ---
        # Output: (B, 40, F/8, T/8)
        x1 = self.stage1_extractor(x)
        v1 = self.pool1(x1)  # -> (B, 40)

        # --- Stage 2 (Mid Res) ---
        # Input is x1. Output: (B, 112, F/16, T/16)
        x2 = self.stage2_extractor(x1)
        v2 = self.pool2(x2)  # -> (B, 112)

        # --- Stage 3 (Low Res) ---
        # Input is x2. Output: (B, 1280, F/32, T/32)
        x3 = self.stage3_extractor(x2)
        v3 = self.pool3(x3)  # -> (B, 1280)

        # --- Fusion & Classification ---
        # Concatenate multi-scale context vectors
        v_concat = torch.cat([v1, v2, v3], dim=1)  # -> (B, 1432)

        # Predict
        out = self.classifier(v_concat)

        return out
