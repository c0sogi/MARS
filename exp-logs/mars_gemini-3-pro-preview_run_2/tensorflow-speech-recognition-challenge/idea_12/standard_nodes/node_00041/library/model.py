import torch
import torch.nn as nn
import timm
from library.config import Config


class SingleHeadAttentionPooling(nn.Module):
    """
    Single-Head 2D Attention Pooling.
    Applies a learned weighting mask over the 2D (Time-Frequency) feature map.
    Acts as a Voice Activity Detector, localizing the speech command and
    suppressing background noise before aggregation.
    """

    def __init__(self, in_channels):
        super(SingleHeadAttentionPooling, self).__init__()
        # 1x1 Conv to generate attention scores from features
        self.attn_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Channels, Freq, Time)

        Returns:
            torch.Tensor: Pooled features of shape (Batch, Channels)
        """
        B, C, H, W = x.shape

        # 1. Generate Attention Scores
        # Shape: (B, 1, H, W)
        scores = self.attn_conv(x)

        # 2. Flatten spatial dimensions
        # Shape: (B, 1, H*W)
        scores = scores.view(B, 1, -1)

        # 3. Apply Softmax to get probability mask (sums to 1 over spatial dims)
        attn_weights = torch.softmax(scores, dim=-1)

        # 4. Weighted Aggregation
        # Flatten input features: (B, C, H*W)
        x_flat = x.view(B, C, -1)

        # Perform weighted sum: (B, C, H*W) @ (B, H*W, 1) -> (B, C, 1)
        # We permute attn_weights to (B, H*W, 1) for matrix multiplication
        out = torch.bmm(x_flat, attn_weights.permute(0, 2, 1))

        # Remove last dimension -> (B, C)
        return out.squeeze(-1)


class EfficientNetV2Audio(nn.Module):
    """
    EfficientNetV2-B0 adapted for 1-channel Audio Spectrograms.
    Features:
    - Fused-MBConv backbone (via timm).
    - 1-channel input adaptation (Summed weights).
    - Single-Head Attention Pooling.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(EfficientNetV2Audio, self).__init__()

        # 1. Load Pretrained Backbone
        # We use 'tf_efficientnetv2_b0' which corresponds to the official V2-B0 architecture.
        # num_classes=0 and global_pool='' ensures we get the raw feature maps.
        self.backbone = timm.create_model(
            "tf_efficientnetv2_b0",
            pretrained=True,
            num_classes=0,
            global_pool="",
        )

        # 2. Adapt First Layer for 1-Channel Input
        # The first layer in EfficientNet is 'conv_stem'.
        old_stem = self.backbone.conv_stem

        # Create new 1-channel convolution with same parameters
        new_stem = nn.Conv2d(
            in_channels=1,
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=old_stem.bias is not None,
        )

        # Initialize weights by summing original RGB weights
        # old_stem.weight shape: (Out, 3, K, K)
        # new_stem.weight shape: (Out, 1, K, K)
        with torch.no_grad():
            new_stem.weight.copy_(old_stem.weight.sum(dim=1, keepdim=True))
            if old_stem.bias is not None:
                new_stem.bias.copy_(old_stem.bias)

        self.backbone.conv_stem = new_stem

        # 3. Determine Feature Dimension
        # Run a dummy forward pass to get the output channels of the backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 128, 128)
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # 4. Define Head
        self.pool = SingleHeadAttentionPooling(in_features)
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms (Batch, 1, Freq, Time)
        """
        # Extract features
        x = self.backbone(x)  # (B, C, H, W)

        # Attention Pooling
        x = self.pool(x)  # (B, C)

        # Classification Head
        x = self.dropout(x)
        x = self.fc(x)

        return x
