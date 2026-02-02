import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling Layer.

    Instead of a single global average or max pooling, this layer learns multiple
    attention masks (heads). Each head focuses on different spatial/temporal
    regions of the spectrogram, allowing the model to capture disjoint features
    (e.g., the start and end of a word) simultaneously.
    """

    def __init__(self, in_features, num_heads):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads

        # Convolution to compute attention scores for each head independently.
        # Maps (Channels, H, W) -> (NumHeads, H, W)
        self.attn_conv = nn.Conv2d(in_features, num_heads, kernel_size=1, bias=True)

    def forward(self, x):
        """
        Args:
            x: Feature map of shape (Batch, Channels, Height, Width)
        Returns:
            Pooled features of shape (Batch, NumHeads * Channels)
        """
        b, c, h, w = x.shape

        # 1. Compute Attention Scores: (B, NumHeads, H, W)
        attn_logits = self.attn_conv(x)

        # 2. Flatten Spatial Dimensions
        # Reshape features to (B, C, N) where N = H*W, then transpose to (B, N, C)
        x_flat = x.view(b, c, -1).transpose(1, 2)

        # Reshape logits to (B, NumHeads, N)
        attn_flat = attn_logits.view(b, self.num_heads, -1)

        # 3. Apply Spatial Softmax
        # Normalize scores across the spatial dimension (dim=-1) to get valid probability distributions
        attn_weights = torch.softmax(attn_flat, dim=-1)

        # 4. Weighted Aggregation
        # Perform batched matrix multiplication: (B, NumHeads, N) x (B, N, C) -> (B, NumHeads, C)
        # This computes the weighted average of features for each head
        pooled = torch.bmm(attn_weights, x_flat)

        # 5. Flatten Heads
        # Concatenate the context vectors from all heads into a single vector
        # Output: (B, NumHeads * C)
        pooled = pooled.view(b, -1)

        return pooled


class EfficientNetV2Audio(nn.Module):
    """
    EfficientNet-V2-B0 adapted for Audio Classification.

    modifications:
    1. First layer modified to accept 1-channel input (weights summed from RGB).
    2. Global pooling and classifier replaced with Multi-Head Attention Pooling.
    """

    def __init__(self, cfg=Config):
        super().__init__()

        # 1. Load Backbone
        # num_classes=0 removes the default classifier and pooling
        # global_pool='' ensures we get the raw spatial feature maps
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=3,
        )

        # 2. Adapt First Layer for 1-Channel Input
        # We manually replace the first layer to ensure the specific initialization strategy
        # (summing RGB weights) is applied correctly.
        if hasattr(self.backbone, "conv_stem"):
            first_conv = self.backbone.conv_stem

            # Create new convolution with in_channels=1
            new_conv = nn.Conv2d(
                in_channels=cfg.in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

            # Initialize weights by summing RGB channels
            with torch.no_grad():
                new_conv.weight[:] = first_conv.weight.sum(dim=1, keepdim=True)
                if first_conv.bias is not None:
                    new_conv.bias[:] = first_conv.bias

            # Replace the layer
            self.backbone.conv_stem = new_conv

        # 3. Determine Feature Dimension
        # Run a dummy forward pass to dynamically determine output channels
        with torch.no_grad():
            # Create a dummy input with the configured shape
            dummy_input = torch.randn(1, cfg.in_channels, 128, 128)
            features = self.backbone(dummy_input)
            self.num_features = features.shape[1]

        # 4. Multi-Head Attention Pooling Head
        self.pooling = MultiHeadAttentionPooling(
            in_features=self.num_features, num_heads=cfg.attention_heads
        )

        # 5. Classifier
        # Input dimension is (NumFeatures * NumHeads)
        self.classifier = nn.Linear(
            self.num_features * cfg.attention_heads, cfg.num_classes
        )

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)

        # Backbone Feature Extraction
        # Output: (Batch, Channels, H, W)
        x = self.backbone(x)

        # Multi-Head Attention Pooling
        # Output: (Batch, Channels * NumHeads)
        x = self.pooling(x)

        # Classification
        # Output: (Batch, NumClasses)
        x = self.classifier(x)

        return x


def get_model():
    """
    Factory function to instantiate the model using the global configuration.
    """
    return EfficientNetV2Audio(Config)
