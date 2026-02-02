import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling module to dynamically weight relevant time-frequency regions.
    Replaces global average pooling.
    """

    def __init__(self, in_channels):
        super(AttentionPooling, self).__init__()
        # 1x1 Conv to project features to a single attention score per spatial location
        self.att_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Feature map of shape (Batch, Channels, Height, Width)
        Returns:
            Pooled feature vector of shape (Batch, Channels)
        """
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Compute attention scores: (B, 1, H, W)
        scores = self.att_conv(x)

        # Flatten spatial dimensions: (B, 1, H*W)
        scores = scores.view(B, 1, -1)

        # Apply Softmax to get attention weights summing to 1
        weights = torch.softmax(scores, dim=-1)

        # Flatten input features: (B, C, H*W)
        x_flat = x.view(B, C, -1)

        # Compute weighted sum: (B, C, H*W) @ (B, H*W, 1) -> (B, C, 1)
        # Permute weights to (B, H*W, 1) for matrix multiplication
        out = torch.bmm(x_flat, weights.permute(0, 2, 1))

        # Remove last dimension: (B, C)
        out = out.squeeze(-1)

        return out


class AudioClassifier(nn.Module):
    """
    Audio Classification model using a timm backbone with Attention Pooling.
    Supports EfficientNet and ConvNeXt architectures.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        super(AudioClassifier, self).__init__()

        # 1. Load Pretrained Backbone
        # num_classes=0 and global_pool='' ensures we get the feature map output
        # We load with in_chans=3 initially to get the pretrained weights correctly,
        # then modify the first layer manually.
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=3,
        )

        # 2. Modify First Layer for 1-Channel Input
        self._modify_first_layer()

        # Get the number of output features from the backbone
        self.num_features = self.backbone.num_features

        # 3. Pooling Mechanism
        if Config.USE_ATTENTION_POOLING:
            self.pool = AttentionPooling(self.num_features)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        # 4. Classification Head
        self.fc = nn.Linear(self.num_features, num_classes)

    def _modify_first_layer(self):
        """
        Replaces the first convolutional layer to accept 1-channel input.
        Weights are initialized by averaging the original RGB weights.
        Handles both EfficientNet (conv_stem) and ConvNeXt (stem[0]) structures.
        """
        # Identify the first layer
        if hasattr(self.backbone, "conv_stem"):
            # EfficientNet family
            layer_name = "conv_stem"
            old_conv = self.backbone.conv_stem
        elif hasattr(self.backbone, "stem"):
            # ConvNeXt family
            layer_name = "stem"
            old_conv = self.backbone.stem[0]
        else:
            # Fallback or error
            raise AttributeError(
                "Could not find first convolutional layer (stem/conv_stem) in backbone."
            )

        if not isinstance(old_conv, nn.Conv2d):
            # In some implementations, stem might be a Sequential
            if isinstance(old_conv, nn.Sequential):
                old_conv = old_conv[0]
            else:
                raise ValueError(f"Expected Conv2d, got {type(old_conv)}")

        # Create new Conv2d layer with in_channels=1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize weights by averaging the original 3 channels
        with torch.no_grad():
            # old_conv.weight shape: (Out, 3, K, K)
            # mean(dim=1) shape: (Out, K, K) -> unsqueeze -> (Out, 1, K, K)
            new_conv.weight.copy_(torch.mean(old_conv.weight, dim=1, keepdim=True))

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        # Replace the layer in the backbone
        if layer_name == "conv_stem":
            self.backbone.conv_stem = new_conv
        elif layer_name == "stem":
            self.backbone.stem[0] = new_conv

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x: Input tensor of shape (Batch, 1, Freq, Time) or (Batch, Freq, Time)
        """
        # Ensure input has channel dimension: (B, 1, F, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Backbone Feature Extraction
        # Output shape: (B, C, H, W) where H, W are reduced frequency/time dimensions
        x = self.backbone(x)

        # Pooling
        if isinstance(self.pool, AttentionPooling):
            x = self.pool(x)  # (B, C)
        else:
            x = self.pool(x)  # (B, C, 1, 1)
            x = x.flatten(1)  # (B, C)

        # Classification
        x = self.fc(x)

        return x
