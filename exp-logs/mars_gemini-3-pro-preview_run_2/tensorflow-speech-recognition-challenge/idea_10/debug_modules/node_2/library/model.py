import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class SingleHeadAttentionPooling(nn.Module):
    """
    Single-Head Attention Pooling module.
    Computes a spatial attention map to pool features, effectively localizing
    the speech command within the spectrogram.
    """

    def __init__(self, in_channels, num_classes):
        super(SingleHeadAttentionPooling, self).__init__()
        # 1x1 Conv to compute a scalar attention score for each spatial location
        self.att_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        # Final classification linear layer
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        """
        Args:
            x: Feature map from backbone of shape (N, C, H, W)
        Returns:
            logits: Class logits of shape (N, num_classes)
        """
        N, C, H, W = x.size()

        # 1. Compute Attention Map
        # Project features to attention scores: (N, 1, H, W)
        attn_logits = self.att_conv(x)

        # Flatten spatial dimensions: (N, 1, H*W)
        attn_logits = attn_logits.view(N, 1, -1)

        # Normalize scores across spatial locations
        attn_weights = self.softmax(attn_logits)

        # 2. Apply Weighted Pooling
        # Flatten input features: (N, C, H*W)
        x_flat = x.view(N, C, -1)

        # Weighted sum: (N, C, H*W) @ (N, H*W, 1) -> (N, C, 1)
        # We permute attn_weights to (N, H*W, 1) for batched matrix multiplication
        pooled = torch.bmm(x_flat, attn_weights.permute(0, 2, 1))

        # Remove the last dimension: (N, C)
        pooled = pooled.view(N, C)

        # 3. Classification
        logits = self.linear(pooled)

        return logits


class EfficientNetV2Audio(nn.Module):
    """
    EfficientNetV2-S adapted for 1-channel Audio Spectrograms.
    Uses Single-Head Attention Pooling for classification.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(EfficientNetV2Audio, self).__init__()

        # 1. Load Pretrained Backbone
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_v2_s(weights=weights)

        # 2. Modify First Convolutional Layer
        # Original layer expects 3 channels (RGB). We need 1 channel (Spectrogram).
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        first_conv_layer = self.backbone.features[0][0]

        new_first_layer = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv_layer.out_channels,
            kernel_size=first_conv_layer.kernel_size,
            stride=first_conv_layer.stride,
            padding=first_conv_layer.padding,
            bias=first_conv_layer.bias is not None,
        )

        # Initialize weights by summing original RGB weights.
        # This preserves the magnitude of activations.
        if pretrained:
            with torch.no_grad():
                # Sum across the input channel dimension (dim 1)
                # Original weight shape: (Out, 3, K, K) -> New: (Out, 1, K, K)
                new_first_layer.weight.data = first_conv_layer.weight.data.sum(
                    dim=1, keepdim=True
                )

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_first_layer

        # 3. Determine Output Channels
        # For EfficientNetV2-B0, the classifier input features (stage 6) is 1280.
        # We can retrieve this dynamically from the original classifier.
        out_channels = self.backbone.classifier[1].in_features

        # 4. Remove Original Head
        # We only use the feature extractor part
        del self.backbone.classifier
        del self.backbone.avgpool

        # 5. Attach Custom Attention Head
        self.head = SingleHeadAttentionPooling(out_channels, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input spectrogram of shape (N, 1, Freq, Time)
        """
        # Extract features using the backbone
        # Output shape: (N, 1280, H', W')
        features = self.backbone.features(x)

        # Apply attention pooling and classification
        logits = self.head(features)

        return logits
