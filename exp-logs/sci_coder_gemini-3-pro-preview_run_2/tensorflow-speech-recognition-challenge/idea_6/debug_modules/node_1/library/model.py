import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from library import config


class AttentionPooling(nn.Module):
    """
    Attention Pooling module to dynamically weight spatial/temporal features.
    Replaces standard Global Average Pooling.
    """

    def __init__(self, in_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        B, C, H, W = x.size()

        # Flatten spatial dimensions: (B, C, H*W) -> (B, H*W, C)
        # We treat H*W as the sequence length (time/frequency steps flattened)
        x = x.view(B, C, -1).permute(0, 2, 1)

        # Calculate attention weights: (B, H*W, 1)
        weights = self.attention(x)

        # Weighted sum: (B, C)
        # Transpose x back to (B, C, H*W) for matrix multiplication with weights (B, H*W, 1)
        # Result is (B, C, 1) -> squeeze to (B, C)
        context = torch.bmm(x.transpose(1, 2), weights).squeeze(2)

        return context


class ConvNeXtSpeech(nn.Module):
    """
    ConvNeXt-Tiny adapted for Speech Command Recognition.
    - 1-channel input modification (summed weights).
    - Attention Pooling.
    - Linear Classifier.
    """

    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True):
        super(ConvNeXtSpeech, self).__init__()

        # 1. Load Pretrained Backbone
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = convnext_tiny(weights=weights)

        # 2. Modify First Layer for 1-Channel Input (Spectrogram)
        # The first layer in ConvNeXt is part of 'features[0]', which is a Conv2dNormActivation
        # features[0][0] is the Conv2d layer.
        original_layer = self.backbone.features[0][0]

        new_layer = nn.Conv2d(
            in_channels=1,
            out_channels=original_layer.out_channels,
            kernel_size=original_layer.kernel_size,
            stride=original_layer.stride,
            padding=original_layer.padding,
            bias=(original_layer.bias is not None),
        )

        # Initialize weights by summing RGB channels
        # original weight shape: (out_channels, 3, k, k)
        # new weight shape: (out_channels, 1, k, k)
        with torch.no_grad():
            new_layer.weight.data = original_layer.weight.data.sum(dim=1, keepdim=True)
            if original_layer.bias is not None:
                new_layer.bias.data = original_layer.bias.data

        self.backbone.features[0][0] = new_layer

        # 3. Remove original classifier and pooling
        # ConvNeXt classifier is a Sequential(LayerNorm2d, Flatten, Linear)
        # We will use the features output directly.
        # The last block of features outputs 768 channels for Tiny.
        self.feature_dim = 768

        # 4. Attention Pooling Head
        self.pool = AttentionPooling(self.feature_dim)

        # 5. Classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch, 1, Freq, Time)

        # Extract features
        # Output shape: (Batch, 768, H', W')
        x = self.backbone.features(x)

        # Apply Attention Pooling
        # Output shape: (Batch, 768)
        x = self.pool(x)

        # Classification
        # Output shape: (Batch, num_classes)
        x = self.classifier(x)

        return x
