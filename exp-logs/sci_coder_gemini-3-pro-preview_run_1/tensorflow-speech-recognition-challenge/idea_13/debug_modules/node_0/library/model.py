import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import NUM_CLASSES, MODEL_NAME


class AttentivePooling(nn.Module):
    """
    Attentive Pooling mechanism to dynamically weight active speech segments
    and suppress background noise.

    Input:  (Batch, Channels, Freq, Time)
    Output: (Batch, Channels)
    """

    def __init__(self, in_channels):
        super(AttentivePooling, self).__init__()
        # Attention mechanism: Learn a weight for each time step
        # Structure: 1x1 Conv -> Tanh -> 1x1 Conv -> Softmax
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(in_channels // 2, 1, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Freq, Time)

        # 1. Pool over Frequency dimension to get time-series features
        # Result: (Batch, Channels, Time)
        x_time = torch.mean(x, dim=2)

        # 2. Calculate attention weights
        # w shape: (Batch, 1, Time)
        w = self.attention(x_time)

        # 3. Apply weights (Weighted Sum)
        # x_time * w shape: (Batch, Channels, Time)
        # Sum over Time dimension -> (Batch, Channels)
        out = torch.sum(x_time * w, dim=2)

        return out


class DilatedEfficientNet(nn.Module):
    """
    Dilated EfficientNet-B2 with Attentive Pooling.

    - Backbone: EfficientNet-B2
    - Input: 1-Channel Spectrogram (Weights averaged from RGB)
    - Dilation: Stride=1, Dilation=2 in final stage (via output_stride=16)
    - Head: Attentive Pooling + Linear Classifier
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(DilatedEfficientNet, self).__init__()

        # Load EfficientNet-B2
        # output_stride=16 ensures the last stage uses dilation=2 and stride=1,
        # preserving spatial/temporal resolution.
        # global_pool='' ensures we get the feature map, not a vector.
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            in_chans=3,  # Load RGB weights first, then modify
            num_classes=0,
            global_pool="",
            output_stride=16,
        )

        # Modify first layer for 1-channel input by averaging weights
        self._modify_first_layer()

        # Determine feature dimension
        self.num_features = self.backbone.num_features

        # Attentive Pooling Head
        self.pool = AttentivePooling(self.num_features)

        # Classifier for Fine-Grained Labels
        self.fc = nn.Linear(self.num_features, num_classes)

    def _modify_first_layer(self):
        """
        Replaces the first convolutional layer (conv_stem) with a 1-channel version.
        Weights are initialized by averaging the original RGB weights.
        """
        old_conv = self.backbone.conv_stem

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize weights
        # old_conv.weight shape: (Out, 3, K, K)
        # new_conv.weight shape: (Out, 1, K, K)
        with torch.no_grad():
            new_conv.weight.copy_(torch.mean(old_conv.weight, dim=1, keepdim=True))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_conv

    def forward(self, x):
        # Input x: (Batch, 1, Freq, Time)

        # Extract features
        # Output shape: (Batch, Channels, Freq', Time')
        # With output_stride=16, resolution is reduced by factor of 16
        features = self.backbone(x)

        # Apply Attentive Pooling
        # Output shape: (Batch, Channels)
        embedding = self.pool(features)

        # Classification
        # Output shape: (Batch, Num_Classes)
        logits = self.fc(embedding)

        return logits


def get_model(num_classes=NUM_CLASSES):
    """
    Factory function to instantiate the model.
    """
    return DilatedEfficientNet(num_classes=num_classes)
