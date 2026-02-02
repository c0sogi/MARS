import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentivePooling(nn.Module):
    """
    Attentive Pooling module to dynamically weight active speech segments.
    Applies attention over the time dimension.
    """

    def __init__(self, in_channels):
        super(AttentivePooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        # Input x shape: (Batch, Channels, Time)
        # Calculate attention weights
        w = self.attention(x)  # Shape: (Batch, 1, Time)

        # Apply weighted sum
        return torch.sum(x * w, dim=2)  # Shape: (Batch, Channels)


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout head.
    Applies multiple dropout masks to the same features and averages the predictions.
    Acts as an implicit ensemble within a single model.
    """

    def __init__(self, in_features, out_features, dropout_rate=0.5, num_samples=8):
        super(MultiSampleDropout, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(in_features, out_features)
        self.num_samples = num_samples

    def forward(self, x):
        # Input x shape: (Batch, Features)
        logits = []
        for _ in range(self.num_samples):
            logits.append(self.linear(self.dropout(x)))

        # Stack and average: (Num_Samples, Batch, Out) -> (Batch, Out)
        return torch.mean(torch.stack(logits), dim=0)


class DilatedEfficientNet(nn.Module):
    """
    Dilated EfficientNet-B2 with Attentive Pooling and Multi-Sample Dropout.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(DilatedEfficientNet, self).__init__()

        # Load EfficientNet-B2 with Noisy Student weights.
        # output_stride=16 ensures the final stage uses dilated convolutions (stride 1, dilation 2),
        # preserving a larger feature map for temporal resolution.
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, output_stride=16
        )

        # Adapt the first convolution layer for 1-channel input (Spectrogram).
        # We initialize by averaging the pretrained RGB weights.
        old_conv = self.backbone.conv_stem
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Average weights along the channel dimension (dim 1)
        # Shape: (Out, 3, K, K) -> (Out, 1, K, K)
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data

        self.backbone.conv_stem = new_conv

        # Retrieve the number of features from the backbone (usually 1408 for B2)
        self.num_features = self.backbone.num_features

        # Pooling Layer
        self.pool = AttentivePooling(self.num_features)

        # Classification Head
        self.head = MultiSampleDropout(
            in_features=self.num_features,
            out_features=num_classes,
            dropout_rate=Config.DROP_RATE,
            num_samples=Config.MULTI_SAMPLE_DROPOUT_COUNT,
        )

    def forward(self, x):
        # Input x: (Batch, 1, Freq, Time)

        # Extract features from backbone
        # Output shape: (Batch, Channels, Freq', Time')
        x = self.backbone.forward_features(x)

        # Global Average Pooling over the Frequency dimension
        # We want to preserve the Time dimension for Attentive Pooling.
        # (Batch, C, F', T') -> (Batch, C, T')
        x = x.mean(dim=2)

        # Apply Attentive Pooling over the Time dimension
        # (Batch, C, T') -> (Batch, C)
        x = self.pool(x)

        # Apply Multi-Sample Dropout Head
        # (Batch, C) -> (Batch, Num_Classes)
        x = self.head(x)

        return x
