import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling layer to weight time frames dynamically.
    Formula: w = softmax(W2 * tanh(W1 * x))
    Output: sum(x * w)
    """

    def __init__(self, input_dim, hidden_dim=None):
        super(AttentionPooling, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.w2 = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Input x shape: (Batch, Time, Channels)

        # Calculate attention scores
        # (B, T, C) -> (B, T, H)
        u = self.tanh(self.w1(x))
        # (B, T, H) -> (B, T, 1)
        scores = self.w2(u)

        # Normalize scores across time dimension
        weights = self.softmax(scores)

        # Weighted sum
        # (B, T, C) * (B, T, 1) -> (B, T, C) -> sum(dim=1) -> (B, C)
        out = torch.sum(x * weights, dim=1)

        return out


class AudioClassifier(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        super(AudioClassifier, self).__init__()

        # 1. Load Backbone (EfficientNet-B0)
        # global_pool='' ensures we get the spatial feature map (B, C, H, W)
        # num_classes=0 removes the default classifier head
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # 2. Input Adaptation (3 channels -> 1 channel)
        # The first layer in EfficientNet is named 'conv_stem'
        original_conv = self.backbone.conv_stem

        # Create new 1-channel convolution with same parameters
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Initialize with sum of RGB weights to preserve spatial filters
        # original weights: (Out, 3, K, K) -> sum(dim=1) -> (Out, 1, K, K)
        with torch.no_grad():
            new_conv.weight.copy_(original_conv.weight.sum(dim=1, keepdim=True))

        self.backbone.conv_stem = new_conv

        # 3. Pooling Head
        self.pooling_type = Config.POOLING_TYPE
        self.num_features = self.backbone.num_features  # 1280 for EfficientNet-B0

        if self.pooling_type == "attention":
            self.pooling = AttentionPooling(self.num_features)
        else:
            # Fallback to Global Average Pooling if config changes
            self.pooling = nn.AdaptiveAvgPool1d(1)

        # 4. Classifier
        self.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        # Input x: (Batch, 1, Freq, Time)

        # Extract features
        # Output: (Batch, Channels, Freq', Time')
        # EfficientNet-B0 downsamples by 32.
        # For 128 mels, Freq' will be 4.
        x = self.backbone.forward_features(x)

        # Aggregate Frequency dimension (Average Pooling)
        # We treat frequency as "height" and time as "width".
        # We collapse frequency to get a sequence of time-step features.
        # (B, C, F', T') -> (B, C, T')
        x = torch.mean(x, dim=2)

        if self.pooling_type == "attention":
            # Permute for Linear layer: (B, C, T') -> (B, T', C)
            x = x.permute(0, 2, 1)
            # Apply Attention Pooling over time
            x = self.pooling(x)  # Output: (B, C)
        else:
            # AdaptiveAvgPool expects (B, C, L)
            x = self.pooling(x)  # Output: (B, C, 1)
            x = x.flatten(1)  # Output: (B, C)

        # Classification
        logits = self.fc(x)

        return logits
