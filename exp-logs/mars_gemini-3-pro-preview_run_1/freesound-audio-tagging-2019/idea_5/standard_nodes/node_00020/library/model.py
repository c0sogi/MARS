import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling module for aggregating temporal features.

    Learns a weight for each time step and computes a weighted sum.
    Input: (Batch, Channels, Time)
    Output: (Batch, Channels)
    """

    def __init__(self, in_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, 128), nn.Tanh(), nn.Linear(128, 1), nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        # Transpose to (Batch, Time, Channels) for Linear layers
        w = x.transpose(1, 2)
        w = self.attention(w)  # (Batch, Time, 1)

        # Transpose back to apply weights: (Batch, 1, Time)
        w = w.transpose(1, 2)

        # Weighted sum over time dimension
        # x * w -> (Batch, Channels, Time)
        out = torch.sum(x * w, dim=2)
        return out


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout Head.

    Applies multiple dropout masks to the input features, passes them through
    a shared linear layer, and averages the outputs.
    """

    def __init__(self, in_features, out_features, p=0.5, num_samples=5):
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(num_samples)])
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (Batch, In_Features)
        logits = []
        for dropout in self.dropouts:
            # Apply dropout then linear
            op = self.linear(dropout(x))
            logits.append(op)

        # Stack and average: (Batch, Out_Features)
        return torch.stack(logits).mean(dim=0)


class AudioEfficientNet(nn.Module):
    """
    Audio Tagging Model based on EfficientNet-B3.

    Pipeline:
    1. Learnable BatchNorm on Input.
    2. Input Repetition (1ch -> 3ch).
    3. EfficientNet-B3 Backbone (Pretrained).
    4. Frequency Pooling (Avg).
    5. Time Aggregation (Attention Pooling).
    6. Classification Head (Multi-Sample Dropout).
    """

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(AudioEfficientNet, self).__init__()

        # 1. Learnable Batch Normalization for input spectrograms
        # Input shape: (Batch, 1, Freq, Time)
        self.bn0 = nn.BatchNorm2d(Config.IN_CHANNELS)

        # 2. Backbone
        # We use timm to create EfficientNet-B3.
        # in_chans=3 because we will repeat the input.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=3,
            num_classes=0,  # No classification head
            global_pool="",  # No global pooling, we need feature maps
        )

        # Get feature dimension of the backbone (1536 for B3)
        self.n_features = self.backbone.num_features

        # 3. Pooling & Aggregation
        # Attention Pooling over time
        self.pool = AttentionPooling(self.n_features)

        # 4. Classification Head
        if Config.USE_MULTI_SAMPLE_DROPOUT:
            self.fc = MultiSampleDropout(
                in_features=self.n_features,
                out_features=num_classes,
                p=Config.DROPOUT_RATE,
                num_samples=5,
            )
        else:
            self.fc = nn.Sequential(
                nn.Dropout(Config.DROPOUT_RATE), nn.Linear(self.n_features, num_classes)
            )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input spectrograms of shape (Batch, 1, Freq, Time)

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        # 1. Normalize
        x = self.bn0(x)

        # 2. Input Repetition (1ch -> 3ch)
        # EfficientNet expects RGB-like input
        if Config.INPUT_REPETITION:
            x = x.repeat(1, 3, 1, 1)

        # 3. Backbone Features
        # Output shape: (Batch, Channels, H_freq, W_time)
        # For B3: (Batch, 1536, Freq/32, Time/32)
        x = self.backbone(x)

        # 4. Frequency Pooling
        # Average over the frequency dimension (H) to make it invariant to pitch shifts within the band
        # and reduce dimensionality.
        # Shape: (Batch, Channels, Time)
        x = torch.mean(x, dim=2)

        # 5. Time Aggregation (Attention Pooling)
        # Shape: (Batch, Channels)
        x = self.pool(x)

        # 6. Classification Head
        # Shape: (Batch, Num_Classes)
        x = self.fc(x)

        return x
