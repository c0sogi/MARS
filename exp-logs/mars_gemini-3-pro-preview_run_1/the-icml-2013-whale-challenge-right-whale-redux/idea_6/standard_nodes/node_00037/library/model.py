import torch
import torch.nn as nn
import timm
import math
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling Layer.
    Computes a weighted average of the input sequence based on learned attention scores.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        # x shape: (Batch, Time, Features)

        # Calculate attention scores
        # scores shape: (Batch, Time, 1)
        scores = self.attention(x)

        # Normalize scores across the time dimension
        weights = torch.softmax(scores, dim=1)

        # Weighted sum
        # Output shape: (Batch, Features)
        output = torch.sum(x * weights, dim=1)

        return output


class TimePreservingResNetBiGRU(nn.Module):
    """
    CRNN with ResNet18 backbone and Bi-directional GRU head.
    Modifies backbone strides to preserve temporal resolution.
    Cite Lesson 00031: Robustness of Standard ResNets over EfficientNets.
    """

    def __init__(self):
        super(TimePreservingResNetBiGRU, self).__init__()

        # 1. Load Pretrained Backbone
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=Config.PRETRAINED, features_only=False
        )

        # 2. Modify Input Layer (3 Channels -> 1 Channel)
        # Cite Lesson 00020: Adapting ImageNet Weights for Single-Channel Spectrograms
        # Cite debug_lesson_6: Treat Naming Mismatches as Signals of Incomplete Refactoring
        if hasattr(self.backbone, "conv1"):
            original_conv = self.backbone.conv1
            layer_name = "conv1"
        elif hasattr(self.backbone, "conv_stem"):
            original_conv = self.backbone.conv_stem
            layer_name = "conv_stem"
        else:
            raise AttributeError(
                "Could not find first convolutional layer (conv1 or conv_stem)."
            )

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False if original_conv.bias is None else True,
        )

        # Initialize with averaged weights from RGB
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data

        setattr(self.backbone, layer_name, new_conv)

        # 3. Modify Strides for Time Preservation
        # Generic approach: Iterate all Conv2d layers.
        # If stride is (2, 2) or 2, change to (2, 1) to preserve Time dimension.
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (2, 1)
                elif isinstance(module.stride, int) and module.stride == 2:
                    module.stride = (2, 1)

        # 4. Determine Backbone Output Size
        # Run a dummy forward pass
        # Input: (Batch, 1, F, T)
        dummy_input = torch.randn(1, 1, Config.N_MELS, 128)
        with torch.no_grad():
            features = self.backbone.forward_features(dummy_input)

        backbone_out_channels = features.shape[1]

        # 5. RNN Head
        self.gru = nn.GRU(
            input_size=backbone_out_channels,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.RNN_DROPOUT if Config.RNN_LAYERS > 1 else 0,
        )

        # 6. Aggregation and Classification
        self.attn_pooling = AttentionPooling(Config.RNN_HIDDEN_SIZE * 2)
        self.fc = nn.Linear(Config.RNN_HIDDEN_SIZE * 2, 1)

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)

        # Extract features using modified backbone
        # Output: (Batch, Channels, F_out, T_out)
        x = self.backbone.forward_features(x)

        # Frequency Pooling: Average over the frequency dimension
        # We want to keep the Time dimension for the RNN
        # Output: (Batch, Channels, T_out)
        x = torch.mean(x, dim=2)

        # Permute for RNN: (Batch, Time, Channels)
        x = x.permute(0, 2, 1)

        # RNN Processing
        # Output: (Batch, Time, Hidden*2)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        # Attention Pooling
        # Output: (Batch, Hidden*2)
        x = self.attn_pooling(x)

        # Classification
        # Output: (Batch, 1)
        x = self.fc(x)

        return x
