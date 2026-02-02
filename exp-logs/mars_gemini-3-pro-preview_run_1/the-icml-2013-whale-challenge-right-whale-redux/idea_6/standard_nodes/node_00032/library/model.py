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
    Cite solution_lesson_node_00031: Prefer ResNet over EfficientNet for stride modification.
    Cite solution_lesson_node_00012: Preserve temporal resolution.
    """

    def __init__(self):
        super(TimePreservingResNetBiGRU, self).__init__()

        # 1. Load Pretrained Backbone
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=Config.PRETRAINED, features_only=False
        )

        # 2. Modify Input Layer (3 Channels -> 1 Channel)
        # ResNet uses 'conv1'
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )
        # Initialize with averaged weights from RGB (Cite solution_lesson_node_00020)
        with torch.no_grad():
            self.backbone.conv1.weight.data = original_conv1.weight.data.mean(
                dim=1, keepdim=True
            )

        # 3. Modify Strides for Time Preservation
        # Standard ResNet18 downsamples by 32x (Stem 4x, Layer2 2x, Layer3 2x, Layer4 2x).
        # We want to reduce time downsampling.
        # We change strides of Layer 2, 3, 4 to (2, 1) -> Freq /2, Time /1.
        # Total Time Downsample: 4x (Stem).

        layers_to_modify = [
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]

        for layer in layers_to_modify:
            for block in layer:
                # Modify conv1 stride if it is (2, 2)
                if hasattr(block, "conv1") and block.conv1.stride == (2, 2):
                    block.conv1.stride = (2, 1)

                # Modify downsample stride if it exists
                if hasattr(block, "downsample") and block.downsample is not None:
                    for m in block.downsample.modules():
                        if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                            m.stride = (2, 1)

        # 4. Determine Backbone Output Size
        # Run a dummy forward pass
        dummy_input = torch.randn(1, 1, Config.N_MELS, 128)
        with torch.no_grad():
            # forward_features in timm ResNet returns the output of the last layer (before pooling/fc)
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

        # Extract features
        # Output: (Batch, Channels, F_out, T_out)
        x = self.backbone.forward_features(x)

        # Frequency Pooling
        x = torch.mean(x, dim=2)

        # Permute for RNN: (Batch, Time, Channels)
        x = x.permute(0, 2, 1)

        # RNN Processing
        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        # Attention Pooling
        x = self.attn_pooling(x)

        # Classification
        x = self.fc(x)

        return x
