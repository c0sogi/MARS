import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class AttentivePooling(nn.Module):
    """
    Attentive Pooling module that learns to weight time frames dynamically.
    It helps in focusing on the active speech segments and suppressing background noise.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(in_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, C, T)
        # Calculate attention weights
        w = self.attn(x)  # (B, 1, T)
        w = F.softmax(w, dim=2)
        # Weighted sum over time
        x = torch.sum(x * w, dim=2)  # (B, C)
        return x


class DilatedEfficientNet(nn.Module):
    """
    Dilated EfficientNet-B2 with Multi-Sample Dropout.

    Modifications:
    1. Input layer adapted for 1-channel Log-Mel Spectrograms.
    2. Last stage blocks modified to use Dilated Convolutions (rate=2) to preserve temporal resolution.
    3. Attentive Pooling for temporal aggregation.
    4. Multi-Sample Dropout head for implicit ensembling.
    """

    def __init__(self, num_classes):
        super().__init__()
        # Load EfficientNet B2
        # features_only=False allows us to use forward_features easily
        self.backbone = timm.create_model(
            "efficientnet_b2", pretrained=True, features_only=False
        )

        # 1. Modify Input Layer (3 channels -> 1 channel)
        original_conv = self.backbone.conv_stem
        new_conv = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Initialize by averaging the pretrained RGB weights
        # Shape: (Out, 3, K, K) -> (Out, 1, K, K)
        with torch.no_grad():
            new_conv.weight[:] = torch.mean(original_conv.weight, dim=1, keepdim=True)
        self.backbone.conv_stem = new_conv

        # 2. Dilated Convolutions in the last stage
        # We iterate over the blocks of the last stage to increase dilation and adjust padding/stride.
        last_stage_idx = len(self.backbone.blocks) - 1
        for block in self.backbone.blocks[last_stage_idx]:
            # Iterate through modules to find Conv2d layers
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    # Heuristic: Identify depthwise convolutions by groups == in_channels
                    if m.groups == m.in_channels and m.in_channels > 1:
                        m.dilation = (2, 2)
                        m.padding = (2, 2)  # Adjust padding for dilation 2, kernel 3
                        # If stride was 2 (downsampling), set to 1 to preserve resolution
                        if m.stride == (2, 2):
                            m.stride = (1, 1)

        # 3. Pooling & Head
        self.pool = AttentivePooling(self.backbone.num_features)

        # Multi-Sample Dropout
        # Create multiple dropout masks that feed into the same shared linear layer
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.DROPOUT_RATE) for _ in range(Config.NUM_DROPOUTS)]
        )
        self.fc = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        # Input x: (Batch, 1, Freq, Time)

        # Extract features from backbone
        x = self.backbone.forward_features(x)  # (Batch, Channels, Freq', Time')

        # Collapse Frequency dimension via Global Average Pooling
        x = torch.mean(x, dim=2)  # (Batch, Channels, Time')

        # Attentive Pooling over Time
        x = self.pool(x)  # (Batch, Channels)

        # Multi-Sample Dropout
        logits = []
        for dropout in self.dropouts:
            # Apply dropout and then the shared classifier
            logits.append(self.fc(dropout(x)))

        # Average the logits from all dropout samples
        return torch.mean(torch.stack(logits), dim=0)
