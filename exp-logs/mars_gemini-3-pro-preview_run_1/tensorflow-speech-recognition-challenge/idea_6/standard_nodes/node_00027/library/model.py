import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import ModelConfig


class AttentivePooling(nn.Module):
    """
    Attentive Pooling mechanism to dynamically weight relevant time-frequency bins.
    """

    def __init__(self, in_channels):
        super(AttentivePooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: (Batch, Channels, Height, Width)
        # Calculate attention scores
        attn_logits = self.attention(x)  # (B, 1, H, W)

        # Flatten spatial dimensions for softmax
        b, c, h, w = x.size()
        attn_logits = attn_logits.view(b, 1, -1)  # (B, 1, H*W)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, 1, H*W)

        # Flatten features
        x_flat = x.view(b, c, -1)  # (B, C, H*W)

        # Weighted sum (Matrix multiplication)
        # (B, C, H*W) @ (B, H*W, 1) -> (B, C, 1)
        pooled = torch.bmm(x_flat, attn_weights.transpose(1, 2))

        return pooled.view(b, c)


class DilatedEfficientNet(nn.Module):
    """
    EfficientNet-B2 with Dilated Convolutions and Attentive Pooling.
    Replaces FPN based on Lesson 25 (Prefer Dilated Convolutions over FPN).
    """

    def __init__(self):
        super(DilatedEfficientNet, self).__init__()

        # 1. Backbone: EfficientNet-B2
        # Cite Lesson 17: Use Dilated Convolutions (output_stride=16) to preserve resolution
        # Cite Lesson 25: Prefer Dilated Convolutions over FPN
        self.backbone = timm.create_model(
            ModelConfig.MODEL_NAME,
            pretrained=ModelConfig.PRETRAINED,
            in_chans=ModelConfig.IN_CHANNELS,
            num_classes=0,
            global_pool="",  # Keep spatial dimensions
            output_stride=16,
        )

        # Cite Lesson 16: Initialize input layer by averaging weights
        # timm sums weights by default for in_chans=1 adaptation. We divide by 3 to get average.
        if ModelConfig.IN_CHANNELS == 1:
            self.backbone.conv_stem.weight.data /= 3.0

        dim = self.backbone.num_features

        # 2. Head
        # Cite Lesson 17: Attentive Pooling
        self.pool = AttentivePooling(dim)
        self.dropout = nn.Dropout(ModelConfig.DROPOUT)
        self.fc = nn.Linear(dim, ModelConfig.NUM_CLASSES)

    def forward(self, x):
        # Backbone Forward
        x = self.backbone(x)

        # Attentive Pooling
        embedding = self.pool(x)

        # Classification
        embedding = self.dropout(embedding)
        logits = self.fc(embedding)

        return logits
