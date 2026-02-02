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
    Uses output_stride=16 to preserve spatial resolution.
    Cite {solution_lesson_node_00017}
    Cite {solution_lesson_node_00025}
    """

    def __init__(self):
        super(DilatedEfficientNet, self).__init__()

        # 1. Backbone: EfficientNet-B2
        # Load with 3 channels first to get standard pretrained weights
        # output_stride=16 enables dilated convolutions in the final stages
        self.backbone = timm.create_model(
            ModelConfig.MODEL_NAME,
            pretrained=ModelConfig.PRETRAINED,
            in_chans=3,
            output_stride=16,
        )

        # 2. Adapt First Layer (RGB -> 1 Channel)
        # Cite {solution_lesson_node_00016}: Average weights instead of random init
        original_conv = self.backbone.conv_stem
        new_conv = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

        self.backbone.conv_stem = new_conv

        # 3. Head
        self.pool = AttentivePooling(self.backbone.num_features)
        self.dropout = nn.Dropout(ModelConfig.DROPOUT)
        self.fc = nn.Linear(self.backbone.num_features, ModelConfig.NUM_CLASSES)

    def forward(self, x):
        # Backbone Forward
        # forward_features returns the final feature map (B, C, H, W)
        x = self.backbone.forward_features(x)

        # Attentive Pooling
        embedding = self.pool(x)

        # Classification
        embedding = self.dropout(embedding)
        logits = self.fc(embedding)

        return logits
