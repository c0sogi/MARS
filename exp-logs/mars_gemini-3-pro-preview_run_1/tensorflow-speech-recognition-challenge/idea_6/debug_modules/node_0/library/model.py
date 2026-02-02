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


class FeaturePyramidEfficientNet(nn.Module):
    """
    EfficientNet-B2 with Feature Pyramid Network (FPN) neck and Attentive Pooling head.
    """

    def __init__(self):
        super(FeaturePyramidEfficientNet, self).__init__()

        # 1. Backbone: EfficientNet-B2
        # in_chans=1 automatically adapts the first conv layer by averaging RGB weights
        # out_indices=(2, 3, 4) corresponds to stages with strides 8, 16, 32
        self.backbone = timm.create_model(
            ModelConfig.MODEL_NAME,
            pretrained=ModelConfig.PRETRAINED,
            features_only=True,
            out_indices=(2, 3, 4),
            in_chans=ModelConfig.IN_CHANNELS,
        )

        # Get channel counts for the extracted features
        # feature_info returns info for the selected out_indices
        feature_channels = self.backbone.feature_info.channels()
        c3, c4, c5 = feature_channels

        fpn_dim = ModelConfig.FPN_CHANNELS

        # 2. FPN Lateral Connections (1x1 Convs)
        self.lat5 = nn.Conv2d(c5, fpn_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, fpn_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, fpn_dim, kernel_size=1)

        # 3. Head
        self.pool = AttentivePooling(fpn_dim)
        self.dropout = nn.Dropout(ModelConfig.DROPOUT)
        self.fc = nn.Linear(fpn_dim, ModelConfig.NUM_CLASSES)

    def forward(self, x):
        # Backbone Forward
        # features is a list of tensors [P3, P4, P5]
        features = self.backbone(x)
        p3_in, p4_in, p5_in = features

        # Lateral Projections
        p5 = self.lat5(p5_in)
        p4 = self.lat4(p4_in)
        p3 = self.lat3(p3_in)

        # Top-Down Pathway (FPN)
        # Upsample P5 and add to P4
        p4 = p4 + F.interpolate(
            p5, size=p4.shape[-2:], mode="bilinear", align_corners=False
        )

        # Upsample P4 (fused) and add to P3
        p3 = p3 + F.interpolate(
            p4, size=p3.shape[-2:], mode="bilinear", align_corners=False
        )

        # The output of the FPN is the high-resolution fused map (P3)

        # Attentive Pooling
        embedding = self.pool(p3)

        # Classification
        embedding = self.dropout(embedding)
        logits = self.fc(embedding)

        return logits
