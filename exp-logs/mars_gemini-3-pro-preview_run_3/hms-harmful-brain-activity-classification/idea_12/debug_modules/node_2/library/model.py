import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block to recalibrate channel-wise feature responses.
    """

    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class PhysiologicalAdapter(nn.Module):
    """
    Adapts the 57-channel physiologically aligned input to a 3-channel tensor
    suitable for standard 2D backbones, using SE attention and 1x1 convolution.
    """

    def __init__(self, in_channels, out_channels=3):
        super(PhysiologicalAdapter, self).__init__()
        # Reduction ratio 4 allows capturing inter-channel dependencies (19 electrodes * 3 views)
        # without excessive parameters.
        self.se = SEBlock(in_channels, reduction=4)
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.se(x)
        x = self.project(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TemporalAttentionPooling(nn.Module):
    """
    Pools feature maps by attending to specific time steps, preserving temporal localization
    before aggregation.
    """

    def __init__(self, in_channels, hidden_dim=None):
        super(TemporalAttentionPooling, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_channels // 8  # Lightweight attention head

        self.att_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        # Input x: (Batch, Channels, Freq, Time)

        # 1. Collapse Frequency dimension (Global Average Pooling over Freq)
        # Result: (Batch, Channels, Time)
        x_t = x.mean(dim=2)

        # 2. Compute Attention Weights over Time
        # (Batch, 1, Time)
        attn_logits = self.att_conv(x_t)
        attn_weights = F.softmax(attn_logits, dim=2)

        # 3. Weighted Sum over Time
        # (Batch, C, T) * (Batch, 1, T) -> Sum over T -> (Batch, C)
        x_weighted = (x_t * attn_weights).sum(dim=2)

        return x_weighted


class MultiResNetwork(nn.Module):
    """
    Physiologically-Aligned Multi-Resolution Network with Temporal Attention.
    Dual-stream architecture processing short-term morphological details and long-term context.
    """

    def __init__(self):
        super(MultiResNetwork, self).__init__()

        # ====================================================
        # Stream A: Physiologically-Aligned Morphological Encoder
        # ====================================================
        # Input: (Batch, 57, 128, 500)
        self.adapter = PhysiologicalAdapter(
            in_channels=Config.IN_CHANNELS_A, out_channels=3
        )

        # Backbone: EfficientNet-B2 (Pretrained)
        # We use features_only=True to access spatial feature maps for temporal attention
        self.backbone_a = timm.create_model(
            Config.BACKBONE_A,
            pretrained=Config.PRETRAINED,
            features_only=True,
            in_chans=3,
        )

        # Determine output channels dynamically
        dummy_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            features = self.backbone_a(dummy_input)
            out_channels_a = features[-1].shape[1]  # Typically 1408 for B2

        self.temporal_pool = TemporalAttentionPooling(out_channels_a)

        # ====================================================
        # Stream B: Long-Term Context Encoder
        # ====================================================
        # Input: (Batch, 4, 256, 256)
        # Backbone: EfficientNet-B0 (Pretrained)
        # num_classes=0 returns the global pooled feature vector
        self.backbone_b = timm.create_model(
            Config.BACKBONE_B,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            in_chans=Config.IN_CHANNELS_B,
        )
        out_channels_b = self.backbone_b.num_features  # Typically 1280 for B0

        # ====================================================
        # Fusion Head: Gated Context Mechanism
        # ====================================================
        fusion_dim = 512

        # Project Stream A (Detail)
        self.proj_a = nn.Sequential(
            nn.Linear(out_channels_a, fusion_dim), nn.BatchNorm1d(fusion_dim), nn.ReLU()
        )

        # Project Stream B (Context) to generate Gate
        self.gate_b = nn.Sequential(
            nn.Linear(out_channels_b, fusion_dim), nn.Sigmoid()  # Gate values [0, 1]
        )

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(fusion_dim, Config.NUM_CLASSES)
        )

    def forward(self, inputs):
        # Unpack inputs tuple
        x_a, x_b = inputs

        # --- Stream A Processing ---
        x_a = self.adapter(x_a)  # (B, 3, 128, 500)
        feats_a = self.backbone_a(x_a)[-1]  # (B, 1408, F', T')
        vec_a = self.temporal_pool(feats_a)  # (B, 1408)

        # --- Stream B Processing ---
        vec_b = self.backbone_b(x_b)  # (B, 1280)

        # --- Gated Fusion ---
        # "Context vector modulates detail vector"
        detail = self.proj_a(vec_a)  # (B, 512)
        gate = self.gate_b(vec_b)  # (B, 512)

        fused = detail * gate  # Element-wise multiplication

        # --- Classification ---
        logits = self.classifier(fused)

        # Return probabilities
        return F.softmax(logits, dim=1)
