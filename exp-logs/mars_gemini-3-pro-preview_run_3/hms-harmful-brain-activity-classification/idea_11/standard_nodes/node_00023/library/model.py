import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    Used in the adapter to weight the 57 input channels (bands * electrodes)
    before they are projected to the 3-channel RGB space for the backbone.
    """

    def __init__(self, in_channels, reduction=4):
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


class TemporalAttentionPooling(nn.Module):
    """
    Temporal Attention Pooling.
    Collapses the frequency/height dimension via averaging, then applies a
    learnable attention mechanism over the time/width dimension.
    This allows the model to focus on specific time segments (e.g., the central 10s).
    """

    def __init__(self, in_channels, hidden_dim=None):
        super(TemporalAttentionPooling, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_channels // 2

        # Attention mechanism: Conv1d over the time dimension
        self.att_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        # Input: (Batch, Channels, Freq/Height, Time/Width)

        # 1. Collapse Frequency dimension (Global Average Pooling over Height)
        # Result: (Batch, Channels, Time)
        x_time = x.mean(dim=2)

        # 2. Compute Attention Scores
        # Input to conv1d: (Batch, Channels, Time)
        attn_scores = self.att_conv(x_time)  # (Batch, 1, Time)

        # 3. Softmax over Time dimension to get weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (Batch, 1, Time)

        # 4. Weighted Sum over Time
        # (Batch, Channels, Time) * (Batch, 1, Time) -> Sum over Time
        x_weighted = (x_time * attn_weights).sum(dim=-1)  # (Batch, Channels)

        return x_weighted


class BandAdaptiveNet(nn.Module):
    """
    Band-Adaptive Multi-Resolution Network with Temporal Attention.

    Structure:
    1. Stream A (EEG Morphology):
       - Input: (B, 57, 128, 256) [19 electrodes * 3 bands]
       - Adapter: SE-Block -> 1x1 Conv (57->3)
       - Backbone: EfficientNet-B2 (Pretrained)
       - Pooling: Temporal Attention Pooling
    2. Stream B (Long-term Context):
       - Input: (B, 4, 256, 256) [4 spectrogram regions]
       - Adapter: 1x1 Conv (4->3)
       - Backbone: EfficientNet-B0 (Pretrained)
       - Pooling: Global Average Pooling
    3. Fusion:
       - Gated Context: Stream B vector modulates Stream A vector.
    """

    def __init__(self):
        super(BandAdaptiveNet, self).__init__()

        # ==========================
        # Stream A: EEG Morphology
        # ==========================
        # Adapter: Projects 57 band-channels to 3 for the backbone
        self.adapter_a_se = SEBlock(Config.IN_CHANNELS_A, reduction=4)
        self.adapter_a_proj = nn.Conv2d(
            Config.IN_CHANNELS_A, Config.PROJ_CHANNELS_A, kernel_size=1, bias=False
        )
        self.adapter_a_bn = nn.BatchNorm2d(Config.PROJ_CHANNELS_A)
        self.adapter_a_act = nn.SiLU(inplace=True)  # Swish activation

        # Backbone (EfficientNet B2)
        # global_pool='' ensures we get the feature map (B, C, H, W)
        self.backbone_a = timm.create_model(
            Config.BACKBONE_A,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )

        # Determine feature dimension dynamically
        with torch.no_grad():
            # Dummy forward pass to check output shape
            dummy = torch.randn(1, 3, 128, 128)
            feat_a = self.backbone_a(dummy)
            self.dim_a = feat_a.shape[1]

        # Temporal Attention Pooling for Stream A
        self.pool_a = TemporalAttentionPooling(self.dim_a)

        # ==========================
        # Stream B: Long-term Context
        # ==========================
        # Adapter: Projects 4 regions to 3 channels
        self.adapter_b_proj = nn.Conv2d(
            Config.IN_CHANNELS_B, 3, kernel_size=1, bias=False
        )
        self.adapter_b_bn = nn.BatchNorm2d(3)
        self.adapter_b_act = nn.SiLU(inplace=True)

        # Backbone (EfficientNet B0)
        # global_pool='avg' gives us the vector directly
        self.backbone_b = timm.create_model(
            Config.BACKBONE_B,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        # Determine feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 256, 256)
            feat_b = self.backbone_b(dummy)
            self.dim_b = feat_b.shape[1]

        # ==========================
        # Fusion & Head
        # ==========================
        # Context Gate: Projects Stream B to match Stream A's dimension
        self.context_gate = nn.Sequential(
            nn.Linear(self.dim_b, self.dim_a),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(Config.DROPOUT)
        self.classifier = nn.Linear(self.dim_a, Config.NUM_CLASSES)

    def forward(self, x_eeg, x_spec):
        """
        Args:
            x_eeg: Tensor (Batch, 57, 128, 256)
            x_spec: Tensor (Batch, 4, 256, 256)
        """

        # --- Stream A Processing ---
        # 1. Adapter
        a = self.adapter_a_se(x_eeg)  # Channel attention
        a = self.adapter_a_proj(a)  # Project 57 -> 3
        a = self.adapter_a_bn(a)
        a = self.adapter_a_act(a)

        # 2. Backbone
        feat_map_a = self.backbone_a(a)  # (B, C_a, H, W)

        # 3. Temporal Attention Pooling
        vec_a = self.pool_a(feat_map_a)  # (B, C_a)

        # --- Stream B Processing ---
        # 1. Adapter
        b = self.adapter_b_proj(x_spec)  # Project 4 -> 3
        b = self.adapter_b_bn(b)
        b = self.adapter_b_act(b)

        # 2. Backbone + GAP
        vec_b = self.backbone_b(b)  # (B, C_b)

        # --- Fusion ---
        # Calculate Gate from Context (Stream B)
        gate = self.context_gate(vec_b)  # (B, C_a)

        # Modulate Detail (Stream A) with Gate
        feat_fused = vec_a * gate

        # --- Classification ---
        out = self.dropout(feat_fused)
        logits = self.classifier(out)

        return logits
