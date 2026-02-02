import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class InceptionBlock1D(nn.Module):
    """
    A single Inception block for 1D data (EEG).
    Consists of parallel convolutional branches with different kernel sizes.
    """

    def __init__(self, in_channels, out_channels_per_branch, kernels):
        super().__init__()
        self.branches = nn.ModuleList()

        # Create parallel branches
        for k, out_ch in zip(kernels, out_channels_per_branch):
            # Calculate padding to maintain temporal dimension (same padding)
            pad = k // 2
            self.branches.append(
                nn.Conv1d(in_channels, out_ch, kernel_size=k, padding=pad, bias=False)
            )

        total_out_channels = sum(out_channels_per_branch)
        self.bn = nn.BatchNorm1d(total_out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)
        out = self.bn(out)
        out = self.act(out)
        return out


class Inception1DEncoder(nn.Module):
    """
    Stream A: Raw EEG Encoder using Multi-Scale 1D CNNs.
    Downsamples the 5000-step EEG signal into a sequence of feature tokens.
    """

    def __init__(self):
        super().__init__()

        self.in_channels = Config.EEG_CHANNELS
        self.kernels = Config.EEG_KERNELS
        self.filters = Config.EEG_FILTERS
        self.embed_dim = Config.EEG_EMBED_DIM

        # Determine the output dimension of a single inception block
        # If filters are [32, 32, 32, 32], sum is 128.
        self.block_out_dim = sum(self.filters)

        # Ensure configuration consistency
        if self.block_out_dim != self.embed_dim:
            # If they don't match, we might need a projection, but for this specific
            # design we assume they match or we project at the end.
            # Based on prompt logic, we'll project if needed, but Config implies 128.
            pass

        # Build hierarchical structure
        # Layer 1
        self.layer1 = InceptionBlock1D(self.in_channels, self.filters, self.kernels)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Layer 2
        self.layer2 = InceptionBlock1D(self.block_out_dim, self.filters, self.kernels)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Layer 3
        self.layer3 = InceptionBlock1D(self.block_out_dim, self.filters, self.kernels)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Layer 4
        self.layer4 = InceptionBlock1D(self.block_out_dim, self.filters, self.kernels)
        # Final pooling to get a reasonable sequence length
        # 5000 / (4*4*4) approx 78. Leaving it as sequence for attention.

        self.out_proj = (
            nn.Conv1d(self.block_out_dim, self.embed_dim, kernel_size=1)
            if self.block_out_dim != self.embed_dim
            else nn.Identity()
        )

    def forward(self, x):
        # x: (B, 20, 5000)
        x = self.layer1(x)
        x = self.pool1(x)

        x = self.layer2(x)
        x = self.pool2(x)

        x = self.layer3(x)
        x = self.pool3(x)

        x = self.layer4(x)

        x = self.out_proj(x)
        # Output: (B, 128, T') where T' is approx 78

        # Permute for Attention (Batch, Seq, Dim)
        return x.permute(0, 2, 1)


class SpecEncoderFPN(nn.Module):
    """
    Stream B: Pyramid Coordinate-Aware Spectrogram Encoder.
    Uses EfficientNet-B0 + FPN Neck to produce high-resolution spatial-temporal tokens.
    """

    def __init__(self):
        super().__init__()

        # Input adapter: 5 channels -> 3 channels
        self.input_adapter = nn.Conv2d(
            Config.SPEC_CHANNELS, 3, kernel_size=1, bias=False
        )

        # Backbone: EfficientNet B0
        # Features only, extracting indices 2 (C3), 3 (C4), 4 (C5)
        # Strides: C3=8, C4=16, C5=32
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(2, 3, 4),
        )

        feature_info = self.backbone.feature_info.channels()
        c3_dim, c4_dim, c5_dim = feature_info[0], feature_info[1], feature_info[2]
        out_dim = Config.FPN_OUT_CHANNELS

        # FPN Lateral Connections (1x1 Convs)
        self.lat_c5 = nn.Conv2d(c5_dim, out_dim, kernel_size=1)
        self.lat_c4 = nn.Conv2d(c4_dim, out_dim, kernel_size=1)
        self.lat_c3 = nn.Conv2d(c3_dim, out_dim, kernel_size=1)

        # FPN Output Convs (3x3 Convs to smooth aliasing)
        self.out_p3 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, 5, 512, 512)
        x = self.input_adapter(x)

        # Extract features
        features = self.backbone(x)
        c3, c4, c5 = features[0], features[1], features[2]

        # Top-down pathway
        p5 = self.lat_c5(c5)  # (B, 128, H/32, W/32)

        p4_in = self.lat_c4(c4)  # (B, 128, H/16, W/16)
        p4 = p4_in + F.interpolate(p5, size=p4_in.shape[-2:], mode="nearest")

        p3_in = self.lat_c3(c3)  # (B, 128, H/8, W/8)
        p3 = p3_in + F.interpolate(p4, size=p3_in.shape[-2:], mode="nearest")

        # Smooth output
        p3 = self.out_p3(p3)  # (B, 128, 64, 64)

        # Tokenize: Flatten spatial dimensions
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        b, c, h, w = p3.shape
        tokens = p3.view(b, c, h * w).permute(0, 2, 1)

        return tokens


class PyramidFusionNet(nn.Module):
    """
    Main Model: Pyramid-Resolution Coordinate-Guided Fusion Network.
    Combines EEG and Spectrogram streams via Asymmetric Cross-Attention.
    """

    def __init__(self):
        super().__init__()

        self.eeg_encoder = Inception1DEncoder()
        self.spec_encoder = SpecEncoderFPN()

        embed_dim = Config.EEG_EMBED_DIM  # 128

        # Fusion: Asymmetric Cross-Attention
        # Query: EEG (High res temporal features)
        # Key/Value: Spectrogram (High res spatial-temporal context)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=Config.ATTN_NUM_HEADS,
            dropout=Config.ATTN_DROPOUT,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(Config.ATTN_DROPOUT)

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, Config.NUM_CLASSES),
        )

    def forward(self, eeg, spec):
        # Stream A: EEG -> (B, T_eeg, C)
        eeg_feats = self.eeg_encoder(eeg)

        # Stream B: Spectrogram -> (B, T_spec, C)
        # T_spec is H*W from FPN output (e.g., 4096)
        spec_feats = self.spec_encoder(spec)

        # Cross Attention: EEG queries Spectrogram
        # Q=EEG, K=Spec, V=Spec
        attn_out, _ = self.cross_attn(query=eeg_feats, key=spec_feats, value=spec_feats)

        # Residual connection + Norm
        # Note: Standard transformer add-norm usually happens here
        fused = self.norm(eeg_feats + self.dropout(attn_out))

        # Global Average Pooling over the EEG time dimension
        # (B, T_eeg, C) -> (B, C)
        pooled = torch.mean(fused, dim=1)

        # Classification
        logits = self.classifier(pooled)

        # Output probabilities via Softmax
        return F.softmax(logits, dim=1)
