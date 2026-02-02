import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class InceptionBlock1D(nn.Module):
    """
    A single Inception-style block for 1D EEG data.
    Applies parallel convolutions with different kernel sizes, concatenates results,
    and applies Batch Normalization, Activation, and Max Pooling.
    """

    def __init__(
        self, in_channels, out_channels, kernel_sizes, pool_size=4, drop_rate=0.0
    ):
        super().__init__()
        self.branches = nn.ModuleList()

        # Calculate output channels per branch to sum up to roughly out_channels
        # We want the concatenated output to be 'out_channels'.
        # If out_channels is not divisible by len(kernel_sizes), the last branch takes the remainder.
        n_branches = len(kernel_sizes)
        branch_channels = out_channels // n_branches

        current_in = 0
        for i, k in enumerate(kernel_sizes):
            # Adjust last branch to match exact out_channels count
            c_out = branch_channels if i < n_branches - 1 else out_channels - current_in

            # Padding to keep length consistent: k // 2
            self.branches.append(
                nn.Conv1d(in_channels, c_out, kernel_size=k, padding=k // 2, bias=False)
            )
            current_in += c_out

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        # x: (B, C_in, L)
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)  # (B, C_out, L)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.pool(out)  # (B, C_out, L // pool_size)
        return out


class EEGEncoder(nn.Module):
    """
    Stream A: Raw EEG Encoder using Multi-Scale 1D Convolutions.
    Processes 20-channel EEG signals to extract phase and frequency features.
    """

    def __init__(self, config):
        super().__init__()
        self.kernels = config.EEG_KERNEL_SIZES
        in_channels = config.EEG_CHANNELS

        # Define a sequence of Inception blocks
        # We progressively increase channels and reduce sequence length.
        # Input: (B, 20, 5000)

        # Block 1: 20 -> 64, Pool 4 -> L=1250
        self.block1 = InceptionBlock1D(
            in_channels, 64, self.kernels, pool_size=4, drop_rate=0.1
        )

        # Block 2: 64 -> 128, Pool 4 -> L=312
        self.block2 = InceptionBlock1D(
            64, 128, self.kernels, pool_size=4, drop_rate=0.1
        )

        # Block 3: 128 -> 256, Pool 4 -> L=78
        self.block3 = InceptionBlock1D(
            128, 256, self.kernels, pool_size=4, drop_rate=0.1
        )

        # Block 4: 256 -> 512, Pool 2 -> L=39
        self.block4 = InceptionBlock1D(
            256, 512, self.kernels, pool_size=2, drop_rate=0.1
        )

        self.out_channels = 512

    def forward(self, x):
        # x: (B, 20, 5000)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x  # (B, 512, 39)


class SpecEncoder(nn.Module):
    """
    Stream B: Coordinate-Aware Spectrogram Encoder.
    Uses EfficientNet-B0 to process 5-channel inputs (4 regions + 1 coordinate map).
    """

    def __init__(self, config):
        super().__init__()
        # Load EfficientNet-B0, modify input conv to accept 5 channels
        self.backbone = timm.create_model(
            config.ENCODER_NAME,
            pretrained=config.PRETRAINED,
            in_chans=config.SPEC_CHANNELS,
            features_only=True,
            out_indices=(4,),  # Get the last feature map
        )

        # EfficientNet-B0 last stage usually has 1280 channels
        # We verify this dynamically or hardcode based on architecture knowledge (1280 for B0)
        dummy_input = torch.randn(1, config.SPEC_CHANNELS, 256, 256)
        with torch.no_grad():
            out = self.backbone(dummy_input)
            self.out_channels = out[0].shape[1]  # Should be 1280

    def forward(self, x):
        # x: (B, 5, H, W)
        features = self.backbone(x)[0]  # (B, 1280, H/32, W/32)
        return features


class BottleneckProjectedFusionNet(nn.Module):
    """
    The main architecture fusing EEG and Spectrogram streams via Low-Rank Bottleneck Projections
    and Asymmetric Cross-Attention.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # 1. Encoders
        self.eeg_encoder = EEGEncoder(config)
        self.spec_encoder = SpecEncoder(config)

        # 2. Bottleneck Projections
        # Project high-dim features to a compact semantic space (D=128)
        # Using 1x1 Convs is equivalent to Linear projection per token
        self.eeg_proj = nn.Sequential(
            nn.Conv1d(
                self.eeg_encoder.out_channels, config.BOTTLENECK_DIM, kernel_size=1
            ),
            nn.BatchNorm1d(config.BOTTLENECK_DIM),
            nn.ReLU(),
        )

        self.spec_proj = nn.Sequential(
            nn.Conv2d(
                self.spec_encoder.out_channels, config.BOTTLENECK_DIM, kernel_size=1
            ),
            nn.BatchNorm2d(config.BOTTLENECK_DIM),
            nn.ReLU(),
        )

        # 3. Fusion: Cross Attention
        # EEG queries Spectrogram
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.BOTTLENECK_DIM,
            num_heads=4,
            batch_first=True,
            dropout=config.DROP_RATE,
        )

        # 4. Classification Head
        self.fc = nn.Linear(config.BOTTLENECK_DIM, config.NUM_CLASSES)

    def forward(self, eeg, spec, targets=None):
        # --- Stream A: EEG ---
        # Input: (B, 20, 5000)
        eeg_feat = self.eeg_encoder(eeg)  # (B, 512, L_eeg)

        # Bottleneck Projection
        eeg_emb = self.eeg_proj(eeg_feat)  # (B, 128, L_eeg)
        # Permute for Attention: (B, L_eeg, 128)
        eeg_tokens = eeg_emb.permute(0, 2, 1)

        # --- Stream B: Spectrogram ---
        # Input: (B, 5, 512, 512)
        spec_feat = self.spec_encoder(spec)  # (B, 1280, H', W')

        # Bottleneck Projection
        spec_emb = self.spec_proj(spec_feat)  # (B, 128, H', W')
        # Flatten spatial dims to tokens: (B, 128, N_spec)
        B, C, H, W = spec_emb.shape
        spec_tokens = spec_emb.view(B, C, -1).permute(0, 2, 1)  # (B, N_spec, 128)

        # --- Fusion ---
        # Query: EEG (Temporal dynamics looking for context)
        # Key/Value: Spectrogram (Global context with coordinate info)
        attn_out, _ = self.cross_attn(
            query=eeg_tokens, key=spec_tokens, value=spec_tokens
        )  # (B, L_eeg, 128)

        # --- Pooling & Classification ---
        # Global Average Pooling over the EEG sequence length
        pooled = torch.mean(attn_out, dim=1)  # (B, 128)

        logits = self.fc(pooled)  # (B, 6)

        return logits
