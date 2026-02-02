import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D signals.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Inception1D(nn.Module):
    """
    1D Inception Module with Squeeze-and-Excitation.
    Features parallel convolutions with different kernel sizes to capture multi-scale temporal patterns.
    """

    def __init__(self, in_channels, out_filters=32, kernels=[3, 5, 7]):
        super(Inception1D, self).__init__()

        # Parallel branches
        self.branches = nn.ModuleList()
        for k in kernels:
            # Padding = (k - 1) // 2 to maintain temporal dimension
            pad = (k - 1) // 2
            self.branches.append(
                nn.Conv1d(
                    in_channels, out_filters, kernel_size=k, padding=pad, bias=False
                )
            )

        # Output channels = sum of filters from all branches
        self.concat_channels = out_filters * len(kernels)

        self.bn = nn.BatchNorm1d(self.concat_channels)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock1D(self.concat_channels, reduction=8)

    def forward(self, x):
        # Apply parallel convolutions
        outputs = [branch(x) for branch in self.branches]
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        return x


class EEGEncoder(nn.Module):
    """
    Stream A: SE-Inception 1D Encoder.
    Processes raw EEG signals (Channels, Time) into sequence features and a global vector.
    """

    def __init__(self, config: Config):
        super(EEGEncoder, self).__init__()

        # Stem: Initial convolution to project input channels
        # Downsample by 2 initially
        self.stem = nn.Sequential(
            nn.Conv1d(
                config.EEG_CHANNELS, 32, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList()
        in_ch = 32

        # Stack Inception Blocks
        for i in range(config.INCEPTION_DEPTH):
            # Each block outputs len(kernels) * filters = 3 * 32 = 96 channels
            block = Inception1D(
                in_channels=in_ch,
                out_filters=config.INCEPTION_FILTERS,
                kernels=config.INCEPTION_KERNELS,
            )
            self.blocks.append(block)

            # Update in_ch for next block (output of Inception is 96)
            in_ch = config.INCEPTION_FILTERS * len(config.INCEPTION_KERNELS)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.out_channels = in_ch  # 96

    def forward(self, x):
        # x: (Batch, 20, 5000)
        x = self.stem(x)

        # Pass through Inception blocks with MaxPooling
        for block in self.blocks:
            x = block(x)
            x = self.pool(x)

        # x: (Batch, 96, T_reduced)
        # T_reduced approx 5000 / 2 / (2^6) = 39

        seq_features = x
        global_vec = self.global_pool(x).flatten(1)

        return seq_features, global_vec


class SpectrogramEncoder(nn.Module):
    """
    Stream B: Coordinate-Focused Spectrogram Encoder.
    Uses EfficientNet-B0 backbone.
    """

    def __init__(self, config: Config):
        super(SpectrogramEncoder, self).__init__()
        # Load EfficientNet-B0, modify input channels to 5
        self.backbone = timm.create_model(
            config.BACKBONE_2D,
            pretrained=config.PRETRAINED,
            in_chans=config.SPEC_CHANNELS,
            num_classes=0,  # Remove classifier
            global_pool="",  # Return feature maps
        )

        # Get output channels (1280 for EfficientNet-B0)
        self.out_channels = self.backbone.num_features

    def forward(self, x):
        # x: (Batch, 5, 512, 512)
        return self.backbone(x)


class CrossAttention(nn.Module):
    """
    Asymmetric Cross-Attention Block.
    EEG features (Query) attend to Spectrogram features (Key/Value).
    """

    def __init__(self, eeg_dim, spec_dim, embed_dim=256, num_heads=4, dropout=0.1):
        super(CrossAttention, self).__init__()

        self.q_proj = nn.Linear(eeg_dim, embed_dim)
        self.k_proj = nn.Linear(spec_dim, embed_dim)
        self.v_proj = nn.Linear(spec_dim, embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, eeg_seq, spec_map):
        # eeg_seq: (Batch, C_eeg, T)
        # spec_map: (Batch, C_spec, H, W)

        b, c_eeg, t = eeg_seq.size()
        b, c_spec, h, w = spec_map.size()

        # Prepare Query
        # (B, T, C_eeg) -> (B, T, Embed)
        q = self.q_proj(eeg_seq.permute(0, 2, 1))

        # Prepare Key/Value
        # Flatten spatial dims: (B, C_spec, HW) -> (B, HW, C_spec)
        flat_spec = spec_map.flatten(2).permute(0, 2, 1)
        k = self.k_proj(flat_spec)
        v = self.v_proj(flat_spec)

        # Attention
        attn_out, _ = self.attn(q, k, v)

        # Residual + Norm (Optional, but good for stability)
        # Note: Dimensions match only if q was also embed_dim.
        # Here we just return the attended features.
        x = self.norm(attn_out)

        # Pool over time to get global attention vector
        # (B, T, Embed) -> (B, Embed)
        return x.mean(dim=1)


class HarmfulBrainActivityModel(nn.Module):
    """
    Coordinate-Focused Dual-Stream Network.
    Fuses EEG and Spectrogram data with coordinate-aware pooling.
    """

    def __init__(self, config: Config = Config):
        super(HarmfulBrainActivityModel, self).__init__()
        self.config = config

        # Encoders
        self.eeg_encoder = EEGEncoder(config)
        self.spec_encoder = SpectrogramEncoder(config)

        # Cross Attention
        # EEG dim = 96, Spec dim = 1280
        self.cross_attn = CrossAttention(
            eeg_dim=self.eeg_encoder.out_channels,
            spec_dim=self.spec_encoder.out_channels,
            embed_dim=256,
            num_heads=config.ATTENTION_HEADS,
            dropout=0.1,
        )

        # Classification Head
        # Inputs:
        # 1. EEG Global Vector (96)
        # 2. Attention Vector (256)
        # 3. Spec Global Vector (1280)
        # 4. Spec Focused Vector (1280)
        total_dim = (
            self.eeg_encoder.out_channels + 256 + 2 * self.spec_encoder.out_channels
        )

        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT),
            nn.Linear(total_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(512, config.NUM_CLASSES),
        )

    def dual_pooling(self, spec_features, coord_channel):
        """
        Computes Global Context Vector and Focused Event Vector.

        Args:
            spec_features: (B, C, H, W) Deep features from EfficientNet.
            coord_channel: (B, 1, H_in, W_in) Original coordinate map input.
        """
        # 1. Global Context Vector (Standard GAP)
        v_global = spec_features.mean(dim=(2, 3))

        # 2. Focused Event Vector
        # Downsample coordinate channel to feature map size
        # coord_channel is (B, 1, 512, 512), features are (B, C, 16, 16)
        target_h, target_w = spec_features.shape[2], spec_features.shape[3]

        # Use bilinear interpolation to downsample the Gaussian mask
        # The mask is already Gaussian (0 to 1), so interpolation preserves the peak
        mask = F.interpolate(
            coord_channel,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        # Apply mask (Weighted Average Pooling)
        # Sum(Feats * Mask) / Sum(Mask)
        numerator = (spec_features * mask).sum(dim=(2, 3))
        denominator = mask.sum(dim=(2, 3)) + 1e-6

        v_focus = numerator / denominator

        return v_global, v_focus

    def forward(self, x_eeg, x_spec):
        """
        Args:
            x_eeg: (B, 20, 5000)
            x_spec: (B, 5, 512, 512)
        """
        # Extract Coordinate Channel (Channel 4, 0-indexed)
        # We need this for the focused pooling
        coord_channel = x_spec[:, 4:5, :, :]

        # Stream A: EEG
        eeg_seq, eeg_vec = self.eeg_encoder(x_eeg)

        # Stream B: Spectrogram
        spec_feat = self.spec_encoder(x_spec)

        # Dual Pooling
        spec_global, spec_focus = self.dual_pooling(spec_feat, coord_channel)

        # Cross Attention (EEG attends to Spec)
        attn_vec = self.cross_attn(eeg_seq, spec_feat)

        # Fusion
        # Concatenate all vectors
        combined = torch.cat([eeg_vec, attn_vec, spec_global, spec_focus], dim=1)

        # Classification
        logits = self.classifier(combined)

        # Return logits (Softmax applied in loss or submission)
        return logits

    def predict(self, x_eeg, x_spec):
        """Helper for inference returning probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_eeg, x_spec)
            probs = F.softmax(logits, dim=1)
        return probs
