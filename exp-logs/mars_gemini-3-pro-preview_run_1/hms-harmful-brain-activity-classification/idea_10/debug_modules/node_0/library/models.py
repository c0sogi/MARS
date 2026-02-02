import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class Inception1DBlock(nn.Module):
    """
    A single Inception-style block for 1D EEG processing.
    Consists of parallel convolutional branches with different kernel sizes,
    followed by concatenation, normalization, activation, and pooling.
    """

    def __init__(
        self, in_channels, out_channels_per_branch, kernel_sizes=[3, 5, 7], pool_size=4
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels,
                    out_channels_per_branch,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in kernel_sizes
            ]
        )

        total_out_channels = out_channels_per_branch * len(kernel_sizes)
        self.bn = nn.BatchNorm1d(total_out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        # x: (B, C_in, T)
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)  # (B, C_total, T)
        out = self.bn(out)
        out = self.act(out)
        out = self.pool(out)  # (B, C_total, T // pool_size)
        return out


class EEGEncoder(nn.Module):
    """
    Multi-Scale 1D CNN for Raw EEG Waveforms.
    """

    def __init__(self, config=Config):
        super().__init__()
        self.in_channels = config.EEG_CHANNELS
        self.kernel_sizes = config.EEG_KERNEL_SIZES
        self.filters = config.EEG_FILTERS  # e.g. [64, 128, 256]

        layers = []
        current_channels = self.in_channels

        # Build blocks based on config
        for f in self.filters:
            block = Inception1DBlock(
                in_channels=current_channels,
                out_channels_per_branch=f,
                kernel_sizes=self.kernel_sizes,
                pool_size=4,
            )
            layers.append(block)
            # Update current channels: filters * num_branches
            current_channels = f * len(self.kernel_sizes)

        self.encoder = nn.Sequential(*layers)
        self.out_dim = current_channels

    def forward(self, x):
        # x: (B, 20, 5000)
        return self.encoder(x)  # (B, out_dim, T_reduced)


class CoordSpecEncoder(nn.Module):
    """
    EfficientNet backbone with modified input for Coordinate Injection.
    Input: (B, 5, H, W) -> 4 Spec Channels + 1 Coord Channel.
    """

    def __init__(self, config=Config, pretrained=True):
        super().__init__()
        # Use timm to handle in_chans adaptation (3 -> 5)
        # It automatically recycles weights from the first 3 channels
        self.backbone = timm.create_model(
            config.BACKBONE,
            pretrained=pretrained,
            in_chans=config.TOTAL_INPUT_CHANNELS,
            num_classes=0,  # Remove classifier
            global_pool="",  # Remove pooling, keep spatial features
        )

        # Determine output dimension dynamically
        dummy_input = torch.randn(1, config.TOTAL_INPUT_CHANNELS, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        self.out_dim = features.shape[1]

    def forward(self, x):
        # x: (B, 5, H, W)
        x = self.backbone(x)  # (B, C, H', W')
        return x


class DualStreamModel(nn.Module):
    """
    Coordinate-Injected Dual-Stream Network.
    Fuses EEG (Query) and Spectrogram (Key/Value) via Cross-Attention.
    """

    def __init__(self, config=Config, pretrained=True):
        super().__init__()
        self.num_classes = config.NUM_CLASSES
        self.fusion_dim = config.FUSION_DIM

        # --- Encoders ---
        self.eeg_encoder = EEGEncoder(config)
        self.spec_encoder = CoordSpecEncoder(config, pretrained=pretrained)

        # --- Projections ---
        # Project EEG features to fusion dimension
        self.eeg_proj = nn.Conv1d(
            self.eeg_encoder.out_dim, self.fusion_dim, kernel_size=1
        )

        # Project Spec features to fusion dimension
        self.spec_proj = nn.Conv2d(
            self.spec_encoder.out_dim, self.fusion_dim, kernel_size=1
        )

        # --- Fusion (Cross Attention) ---
        # Query: EEG, Key/Value: Spec
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=4,
            batch_first=True,
            dropout=config.DROPOUT_RATE,
        )

        self.layer_norm = nn.LayerNorm(self.fusion_dim)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)

        # --- Classifier ---
        self.classifier = nn.Linear(self.fusion_dim, self.num_classes)

    def forward(self, eeg, spec):
        """
        Args:
            eeg: (B, 20, 5000)
            spec: (B, 5, 512, 512)
        Returns:
            logits: (B, 6)
        """
        # 1. Encode Streams
        # EEG -> (B, C_eeg, T_eeg)
        eeg_feat = self.eeg_encoder(eeg)

        # Spec -> (B, C_spec, H_spec, W_spec)
        spec_feat = self.spec_encoder(spec)

        # 2. Project to common dimension
        q = self.eeg_proj(eeg_feat)  # (B, dim, T_eeg)
        k = self.spec_proj(spec_feat)  # (B, dim, H, W)

        # 3. Prepare for Attention
        # Permute Query to (B, T_eeg, dim)
        q = q.permute(0, 2, 1)

        # Flatten and Permute Key/Value to (B, H*W, dim)
        B, C, H, W = k.shape
        k = k.view(B, C, -1).permute(0, 2, 1)
        v = k  # Value is same as Key (projected spec features)

        # 4. Cross Attention
        # attn_output: (B, T_eeg, dim)
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)

        # Residual + Norm
        x = self.layer_norm(q + attn_out)

        # 5. Global Pooling (over time dimension of EEG)
        x = torch.mean(x, dim=1)  # (B, dim)

        # 6. Classification
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits
