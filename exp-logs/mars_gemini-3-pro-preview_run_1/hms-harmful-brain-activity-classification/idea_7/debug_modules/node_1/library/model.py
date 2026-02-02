import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from library.config import Config


class InceptionBlock1D(nn.Module):
    """
    A 1D Inception Block that applies parallel convolutions with different kernel sizes.
    """

    def __init__(self, in_channels, out_channels, kernels, reduction=4):
        super().__init__()
        self.branches = nn.ModuleList()

        # Determine channels per branch
        # We distribute the output channels evenly across the branches
        branch_channels = out_channels // len(kernels)

        for k in kernels:
            pad = (k - 1) // 2
            branch = nn.Sequential(
                # Bottleneck / Projection
                nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(),
                # Main Convolution
                nn.Conv1d(
                    branch_channels,
                    branch_channels,
                    kernel_size=k,
                    padding=pad,
                    groups=1,
                    bias=False,
                ),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(),
            )
            self.branches.append(branch)

        # Final projection to mix branch outputs and restore full channel depth
        self.projection = nn.Sequential(
            nn.Conv1d(
                branch_channels * len(kernels), out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        x = torch.cat(outputs, dim=1)
        x = self.projection(x)
        return x


class EEGEncoder(nn.Module):
    """
    Encoder for raw EEG data using stacked Inception Blocks.
    """

    def __init__(self, config):
        super().__init__()
        self.kernels = config.EEG_KERNELS
        self.filters = config.EEG_FILTERS
        in_channels = config.EEG_CHANNELS

        layers = []
        for out_channels in self.filters:
            layers.append(InceptionBlock1D(in_channels, out_channels, self.kernels))
            # Pooling to reduce temporal dimension (5000 -> small sequence)
            layers.append(nn.MaxPool1d(kernel_size=4, stride=4))
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)
        self.out_dim = self.filters[-1]

    def forward(self, x):
        # x: (Batch, Channels, Time)
        x = self.encoder(x)
        # Global Average Pooling to get a single vector per sample
        x = torch.mean(x, dim=2)
        return x


class SpectrogramEncoder(nn.Module):
    """
    Encoder for Spectrograms using EfficientNet-B2.
    """

    def __init__(self, config):
        super().__init__()
        # Load backbone, remove classifier, keep spatial features
        # features_only=True returns a list of feature maps
        self.backbone = timm.create_model(
            config.SPEC_BACKBONE,
            pretrained=config.SPEC_PRETRAINED,
            features_only=True,
            out_indices=(4,),  # Get the last feature map (most abstract)
        )

        # Determine output dimensions dynamically
        dummy = torch.randn(1, 3, config.SPEC_IMG_SIZE[0], config.SPEC_IMG_SIZE[1])
        with torch.no_grad():
            feats = self.backbone(dummy)
            out_feat = feats[0]
            self.out_channels = out_feat.shape[1]
            self.time_steps = out_feat.shape[2]  # Height corresponds to Time
            self.freq_bins = out_feat.shape[3]  # Width corresponds to Freq

        # Project flattened frequency features to attention dimension
        self.proj = nn.Linear(self.out_channels * self.freq_bins, config.ATTENTION_DIM)

    def forward(self, x):
        # x: (Batch, 3, Height/Time, Width/Freq)
        feats = self.backbone(x)[0]  # (Batch, C, H, W)

        # Permute to (Batch, Time, Freq, Channels)
        # We treat Height as Time based on data processing logic
        x = feats.permute(0, 2, 3, 1)  # (B, H, W, C)

        B, H, W, C = x.shape

        # Flatten Frequency and Channels: (Batch, Time, Features)
        x = x.reshape(B, H, W * C)

        # Project to attention dimension
        x = self.proj(x)  # (Batch, Time, AttnDim)

        return x


class GuidedAttention(nn.Module):
    """
    Cross-Attention module with Gaussian Bias for temporal alignment.
    """

    def __init__(self, dim, sigma=0.1):
        super().__init__()
        self.dim = dim
        self.sigma = sigma

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.scale = dim**-0.5

    def forward(self, query, key_value, guidance):
        """
        Args:
            query: (Batch, Dim) - EEG embedding
            key_value: (Batch, Time, Dim) - Spectrogram sequence
            guidance: (Batch,) - Normalized offset [0, 1]
        """
        B, T, D = key_value.shape

        # Project Q, K, V
        Q = self.q_proj(query).unsqueeze(1)  # (B, 1, D)
        K = self.k_proj(key_value)  # (B, T, D)
        V = self.v_proj(key_value)  # (B, T, D)

        # Calculate raw attention scores
        # (B, 1, D) @ (B, D, T) -> (B, 1, T)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # --- Gaussian Bias Mask ---
        # Create temporal grid [0, 1]
        grid = torch.linspace(0, 1, T, device=key_value.device).view(1, 1, T)

        # Reshape guidance to (B, 1, 1)
        mu = guidance.view(B, 1, 1)

        # Calculate Gaussian bias: - (t - mu)^2 / (2 * sigma^2)
        # This penalizes time steps far from the guidance offset
        # We assume guidance is normalized [0,1] matching the grid
        bias = -((grid - mu) ** 2) / (2 * self.sigma**2)

        # Add bias to logits
        attn_logits = attn_logits + bias

        # Softmax
        attn_weights = F.softmax(attn_logits, dim=-1)

        # Aggregate Values
        # (B, 1, T) @ (B, T, D) -> (B, 1, D)
        context = torch.matmul(attn_weights, V)
        context = context.squeeze(1)  # (B, D)

        return context


class OffsetGuidedDualStreamModel(nn.Module):
    """
    Main Model Architecture:
    Stream A: EEG (Inception1D)
    Stream B: Spectrogram (EfficientNet)
    Fusion: Offset-Guided Cross Attention
    """

    def __init__(self, config):
        super().__init__()

        # 1. EEG Stream
        self.eeg_encoder = EEGEncoder(config)
        self.eeg_proj = nn.Linear(self.eeg_encoder.out_dim, config.ATTENTION_DIM)

        # 2. Spectrogram Stream
        self.spec_encoder = SpectrogramEncoder(config)

        # 3. Fusion (Guided Attention)
        self.attention = GuidedAttention(
            config.ATTENTION_DIM, sigma=config.ATTENTION_MASK_SIGMA
        )
        self.dropout = nn.Dropout(config.DROPOUT_RATE)

        # 4. Classifier Head
        # Concatenates EEG embedding and Context vector
        self.classifier = nn.Sequential(
            nn.Linear(config.ATTENTION_DIM * 2, config.ATTENTION_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.ATTENTION_DIM, config.NUM_CLASSES),
        )

    def forward(self, spec, eeg, guidance):
        # Process EEG
        eeg_feat = self.eeg_encoder(eeg)  # (B, EEG_Dim)
        eeg_emb = self.eeg_proj(eeg_feat)  # (B, Attn_Dim)

        # Process Spectrogram
        spec_seq = self.spec_encoder(spec)  # (B, Time, Attn_Dim)

        # Guided Attention Fusion
        # Query: EEG, Key/Value: Spec, Guidance: Offset
        context = self.attention(eeg_emb, spec_seq, guidance)  # (B, Attn_Dim)

        # Concatenate
        fused = torch.cat([eeg_emb, context], dim=1)  # (B, Attn_Dim * 2)
        fused = self.dropout(fused)

        # Classification
        logits = self.classifier(fused)

        return logits
