import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseBlock(nn.Module):
    """
    Single-Layer Dilated Residual Block with Dense Connections.
    """

    def __init__(self, in_channels, growth_rate, dilation):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        # Pre-activation: BN -> ReLU
        out = F.relu(self.bn(x))
        # Convolution
        out = self.conv(out)
        # Regularization
        out = self.dropout(out)
        # Dense Connection: Concatenate input and output
        return torch.cat([x, out], dim=1)


class SRDN(nn.Module):
    """
    Stabilized Recurrent Dense Network (SRDN).
    Features:
    - Dilated Dense Convolutional Backbone
    - Symmetric Partner Gathering (Structure-aware)
    - Bidirectional GRU for Global Context
    """

    def __init__(self):
        super().__init__()
        # Input channels:
        # Sequence(4) + Structure(3) + Loop(7) + PartnerId(5) + Recycling(5) = 24
        self.in_channels = Config.NUM_INPUT_CHANNELS

        # Initial 1x1 Conv (Stem)
        self.stem = nn.Conv1d(self.in_channels, Config.GROWTH_RATE, kernel_size=1)

        # Dense Blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        curr_channels = Config.GROWTH_RATE

        for d in Config.DILATIONS:
            blk = DenseBlock(curr_channels, Config.GROWTH_RATE, d)
            self.blocks.append(blk)
            curr_channels += Config.GROWTH_RATE

        # Projection to Latent Space (Compression before RNN)
        self.to_latent = nn.Conv1d(curr_channels, Config.LATENT_DIM, kernel_size=1)

        # Global Aggregation (BiGRU)
        # Input: Latent(128) + PartnerLatent(128) = 256
        self.gru = nn.GRU(
            input_size=Config.LATENT_DIM * 2,
            hidden_size=Config.LATENT_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Output Head
        # BiGRU output is 128 * 2 = 256 (concatenated directions)
        self.head = nn.Linear(Config.LATENT_DIM * 2, Config.NUM_TARGETS)

    def forward(self, x, partner_indices):
        """
        Args:
            x: Input tensor of shape (B, L, C)
            partner_indices: Tensor of shape (B, L) containing indices of paired bases (-1 for unpaired)
        """
        # Permute to (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)

        # Backbone
        feat = self.stem(x)
        for block in self.blocks:
            feat = block(feat)

        # Project to latent space
        latent = self.to_latent(feat)  # (B, 128, L)

        # Permute back to (B, L, 128) for Gather and RNN
        latent = latent.permute(0, 2, 1)
        B, L, C = latent.shape

        # Symmetric Gather: Retrieve features of the paired base
        # 1. Clamp -1 indices to 0 to prevent gather errors (we will mask them later)
        p_idx_clamped = partner_indices.clone()
        p_idx_clamped[p_idx_clamped == -1] = 0

        # 2. Expand indices to match latent feature dimension: (B, L, C)
        gather_idx = p_idx_clamped.unsqueeze(-1).expand(-1, -1, C)

        # 3. Gather partner features
        partner_feat = torch.gather(latent, 1, gather_idx)

        # 4. Mask features for unpaired bases (where original index was -1)
        mask = (partner_indices != -1).unsqueeze(-1).float()
        partner_feat = partner_feat * mask

        # Feature Fusion: Concatenate local and partner features
        fused = torch.cat([latent, partner_feat], dim=2)  # (B, L, 256)

        # Global Aggregation
        gru_out, _ = self.gru(fused)  # (B, L, 256)

        # Prediction Head
        out = self.head(gru_out)  # (B, L, 5)

        return out
