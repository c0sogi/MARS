import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseBlock(nn.Module):
    """
    A single unit of the DenseNet backbone.
    Performs a dilated convolution to generate new features (growth_rate).
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout=0.0):
        super(DenseBlock, self).__init__()

        # Padding calculation to maintain sequence length
        # padding = dilation * (kernel_size - 1) / 2
        padding = (dilation * (kernel_size - 1)) // 2

        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(growth_rate)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        return out


class DenseBackbone(nn.Module):
    """
    Manages the dense connectivity pattern.
    Iteratively applies DenseBlocks, concatenating inputs at each step.
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation_schedule):
        super(DenseBackbone, self).__init__()
        self.blocks = nn.ModuleList()

        current_channels = in_channels

        for dilation in dilation_schedule:
            block = DenseBlock(
                in_channels=current_channels,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            self.blocks.append(block)
            current_channels += growth_rate

        self.out_channels = current_channels

    def forward(self, x):
        # x shape: (Batch, In_Channels, Length)
        features = [x]

        for block in self.blocks:
            # Concatenate all previous features along channel dimension
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Final output is the concatenation of all features
        return torch.cat(features, dim=1)


class LatentRefinement(nn.Module):
    """
    Compresses dense features and performs the Latent Gather operation
    to incorporate structural context from paired bases.
    """

    def __init__(self, in_channels, bottleneck_dim):
        super(LatentRefinement, self).__init__()

        # 1x1 Convolution for compression
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(bottleneck_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, partner_indices):
        """
        Args:
            x: (Batch, In_Channels, Length)
            partner_indices: (Batch, Length) - Indices of paired bases
        """
        # 1. Compress
        # Shape: (Batch, Bottleneck_Dim, Length)
        local_features = self.bottleneck(x)
        local_features = self.bn(local_features)
        local_features = self.act(local_features)

        B, C, L = local_features.shape

        # 2. Latent Gather
        # We need to gather features from partner_indices.
        # partner_indices has shape (B, L). Values are in [0, L-1].

        # Expand indices to match channel dimension for gather
        # Shape: (B, C, L)
        indices_expanded = partner_indices.unsqueeze(1).expand(-1, C, -1)

        # Gather along the length dimension (dim=2)
        # gathered_features[b, c, i] = local_features[b, c, indices_expanded[b, c, i]]
        partner_features = torch.gather(local_features, dim=2, index=indices_expanded)

        # 3. Concatenate Local + Partner
        # Shape: (Batch, Bottleneck_Dim * 2, Length)
        out = torch.cat([local_features, partner_features], dim=1)

        return out


class DenseContextNet(nn.Module):
    """
    The Dense-Context Latent-Refined Hybrid Network (Idea 9).

    Architecture:
    1. Input (One-Hot + PartnerID)
    2. Dense Dilated Backbone (Global Receptive Field via DenseNet)
    3. Latent Refinement (Bottleneck + Partner Gather)
    4. BiGRU (Global Aggregation)
    5. Linear Head
    """

    def __init__(self):
        super(DenseContextNet, self).__init__()

        # --- Configs ---
        input_channels = Config.INPUT_CHANNELS
        growth_rate = Config.GROWTH_RATE
        kernel_size = Config.KERNEL_SIZE
        dilation_schedule = Config.DILATION_SCHEDULE
        bottleneck_dim = Config.BOTTLENECK_DIM
        rnn_hidden = Config.RNN_HIDDEN_DIM
        rnn_layers = Config.RNN_LAYERS
        num_targets = Config.NUM_TARGETS

        # --- Modules ---

        # 1. Dense Backbone
        self.backbone = DenseBackbone(
            in_channels=input_channels,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            dilation_schedule=dilation_schedule,
        )

        # 2. Latent Refinement
        self.refinement = LatentRefinement(
            in_channels=self.backbone.out_channels, bottleneck_dim=bottleneck_dim
        )

        # 3. Global Aggregator (BiGRU)
        # Input to RNN is Bottleneck * 2 (Local + Partner)
        rnn_input_dim = bottleneck_dim * 2
        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )

        # 4. Output Head
        # BiGRU output is hidden_size * 2
        self.head = nn.Linear(rnn_hidden * 2, num_targets)

    def forward(self, x, partner_indices):
        """
        Args:
            x: (Batch, Length, Input_Channels) - Note: Input is usually Channel-Last in loaders
            partner_indices: (Batch, Length)

        Returns:
            (Batch, Length, Num_Targets)
        """
        # Permute to Channel-First for Conv1d: (B, C, L)
        x = x.permute(0, 2, 1)

        # 1. Backbone
        # Output: (B, Dense_Channels, L)
        features = self.backbone(x)

        # 2. Refinement
        # Output: (B, Bottleneck*2, L)
        refined = self.refinement(features, partner_indices)

        # 3. RNN
        # Permute back to Channel-Last for RNN: (B, L, C)
        refined = refined.permute(0, 2, 1)
        rnn_out, _ = self.gru(refined)

        # 4. Head
        # Output: (B, L, Num_Targets)
        out = self.head(rnn_out)

        return out
