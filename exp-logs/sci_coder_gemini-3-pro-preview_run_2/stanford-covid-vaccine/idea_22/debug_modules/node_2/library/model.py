import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseBlock(nn.Module):
    """
    Single-Layer Dilated Convolution Block.
    Used within a DenseNet topology where inputs are concatenated history.
    """

    def __init__(self, in_channels, out_channels, dilation, dropout):
        super(DenseBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class DecoupledDenseNet(nn.Module):
    """
    Decoupled Multi-Scale Dense Network (Idea 22).

    Features:
    1. Dense Dilated TCN Backbone.
    2. Decoupled Local/Global Latent Representations.
    3. Independent Compression.
    4. Partner-Aware Feature Gathering.
    5. BiGRU Aggregation.
    """

    def __init__(self):
        super(DecoupledDenseNet, self).__init__()

        # Hyperparameters
        self.input_dim = Config.INPUT_DIM
        self.channel_width = Config.CHANNEL_WIDTH
        self.dilations = Config.DILATIONS
        self.dropout_rate = Config.DROPOUT
        self.latent_dim = Config.LATENT_DIM
        self.rnn_hidden = Config.RNN_HIDDEN_DIM
        self.num_targets = Config.NUM_TARGETS

        # 1. Input Stem
        # Projects raw features (18) to channel width (64)
        self.stem = nn.Conv1d(self.input_dim, self.channel_width, kernel_size=1)

        # 2. Dense Dilated Backbone
        self.blocks = nn.ModuleList()
        current_input_dim = self.channel_width

        for dilation in self.dilations:
            block = DenseBlock(
                in_channels=current_input_dim,
                out_channels=self.channel_width,
                dilation=dilation,
                dropout=self.dropout_rate,
            )
            self.blocks.append(block)
            # In DenseNet, the input to the next layer grows by the growth rate (channel_width)
            current_input_dim += self.channel_width

        # 3. Decoupled Compression
        # We assume the dilations are split evenly into Local (low d) and Global (high d)
        # Local: d=1, 2, 4 (First 3 blocks)
        # Global: d=8, 16, 32 (Last 3 blocks)
        num_local_blocks = 3
        num_global_blocks = 3

        # Input to compression is the concatenation of the block outputs
        # Each block outputs 'channel_width' channels
        local_in_channels = num_local_blocks * self.channel_width
        global_in_channels = num_global_blocks * self.channel_width

        self.local_compress = nn.Conv1d(
            local_in_channels, self.latent_dim, kernel_size=1
        )
        self.global_compress = nn.Conv1d(
            global_in_channels, self.latent_dim, kernel_size=1
        )

        # 4. Global Aggregation (BiGRU)
        # Input: Self_Local + Self_Global + Partner_Local + Partner_Global
        rnn_input_dim = 4 * self.latent_dim

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=self.rnn_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # 5. Output Head
        # BiGRU outputs 2 * hidden_size
        self.head = nn.Linear(self.rnn_hidden * 2, self.num_targets)

    def forward(self, x, partner_indices):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Seq_Len, Input_Dim).
            partner_indices (torch.Tensor): Indices of paired bases (Batch, Seq_Len).

        Returns:
            torch.Tensor: Predictions (Batch, Seq_Len, Num_Targets).
        """
        # Permute to (Batch, Channels, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)

        # Stem
        stem_out = self.stem(x)

        # Backbone Pass
        # We maintain a list of all feature maps for dense concatenation
        all_features = [stem_out]
        block_outputs = []

        for block in self.blocks:
            # Concatenate all prior features
            in_feat = torch.cat(all_features, dim=1)

            # Forward block
            out_feat = block(in_feat)

            # Store
            all_features.append(out_feat)
            block_outputs.append(out_feat)

        # Decouple Views
        # block_outputs contains [Out_L0, Out_L1, ..., Out_L5]
        # Local: Indices 0, 1, 2
        # Global: Indices 3, 4, 5
        local_features = torch.cat(block_outputs[:3], dim=1)
        global_features = torch.cat(block_outputs[3:], dim=1)

        # Compress
        z_local = self.local_compress(local_features)  # (B, Latent, L)
        z_global = self.global_compress(global_features)  # (B, Latent, L)

        # Gather Partner Features
        # partner_indices is (B, L). We need to gather along L dimension (dim 2).
        # Expand indices to match channel dimension: (B, Latent, L)
        B, C, L = z_local.shape
        p_idx_expanded = partner_indices.unsqueeze(1).expand(-1, C, -1)

        # Gather
        p_local = torch.gather(z_local, 2, p_idx_expanded)
        p_global = torch.gather(z_global, 2, p_idx_expanded)

        # Fusion
        # Concatenate: [Self_Local, Self_Global, Partner_Local, Partner_Global]
        # Shape: (B, 4*Latent, L)
        fused = torch.cat([z_local, z_global, p_local, p_global], dim=1)

        # RNN Aggregation
        # Permute back to (B, L, C) for RNN
        fused = fused.permute(0, 2, 1)

        rnn_out, _ = self.rnn(fused)

        # Output Projection
        out = self.head(rnn_out)

        return out
