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


class UnifiedDenseNet(nn.Module):
    """
    Unified Dense Network.

    Features:
    1. Dense Dilated TCN Backbone.
    2. Unified Feature Aggregation (Cite solution_lesson_node_00070).
    3. Latent Bottleneck Projection (Cite solution_lesson_node_00068).
    4. Partner-Aware Feature Gathering.
    5. BiGRU Aggregation.
    """

    def __init__(self):
        super(UnifiedDenseNet, self).__init__()

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

        # 3. Unified Compression
        # Concatenate Stem + All Block Outputs
        # Total channels = channel_width * (num_blocks + 1)
        num_blocks = len(self.dilations)
        total_channels = self.channel_width * (num_blocks + 1)

        self.projection = nn.Conv1d(total_channels, self.latent_dim, kernel_size=1)

        # 4. Global Aggregation (BiGRU)
        # Input: Self_Latent + Partner_Latent
        rnn_input_dim = 2 * self.latent_dim

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

        for block in self.blocks:
            # Concatenate all prior features
            in_feat = torch.cat(all_features, dim=1)

            # Forward block
            out_feat = block(in_feat)

            # Store
            all_features.append(out_feat)

        # Unified Aggregation
        # Concatenate all features (Stem + Blocks)
        dense_history = torch.cat(all_features, dim=1)

        # Project to Latent Space
        z = self.projection(dense_history)  # (B, Latent, L)

        # Gather Partner Features
        # partner_indices is (B, L). We need to gather along L dimension (dim 2).
        # Expand indices to match channel dimension: (B, Latent, L)
        B, C, L = z.shape
        p_idx_expanded = partner_indices.unsqueeze(1).expand(-1, C, -1)

        # Gather
        p = torch.gather(z, 2, p_idx_expanded)

        # Fusion
        # Concatenate: [Self, Partner]
        # Shape: (B, 2*Latent, L)
        fused = torch.cat([z, p], dim=1)

        # RNN Aggregation
        # Permute back to (B, L, C) for RNN
        fused = fused.permute(0, 2, 1)

        rnn_out, _ = self.rnn(fused)

        # Output Projection
        out = self.head(rnn_out)

        return out
