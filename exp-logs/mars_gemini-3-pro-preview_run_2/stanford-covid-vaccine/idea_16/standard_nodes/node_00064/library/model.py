import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseDilatedBlock(nn.Module):
    """
    Single-Layer Dilated Residual Block with Dense Connectivity context.
    Note: The dense concatenation happens outside this block in the main model loop.
    This block simply transforms the accumulated features.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(DenseDilatedBlock, self).__init__()
        # Calculate padding to maintain sequence length: (kernel_size - 1) * dilation // 2
        # Assumes odd kernel_size
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ScaleAlignedDenseNet(nn.Module):
    """
    Scale-Aligned Dense-Context Hybrid Network (Idea 16).

    Features:
    1. Dense Dilated TCN Backbone: Captures multi-scale structural features.
    2. Scale-Aligned Latent Gather: Fuses full feature history of paired bases.
    3. BiGRU: Global aggregation.
    """

    def __init__(self):
        super(ScaleAlignedDenseNet, self).__init__()

        # Configuration
        self.input_channels = Config.INPUT_CHANNELS
        self.growth_rate = Config.GROWTH_RATE
        self.dilations = Config.DILATIONS
        self.kernel_size = Config.KERNEL_SIZE
        self.dropout = Config.DROPOUT
        self.latent_dim = Config.LATENT_DIM
        self.rnn_hidden_dim = Config.RNN_HIDDEN_DIM
        self.num_targets = Config.NUM_TARGETS

        # 1. Dense Dilated TCN Backbone
        self.blocks = nn.ModuleList()

        current_in_channels = self.input_channels

        for dilation in self.dilations:
            block = DenseDilatedBlock(
                in_channels=current_in_channels,
                out_channels=self.growth_rate,
                kernel_size=self.kernel_size,
                dilation=dilation,
                dropout=self.dropout,
            )
            self.blocks.append(block)
            # In DenseNet, the input to the next layer is the concat of all previous
            current_in_channels += self.growth_rate

        # Total channels in the dense history (Input + All Block Outputs)
        self.total_dense_channels = current_in_channels

        # 2. Latent Projection (Bottleneck) - Cite Lesson 00062
        self.latent_proj = nn.Conv1d(
            in_channels=self.total_dense_channels,
            out_channels=self.latent_dim,
            kernel_size=1,
        )

        # 3. Global Aggregation (BiGRU)
        # Input is Latent (Local) + Latent (Partner)
        self.gru = nn.GRU(
            input_size=self.latent_dim * 2,
            hidden_size=self.rnn_hidden_dim,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # 4. Output Head
        # BiGRU output is 2 * hidden_size
        self.head = nn.Linear(self.rnn_hidden_dim * 2, self.num_targets)

    def forward(self, x, partner_indices):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Channels, Seq_Len).
            partner_indices (torch.Tensor): Indices of paired bases of shape (Batch, Seq_Len).
                                            -1 indicates unpaired.

        Returns:
            torch.Tensor: Predictions of shape (Batch, Seq_Len, Num_Targets).
        """
        # --- 1. Dense Backbone ---
        # Keep track of all feature maps for dense concatenation
        features = [x]

        for block in self.blocks:
            # Concatenate all previous features
            dense_input = torch.cat(features, dim=1)
            # Pass through block
            out = block(dense_input)
            # Store output
            features.append(out)

        # Create the full dense history tensor
        # Shape: (Batch, Total_Dense_Channels, Seq_Len)
        h_dense = torch.cat(features, dim=1)

        # --- 2. Latent Gather (Cite Lesson 00062) ---
        # Project high-dimensional features to latent space BEFORE gathering
        h_latent = self.latent_proj(h_dense)

        batch_size, channels, seq_len = h_latent.shape

        # Handle partner indices
        valid_indices = partner_indices.clone()
        unpaired_mask = valid_indices == -1
        valid_indices[unpaired_mask] = 0

        # Expand indices for gathering across channels
        idx_expanded = valid_indices.unsqueeze(1).expand(-1, channels, -1)

        # Gather partner features from latent space
        h_partner = torch.gather(h_latent, dim=2, index=idx_expanded)

        # Apply mask
        mask = (~unpaired_mask).float().unsqueeze(1).to(x.device)
        h_partner = h_partner * mask

        # Concatenate Local and Partner Latent Features
        # Shape: (Batch, Latent_Dim * 2, Seq_Len)
        h_fused = torch.cat([h_latent, h_partner], dim=1)

        # --- 3. Global Aggregation (RNN) ---
        # Permute for RNN: (Batch, Seq_Len, Channels)
        h_rnn_in = h_fused.permute(0, 2, 1)

        out_rnn, _ = self.gru(h_rnn_in)

        # --- 4. Output Head ---
        logits = self.head(out_rnn)

        return logits
