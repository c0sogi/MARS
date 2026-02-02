import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedConvBlock(nn.Module):
    """
    Single-Layer Dilated Residual Block.
    Consists of Conv1d -> BatchNorm -> ReLU -> Dropout.
    Includes a residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        # Calculate padding to maintain sequence length (same padding)
        # padding = (L_out - 1)*stride + 1 + (kernel_size-1)*dilation - L_in
        # For stride=1, L_out=L_in: padding = (kernel_size-1)*dilation
        # We use padding on both sides, so half of that.
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)

        # Residual connection
        if x.shape[1] == out.shape[1]:
            return x + out
        else:
            return out


class LatentStructureRefinement(nn.Module):
    """
    Latent Structural Refinement Module.
    Gathers features from paired bases using the partner index map,
    concatenates them with local features, and projects back to the original dimension.
    """

    def __init__(self, channels):
        super().__init__()
        # Input: Local features (C) + Partner features (C) = 2*C
        # Output: Projected back to C
        self.projection = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x, partner_indices):
        """
        Args:
            x (torch.Tensor): Feature map of shape (Batch, Channels, SeqLen).
            partner_indices (torch.Tensor): Indices of paired bases of shape (Batch, SeqLen).
                                            Unpaired bases are denoted by -1.
        """
        B, C, L = x.shape

        # 1. Handle unpaired indices (-1)
        # Create a mask for valid pairs (where index != -1)
        valid_mask = partner_indices != -1  # (B, L)

        # Replace -1 with 0 to allow torch.gather to work without error.
        # We will mask the result later, so the value gathered at index 0 doesn't matter.
        safe_indices = partner_indices.clone()
        safe_indices[~valid_mask] = 0

        # 2. Expand indices for gathering
        # We gather along the sequence dimension (dim 2).
        # Indices tensor must match the dimensions of x: (B, C, L)
        safe_indices = safe_indices.unsqueeze(1).expand(B, C, L)

        # 3. Gather partner features
        # x_gathered[b, c, i] = x[b, c, safe_indices[b, c, i]]
        partner_features = torch.gather(x, 2, safe_indices)

        # 4. Zero out features for unpaired bases
        # Expand mask to (B, C, L)
        mask_expanded = valid_mask.unsqueeze(1).expand(B, C, L)
        partner_features = partner_features * mask_expanded.float()

        # 5. Concatenate and Project
        # Concatenate along channel dimension
        combined = torch.cat([x, partner_features], dim=1)

        out = self.projection(combined)
        out = self.bn(out)
        out = self.act(out)

        return out


class HybridNet(nn.Module):
    """
    Latent Structure-Refined Hybrid Network.

    Architecture:
    1. Input Stem (18 -> 128 channels)
    2. Multi-Scale Dilated TCN Backbone (Concatenation of all layer outputs)
    3. Latent Structure Refinement (Partner-aware message passing)
    4. BiGRU (Global aggregation)
    5. Linear Head (Prediction)
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters from Config
        input_dim = Config.INPUT_CHANNELS
        num_filters = Config.NUM_FILTERS
        kernel_size = Config.KERNEL_SIZE
        dilations = Config.DILATION_RATES
        dropout = Config.DROPOUT
        rnn_hidden = Config.RNN_HIDDEN_DIM
        rnn_layers = Config.RNN_LAYERS
        bidirectional = Config.RNN_BIDIRECTIONAL
        num_targets = Config.NUM_TARGETS

        # 1. Stem
        self.stem = nn.Conv1d(input_dim, num_filters, kernel_size=1)

        # 2. Backbone: Stack of Dilated Blocks
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(
                DilatedConvBlock(num_filters, num_filters, kernel_size, d, dropout)
            )

        # Multi-Scale Fusion Dimension
        # We concatenate the outputs of all dilated blocks.
        # Each block outputs `num_filters` channels.
        fusion_dim = num_filters * len(dilations)

        # 3. Latent Structure Refinement
        self.refinement = LatentStructureRefinement(fusion_dim)

        # 4. BiGRU
        # Input size is the refined feature dimension (same as fusion_dim)
        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if rnn_layers > 1 else 0,
        )

        # 5. Output Head
        # BiGRU output dimension is hidden_size * 2 (if bidirectional)
        gru_out_dim = rnn_hidden * 2 if bidirectional else rnn_hidden
        self.head = nn.Linear(gru_out_dim, num_targets)

    def forward(self, inputs, partner_indices):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (Batch, Channels, SeqLen).
            partner_indices (torch.Tensor): Partner index map of shape (Batch, SeqLen).

        Returns:
            torch.Tensor: Predictions of shape (Batch, SeqLen, NumTargets).
        """
        # 1. Stem
        x = self.stem(inputs)  # (B, 128, L)

        # 2. Backbone with Multi-Scale Fusion
        block_outputs = []
        current_x = x
        for block in self.blocks:
            current_x = block(current_x)
            block_outputs.append(current_x)

        # Concatenate outputs from all blocks along channel dimension
        # Shape: (B, 128 * 5, L) -> (B, 640, L)
        fused = torch.cat(block_outputs, dim=1)

        # 3. Latent Structure Refinement
        # Refines features using the partner map
        refined = self.refinement(fused, partner_indices)

        # 4. BiGRU
        # RNN expects (Batch, SeqLen, InputSize)
        rnn_input = refined.permute(0, 2, 1)
        rnn_out, _ = self.gru(rnn_input)

        # 5. Head
        logits = self.head(rnn_out)

        return logits
