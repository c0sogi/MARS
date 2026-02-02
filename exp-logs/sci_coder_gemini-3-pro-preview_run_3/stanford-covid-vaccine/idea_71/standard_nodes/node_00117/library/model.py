import torch
import torch.nn as nn
from library.layers import DecoupledGLUInteraction, PointwiseFFN


class HighCapacityAugmentedBiGRU(nn.Module):
    """
    High-Capacity FFN-Augmented Decoupled BiGRU Model.

    Implements the strategy of augmenting a deep BiGRU backbone with:
    1. Decoupled GLU-Structural Interaction modules for spatial structural injection.
    2. Pointwise Feed-Forward Networks (FFNs) for increased non-linear processing depth.

    Architecture:
    - Input: (Batch, Seq, 14)
    - Stem: Conv1d -> GELU
    - Backbone: 4 Blocks of [BiGRU -> Interaction -> FFN]
    - Head: Linear projection to targets
    """

    def __init__(self, config):
        super().__init__()

        # Configuration
        self.input_channels = config.INPUT_CHANNELS
        self.conv_filters = config.CONV_FILTERS
        self.conv_kernel = config.CONV_KERNEL
        self.hidden_dim = config.HIDDEN_DIM  # Dimension per direction (384)
        self.total_hidden_dim = self.hidden_dim * 2  # Bidirectional total (768)
        self.num_layers = config.NUM_LAYERS
        self.dropout = config.DROPOUT
        self.num_targets = config.NUM_TARGETS

        # 1. Convolutional Stem
        # Projects sparse inputs (14 channels) to dense embedding space (256 channels)
        self.stem_conv = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=self.conv_filters,
            kernel_size=self.conv_kernel,
            padding=self.conv_kernel // 2,
        )
        self.stem_act = nn.GELU()

        # 2. Backbone Layers
        self.layers = nn.ModuleList()

        # Input dimension for the first GRU is the Conv output size
        current_input_dim = self.conv_filters

        for _ in range(self.num_layers):
            # A. BiGRU Layer
            # Captures complex sequential dependencies
            gru = nn.GRU(
                input_size=current_input_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
                bidirectional=True,
            )

            # B. Decoupled GLU-Structural Interaction
            # Spatially integrates structural constraints via GLU messages and gating.
            # Operates on the full hidden dimension (768).
            interaction = DecoupledGLUInteraction(
                hidden_dim=self.total_hidden_dim, dropout=self.dropout
            )

            # C. Pointwise Feed-Forward Network (FFN)
            # Adds non-linear processing capacity to digest structural updates.
            ffn = PointwiseFFN(hidden_dim=self.total_hidden_dim, dropout=self.dropout)

            self.layers.append(nn.ModuleList([gru, interaction, ffn]))

            # For subsequent layers, the input is the output of the previous block (768)
            current_input_dim = self.total_hidden_dim

        # 3. Output Head
        self.head = nn.Linear(self.total_hidden_dim, self.num_targets)

    def forward(self, sequence, bpp_indices, pair_mask):
        """
        Args:
            sequence (torch.Tensor): Input sequence features (Batch, Seq, 14).
            bpp_indices (torch.Tensor): Structural indices for gather ops (Batch, Seq).
            pair_mask (torch.Tensor): Structural mask (Batch, Seq).

        Returns:
            torch.Tensor: Predictions (Batch, Seq, 5).
        """
        # Permute for Conv1d: (Batch, Channels, Seq)
        x = sequence.permute(0, 2, 1)

        # Stem
        x = self.stem_conv(x)
        x = self.stem_act(x)

        # Permute back: (Batch, Seq, Channels)
        x = x.permute(0, 2, 1)

        # Backbone Processing (Sandwich Structure)
        for gru, interaction, ffn in self.layers:
            # 1. BiGRU
            # x shape: (Batch, Seq, Input_Dim) -> (Batch, Seq, Total_Hidden_Dim)
            x, _ = gru(x)

            # 2. Structural Interaction
            # Injects structural info into the sequence features
            x = interaction(x, bpp_indices, pair_mask)

            # 3. FFN
            # Refines features before next layer
            x = ffn(x)

        # Output Head
        logits = self.head(x)

        return logits
