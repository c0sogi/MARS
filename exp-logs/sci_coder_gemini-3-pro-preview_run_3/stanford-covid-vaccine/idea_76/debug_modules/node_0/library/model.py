import torch
import torch.nn as nn
from library.config import Config
from library.layers import DisentangledInteraction


class HCTDBiGRU(nn.Module):
    """
    High-Capacity Topology-Disentangled BiGRU (HC-TD-BiGRU) Model.

    Architecture:
    1. Convolutional Stem: Projects sparse one-hot inputs to dense embeddings.
    2. Backbone: 4 Blocks of Bidirectional GRUs.
       - Blocks 1-3: BiGRU -> Topology-Disentangled Interaction Module.
       - Block 4: BiGRU only (no interaction, as per strategy).
    3. Head: Linear projection to targets.
    """

    def __init__(self):
        super(HCTDBiGRU, self).__init__()

        # =====================================================================
        # Configuration
        # =====================================================================
        self.input_dim = Config.INPUT_DIM
        self.stem_filters = Config.STEM_FILTERS
        self.stem_kernel = Config.STEM_KERNEL_SIZE
        self.hidden_dim = Config.HIDDEN_DIM  # 384 per direction
        self.num_layers = Config.NUM_LAYERS  # 4
        self.dropout_rate = Config.DROPOUT
        self.num_targets = Config.NUM_TARGETS

        # =====================================================================
        # 1. Convolutional Stem
        # =====================================================================
        # Padding to maintain sequence length: (k - 1) / 2
        padding = (self.stem_kernel - 1) // 2
        self.stem = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.stem_filters,
            kernel_size=self.stem_kernel,
            padding=padding,
        )
        self.stem_act = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_rate)

        # =====================================================================
        # 2. Backbone (BiGRU + Interaction)
        # =====================================================================
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()

        # The output dimension of the BiGRU is 2 * hidden_dim
        gru_output_dim = self.hidden_dim * 2

        for i in range(self.num_layers):
            # First layer takes stem output, subsequent layers take previous GRU output
            # Note: Interaction module maintains dimension, so input to next GRU is gru_output_dim
            input_size = self.stem_filters if i == 0 else gru_output_dim

            # High-Capacity BiGRU
            self.gru_layers.append(
                nn.GRU(
                    input_size=input_size,
                    hidden_size=self.hidden_dim,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Interleave Interaction Module (except for the final block)
            if i < self.num_layers - 1:
                self.interaction_layers.append(
                    DisentangledInteraction(
                        hidden_dim=gru_output_dim, dropout=self.dropout_rate
                    )
                )

        # =====================================================================
        # 3. Output Head
        # =====================================================================
        self.head = nn.Linear(gru_output_dim, self.num_targets)

    def forward(self, inputs, pair_indices, pair_mask):
        """
        Forward pass of the HC-TD-BiGRU.

        Args:
            inputs (torch.Tensor): (Batch, Seq_Len, Input_Dim)
            pair_indices (torch.Tensor): (Batch, Seq_Len) - Indices for paired bases.
            pair_mask (torch.Tensor): (Batch, Seq_Len) - 1.0 if paired, 0.0 if unpaired.

        Returns:
            torch.Tensor: (Batch, Seq_Len, Num_Targets)
        """
        # ---------------------------------------------------------------------
        # 1. Stem
        # ---------------------------------------------------------------------
        # Conv1d expects (Batch, Channels, Seq_Len)
        x = inputs.transpose(1, 2)
        x = self.stem(x)
        x = self.stem_act(x)
        x = x.transpose(1, 2)  # Back to (Batch, Seq_Len, Channels)
        x = self.dropout(x)

        # ---------------------------------------------------------------------
        # 2. Backbone
        # ---------------------------------------------------------------------
        for i, gru in enumerate(self.gru_layers):
            # BiGRU Pass
            # Output shape: (Batch, Seq_Len, 2 * Hidden_Dim)
            x, _ = gru(x)

            # Topology-Disentangled Interaction
            # Applied to all blocks except the last one
            if i < len(self.interaction_layers):
                x = self.interaction_layers[i](x, pair_indices, pair_mask)

        # ---------------------------------------------------------------------
        # 3. Head
        # ---------------------------------------------------------------------
        out = self.head(x)

        return out
