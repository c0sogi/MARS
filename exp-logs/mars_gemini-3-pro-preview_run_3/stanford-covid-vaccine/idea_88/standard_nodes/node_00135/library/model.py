import torch
import torch.nn as nn
from library.config import Config
from library.layers import StabilizedGLUInteraction


class HighCapacityBiGRU(nn.Module):
    """
    High-Capacity Stabilized GLU-Decoupled BiGRU Model.

    Architecture:
    1. 1D Convolutional Stem (Projection & Local Aggregation).
    2. Deep Backbone (4 Layers):
       - Layers 1-3: Bidirectional GRU -> Stabilized GLU-Decoupled Interaction.
       - Layer 4: Bidirectional GRU (No interaction, feeds into head).
    3. Linear Output Head.

    This architecture prioritizes backbone capacity (768 dim) and stabilizes
    structural message passing using the Decoupled GLU mechanism.
    """

    def __init__(self):
        super(HighCapacityBiGRU, self).__init__()

        # =====================================================================
        # Configuration
        # =====================================================================
        self.input_channels = Config.INPUT_CHANNELS
        self.cnn_filters = Config.CNN_FILTERS
        self.kernel_size = Config.KERNEL_SIZE

        # GRU Settings
        self.hidden_dim = Config.HIDDEN_DIM  # 384 per direction
        self.bidirectional = True
        self.gru_output_dim = (
            self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        )  # 768
        self.num_layers = Config.NUM_LAYERS  # 4
        self.dropout = Config.DROPOUT

        # =====================================================================
        # 1. Convolutional Stem
        # =====================================================================
        # Projects sparse one-hot inputs (14 channels) to dense embedding (256 channels)
        # Preserves sequence length via padding.
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.cnn_filters,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.GELU(),
        )

        # =====================================================================
        # 2. High-Capacity Backbone
        # =====================================================================
        # We manually stack layers to interleave the Interaction Module.
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()

        # Input dimension for the first GRU is the CNN output size
        current_input_dim = self.cnn_filters

        for i in range(self.num_layers):
            # BiGRU Layer
            gru = nn.GRU(
                input_size=current_input_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
            self.gru_layers.append(gru)

            # Interaction Module
            # Applied after every block EXCEPT the final one.
            # If num_layers=4, we apply interaction at indices 0, 1, 2.
            if i < self.num_layers - 1:
                interaction = StabilizedGLUInteraction(
                    hidden_dim=self.gru_output_dim, dropout=self.dropout
                )
                self.interaction_layers.append(interaction)

            # Update input dimension for the next layer (output of current BiGRU)
            current_input_dim = self.gru_output_dim

        # =====================================================================
        # 3. Output Head
        # =====================================================================
        self.head = nn.Linear(self.gru_output_dim, Config.NUM_TARGETS)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, 14).
            pair_indices (torch.Tensor): Structural indices (Batch, Seq_Len).
            pair_mask (torch.Tensor): Structural mask (Batch, Seq_Len).

        Returns:
            torch.Tensor: Predictions of shape (Batch, Seq_Len, 5).
        """
        # ---------------------------------------------------------------------
        # Stem
        # ---------------------------------------------------------------------
        # Conv1d expects (Batch, Channels, Seq_Len)
        x = x.transpose(1, 2)
        x = self.stem(x)
        # Transpose back to (Batch, Seq_Len, Channels) for RNN
        x = x.transpose(1, 2)

        # ---------------------------------------------------------------------
        # Backbone
        # ---------------------------------------------------------------------
        for i in range(self.num_layers):
            # Apply GRU
            # GRU returns (output, h_n). We only need output.
            x, _ = self.gru_layers[i](x)

            # Apply Interaction if this layer has one
            if i < len(self.interaction_layers):
                x = self.interaction_layers[i](x, pair_indices, pair_mask)

        # ---------------------------------------------------------------------
        # Head
        # ---------------------------------------------------------------------
        out = self.head(x)

        return out
