import torch
import torch.nn as nn
from library.config import Config
from library.layers import StructuralInjectionLayer


class InterleavedBiGRU(nn.Module):
    """
    Deep BiGRU architecture with Interleaved Gated Structural Injection.

    Architecture:
    1. Convolutional Stem: Projects sparse one-hot inputs to dense embeddings.
    2. Interleaved Backbone: Stacks BiGRU layers with StructuralInjectionLayers
       placed between them. This allows the model to alternate between sequential
       processing (RNN) and spatial structural updates (Gating).
    3. Regression Head: Projects final hidden states to target values.
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.input_channels = Config.INPUT_CHANNELS
        self.conv_filters = Config.CONV_FILTERS
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.dropout_p = Config.DROPOUT
        self.num_targets = Config.NUM_TARGETS
        self.kernel_size = Config.CONV_KERNEL_SIZE

        # =====================================================================
        # 1. Convolutional Stem (Local Feature Extraction)
        # =====================================================================
        # Projects (B, 14, L) -> (B, 256, L)
        # Padding is calculated to maintain sequence length (Same padding)
        padding = self.kernel_size // 2

        self.stem_conv = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=self.conv_filters,
            kernel_size=self.kernel_size,
            padding=padding,
        )
        self.stem_act = nn.GELU()
        self.stem_dropout = nn.Dropout(self.dropout_p)

        # =====================================================================
        # 2. Interleaved Backbone (Iterative Refinement)
        # =====================================================================
        self.gru_layers = nn.ModuleList()
        self.injection_layers = nn.ModuleList()

        # BiGRU hidden size is half of HIDDEN_DIM because BiGRU concatenates
        # forward and backward states: Output Dim = 2 * (HIDDEN_DIM // 2) = HIDDEN_DIM
        gru_hidden_size = self.hidden_dim // 2

        for i in range(self.num_layers):
            # The first GRU layer takes the Conv stem output (256)
            # Subsequent layers take the output of the previous block (384)
            input_dim = self.conv_filters if i == 0 else self.hidden_dim

            gru = nn.GRU(
                input_size=input_dim,
                hidden_size=gru_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(gru)

            # Add Structural Injection Layer after every GRU except the final one
            # This corresponds to the "Interleaved" strategy
            if i < self.num_layers - 1:
                injection = StructuralInjectionLayer(
                    hidden_dim=self.hidden_dim, dropout=self.dropout_p
                )
                self.injection_layers.append(injection)

        # =====================================================================
        # 3. Output Head
        # =====================================================================
        self.head = nn.Linear(self.hidden_dim, self.num_targets)

    def forward(self, x, pair_indices):
        """
        Forward pass of the Interleaved BiGRU model.

        Args:
            x (torch.Tensor): Input features of shape (Batch, Seq_Len, Channels).
                              Channels = 14 (4 Seq + 3 Struct + 7 Loop).
            pair_indices (torch.Tensor): Structural pair indices of shape (Batch, Seq_Len).
                                         Used by the Injection Layers.

        Returns:
            torch.Tensor: Predictions of shape (Batch, Seq_Len, Num_Targets).
        """
        # ---------------------------------------------------------------------
        # Stem Processing
        # ---------------------------------------------------------------------
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)

        x = self.stem_conv(x)
        x = self.stem_act(x)

        # Permute back for RNN: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)
        x = self.stem_dropout(x)

        # ---------------------------------------------------------------------
        # Backbone Processing
        # ---------------------------------------------------------------------
        for i in range(self.num_layers):
            # 1. Sequential Context (BiGRU)
            # Output shape: (B, L, HIDDEN_DIM)
            x, _ = self.gru_layers[i](x)

            # 2. Structural Refinement (Injection)
            # Applied between GRU layers to mix spatial information
            if i < len(self.injection_layers):
                x = self.injection_layers[i](x, pair_indices)

        # ---------------------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------------------
        out = self.head(x)

        return out
