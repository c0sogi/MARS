import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import VerticalResidualBiGRU, UnifiedGLUInteraction


class DeepResGLUBiGRU(nn.Module):
    """
    Deep Residual High-Capacity GLU-BiGRU Model.

    Architecture:
    1. Convolutional Stem: 1D Conv -> GELU -> Dropout.
    2. Backbone: 4 Layers of VerticalResidualBiGRU interleaved with UnifiedGLUInteraction.
       - Layer 1 expands dim from Stem to Hidden*2.
       - Layers 2-4 maintain dim and use vertical residuals.
    3. Head: Linear projection to targets.
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Convolutional Stem
        # =====================================================================
        # Input: (Batch, Seq_Len, Input_Dim) -> Needs permutation for Conv1d
        self.stem_conv = nn.Conv1d(
            in_channels=Config.INPUT_DIM,
            out_channels=Config.CONV_FILTERS,
            kernel_size=Config.CONV_KERNEL,
            padding=Config.CONV_KERNEL // 2,  # Padding 1 for Kernel 3 preserves length
        )
        self.stem_dropout = nn.Dropout(Config.DROPOUT)

        # =====================================================================
        # 2. Deep Residual Backbone
        # =====================================================================
        self.layers = nn.ModuleList()

        # Dimensions
        # Stem output: 256
        # Backbone hidden (bidirectional): 384 * 2 = 768
        current_input_dim = Config.CONV_FILTERS
        rnn_hidden_dim = Config.HIDDEN_DIM
        backbone_dim = rnn_hidden_dim * 2

        for i in range(Config.NUM_LAYERS):
            # A. Vertical Residual BiGRU
            # Layer 1: 256 -> 768 (No residual due to dim mismatch)
            # Layer 2-4: 768 -> 768 (Residual active)
            gru_layer = VerticalResidualBiGRU(
                input_dim=current_input_dim,
                hidden_dim=rnn_hidden_dim,
                dropout=Config.DROPOUT,
            )

            # B. Unified GLU-Decoupled Interaction Module
            # Operates on the output of the GRU (backbone_dim)
            interaction_layer = UnifiedGLUInteraction(hidden_dim=backbone_dim)

            # Group into a block
            self.layers.append(nn.ModuleList([gru_layer, interaction_layer]))

            # Update input dimension for the next layer
            current_input_dim = backbone_dim

        # =====================================================================
        # 3. Output Head
        # =====================================================================
        self.head = nn.Linear(backbone_dim, Config.NUM_TARGETS)

    def forward(self, x, pair_indices):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Seq_Len, Input_Dim).
            pair_indices (torch.Tensor): Structural indices (Batch, Seq_Len).

        Returns:
            torch.Tensor: Predictions (Batch, Seq_Len, Num_Targets).
        """
        # --- Stem ---
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)

        x = self.stem_conv(x)
        x = F.gelu(x)
        x = self.stem_dropout(x)

        # Permute back for RNN: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)

        # --- Backbone ---
        for gru_layer, interaction_layer in self.layers:
            # 1. Recurrent Processing (with optional residual)
            x = gru_layer(x)

            # 2. Structural Interaction
            x = interaction_layer(x, pair_indices)

        # --- Head ---
        out = self.head(x)

        return out
