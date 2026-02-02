import torch
import torch.nn as nn
from library.config import Config
from library.layers import StructuralInteractionLayer


class SDBR_BiGRU(nn.Module):
    """
    Stabilized Decoupled Bias-Refined BiGRU (SDBR-BiGRU) Architecture.

    Structure:
    1. Convolutional Stem: 1D Conv (k=3) -> GELU
    2. Backbone (3 Blocks):
       - Block 1: BiGRU -> StructuralInteractionLayer
       - Block 2: BiGRU -> StructuralInteractionLayer
       - Block 3: BiGRU
    3. Head: Linear Projection to Targets
    """

    def __init__(self):
        super(SDBR_BiGRU, self).__init__()

        # ==========================================
        # 1. Convolutional Stem
        # ==========================================
        # Projects sparse one-hot inputs (14 channels) to dense embedding space (256)
        # Preserves sequence length via padding
        self.conv_stem = nn.Sequential(
            nn.Conv1d(
                in_channels=Config.INPUT_DIM,
                out_channels=Config.CONV_FILTERS,
                kernel_size=Config.CONV_KERNEL_SIZE,
                padding=Config.CONV_KERNEL_SIZE // 2,
            ),
            nn.GELU(),
        )

        # ==========================================
        # 2. Backbone
        # ==========================================
        # Hidden dimension for GRU is 384. Bidirectional output is 384 * 2 = 768.
        gru_hidden = Config.HIDDEN_DIM
        gru_out_dim = gru_hidden * 2

        # Block 1
        # Input: 256 (from Conv) -> Output: 768
        self.gru1 = nn.GRU(
            input_size=Config.CONV_FILTERS,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.sdim1 = StructuralInteractionLayer(
            hidden_dim=gru_out_dim, dropout=Config.DROPOUT
        )

        # Block 2
        # Input: 768 -> Output: 768
        self.gru2 = nn.GRU(
            input_size=gru_out_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.sdim2 = StructuralInteractionLayer(
            hidden_dim=gru_out_dim, dropout=Config.DROPOUT
        )

        # Block 3 (Final Block - No SDIM per strategy)
        # Input: 768 -> Output: 768
        self.gru3 = nn.GRU(
            input_size=gru_out_dim,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # ==========================================
        # 3. Output Head
        # ==========================================
        self.head = nn.Linear(gru_out_dim, Config.NUM_TARGETS)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Seq_Len, 14)
            pair_indices (torch.Tensor): Structural pair indices (Batch, Seq_Len)
            pair_mask (torch.Tensor): Structural pair mask (Batch, Seq_Len)

        Returns:
            torch.Tensor: Predictions (Batch, Seq_Len, 5)
        """
        # 1. Convolutional Stem
        # Permute to (Batch, Channels, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv_stem(x)
        # Permute back to (Batch, Seq_Len, Channels)
        x = x.permute(0, 2, 1)

        # 2. Backbone

        # Block 1
        x, _ = self.gru1(x)
        x = self.sdim1(x, pair_indices, pair_mask)

        # Block 2
        x, _ = self.gru2(x)
        x = self.sdim2(x, pair_indices, pair_mask)

        # Block 3
        x, _ = self.gru3(x)
        # No SDIM in the final block

        # 3. Head
        logits = self.head(x)

        return logits
