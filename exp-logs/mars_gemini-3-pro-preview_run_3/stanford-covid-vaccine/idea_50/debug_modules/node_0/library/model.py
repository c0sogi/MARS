import torch
import torch.nn as nn
from library.config import Config
from library.layers import StructuralInteractionModule


class SDBR_BiGRU(nn.Module):
    """
    Stabilized Decoupled Bias-Refined BiGRU (SDBR-BiGRU) Architecture.

    Structure:
    1. 1D Convolutional Stem (Embedding & Local Aggregation)
    2. 3-Layer Backbone:
       - Bidirectional GRU
       - Decoupled Structural Interaction Module (SIM)
    3. Linear Output Head
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters from Config
        self.input_dim = Config.INPUT_DIM
        self.stem_filters = Config.STEM_FILTERS
        self.kernel_size = Config.KERNEL_SIZE
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.dropout = Config.DROPOUT
        self.num_targets = Config.NUM_TARGETS

        # ----------------------------------------------------------------
        # 1. Convolutional Stem
        # ----------------------------------------------------------------
        # Projects sparse one-hot inputs (14) to dense embeddings (256).
        # Padding ensures output length equals input length.
        padding = (self.kernel_size - 1) // 2
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=self.stem_filters,
                kernel_size=self.kernel_size,
                padding=padding,
            ),
            nn.GELU(),
        )

        # ----------------------------------------------------------------
        # 2. Stabilized Backbone
        # ----------------------------------------------------------------
        self.blocks = nn.ModuleList()

        # Track input dimension for the stacked layers
        # Layer 1 Input: Stem Filters (256)
        # Layer 2+ Input: BiGRU Output (Hidden * 2 = 768)
        current_input_dim = self.stem_filters
        gru_hidden_dim = self.hidden_dim

        for i in range(self.num_layers):
            # Bidirectional GRU
            # Output shape: (Batch, Seq_Len, Hidden_Dim * 2)
            gru = nn.GRU(
                input_size=current_input_dim,
                hidden_size=gru_hidden_dim,
                batch_first=True,
                bidirectional=True,
            )

            # Structural Interaction Module
            # Operates on the BiGRU output size (768)
            # Maintains the same dimension
            sim_dim = gru_hidden_dim * 2
            sim = StructuralInteractionModule(sim_dim, dropout=self.dropout)

            # Add block components
            self.blocks.append(nn.ModuleList([gru, sim]))

            # Update input dimension for the next layer
            current_input_dim = sim_dim

        # ----------------------------------------------------------------
        # 3. Output Head
        # ----------------------------------------------------------------
        # Projects final refined states to 5 targets
        self.head = nn.Linear(current_input_dim, self.num_targets)

    def forward(self, features, pair_indices, pair_masks):
        """
        Args:
            features: (Batch, Seq_Len, 14) - One-hot encoded inputs
            pair_indices: (Batch, Seq_Len) - Indices of paired bases
            pair_masks: (Batch, Seq_Len) - 1.0 if paired, 0.0 if unpaired

        Returns:
            logits: (Batch, Seq_Len, 5) - Predicted values
        """
        # --- Stem ---
        # Conv1d expects (Batch, Channels, Length)
        x = features.transpose(1, 2)
        x = self.stem(x)
        # Transpose back to (Batch, Length, Channels) for GRU
        x = x.transpose(1, 2)

        # --- Backbone ---
        for gru, sim in self.blocks:
            # GRU Forward
            # gru returns (output, h_n). We use the output sequence.
            x, _ = gru(x)

            # Interaction Module Forward
            # Refines the sequence using structural context
            x = sim(x, pair_indices, pair_masks)

        # --- Head ---
        logits = self.head(x)

        return logits
