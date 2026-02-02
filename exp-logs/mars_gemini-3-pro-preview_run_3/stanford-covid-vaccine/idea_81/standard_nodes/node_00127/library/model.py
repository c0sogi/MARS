import torch
import torch.nn as nn
from library.config import Config
from library.layers import VerticalResBiGRU, StabilizedGLUInteraction


class DeepResBiGRU(nn.Module):
    """
    Deep Residual High-Capacity BiGRU with Multi-Layer Feature Aggregation.

    Architecture:
    1. Conv1d Stem (Local Context Aggregation)
    2. 4-Layer Backbone:
       - Vertical Residual BiGRU (Deep Sequence Modeling)
       - Stabilized GLU-Decoupled Interaction (Structural Injection)
    3. Multi-Layer Feature Aggregation (MLFA) Head
    """

    def __init__(self):
        super().__init__()

        # =================================================================
        # 1. Convolutional Stem
        # =================================================================
        # Projects input features (14) to dense embedding (256)
        # Aggregates local context via kernel size 3
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=Config.INPUT_CHANNELS,
                out_channels=Config.STEM_FILTERS,
                kernel_size=Config.STEM_KERNEL_SIZE,
                padding=Config.STEM_KERNEL_SIZE // 2,
            ),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
        )

        # =================================================================
        # 2. Deep Residual Backbone
        # =================================================================
        self.layers = nn.ModuleList()

        # BiGRU hidden size is defined per direction in Config, so total is * 2
        self.total_hidden_dim = Config.HIDDEN_DIM * 2

        # Input dimension for the first layer comes from the stem
        current_input_dim = Config.STEM_FILTERS

        for _ in range(Config.N_LAYERS):
            # A. Vertical Residual BiGRU
            # The first layer handles the dimension projection (256 -> 768)
            # Subsequent layers use residual connections (768 -> 768)
            gru_block = VerticalResBiGRU(
                input_dim=current_input_dim,
                hidden_dim=self.total_hidden_dim,
                dropout=Config.DROPOUT,
            )

            # B. Stabilized GLU-Decoupled Interaction
            # Injects structural information into the sequence stream
            interaction_block = StabilizedGLUInteraction(
                hidden_dim=self.total_hidden_dim, dropout=Config.DROPOUT
            )

            self.layers.append(nn.ModuleList([gru_block, interaction_block]))

            # Update input dimension for the next layer
            current_input_dim = self.total_hidden_dim

        # =================================================================
        # 3. Output Head (Multi-Layer Feature Aggregation)
        # =================================================================
        # Concatenates outputs from all layers to capture multi-scale features
        self.head_input_dim = self.total_hidden_dim * Config.N_LAYERS

        self.head = nn.Linear(self.head_input_dim, Config.NUM_TARGETS)

    def forward(self, x, adjacency, pair_mask):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input features (Batch, Seq_Len, Channels).
            adjacency (torch.Tensor): Adjacency indices (Batch, Seq_Len).
            pair_mask (torch.Tensor): Pair mask (Batch, Seq_Len).

        Returns:
            torch.Tensor: Predictions (Batch, Seq_Len, Num_Targets).
        """
        # ----------------------------------------------------------------
        # Stem
        # ----------------------------------------------------------------
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        # Permute back: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # ----------------------------------------------------------------
        # Backbone
        # ----------------------------------------------------------------
        layer_outputs = []

        for gru, interaction in self.layers:
            # Apply Vertical Residual BiGRU
            x = gru(x)

            # Apply Structural Interaction
            x = interaction(x, adjacency, pair_mask)

            # Store output for MLFA
            layer_outputs.append(x)

        # ----------------------------------------------------------------
        # Head (MLFA)
        # ----------------------------------------------------------------
        # Concatenate all layer outputs along the feature dimension
        # Shape: (Batch, Seq_Len, N_Layers * Hidden_Dim)
        combined_features = torch.cat(layer_outputs, dim=-1)

        # Final Projection
        out = self.head(combined_features)

        return out
