import torch
import torch.nn as nn
from library.config import Config
from library.layers import ConvStem, DecoupledStructuralInteraction


class DeepStabilizedBiGRU(nn.Module):
    """
    Deep Stabilized Bias-Refined Decoupled BiGRU Architecture.

    Implements the strategy:
    1. Input Representation: (N, 107, 14)
    2. Convolutional Stem: Projects to embedding space (256 dim).
    3. Deep Stabilized Backbone: 4 Layers.
       - Layers 0-2: BiGRU -> DecoupledStructuralInteraction.
       - Layer 3: BiGRU -> LayerNorm (Interaction skipped for final block).
    4. Output Head: Linear projection to 5 targets.
    """

    def __init__(self):
        super(DeepStabilizedBiGRU, self).__init__()

        # 1. Convolutional Stem
        # Projects sparse inputs (14 channels) to dense embedding (256 channels)
        self.stem = ConvStem(
            in_channels=Config.NUM_FEATURES,
            out_channels=Config.CONV_FILTERS,
            kernel_size=Config.KERNEL_SIZE,
        )

        # 2. Deep Stabilized Backbone
        self.layers = nn.ModuleList()
        self.interactions = nn.ModuleList()

        # Explicit final normalization for the last block if interaction is skipped
        self.final_norm = nn.LayerNorm(Config.HIDDEN_DIM)

        for i in range(Config.NUM_LAYERS):
            # Determine input dimension:
            # First layer takes Stem output (256), others take Hidden dim (384)
            input_dim = Config.CONV_FILTERS if i == 0 else Config.HIDDEN_DIM

            # BiGRU Layer
            # hidden_size is halved because bidirectional=True concatenates outputs
            # Output shape: (N, L, HIDDEN_DIM)
            gru = nn.GRU(
                input_size=input_dim,
                hidden_size=Config.HIDDEN_DIM // 2,
                bidirectional=True,
                batch_first=True,
            )
            self.layers.append(gru)

            # Structural Interaction Module
            # "The backbone consists of 4 Blocks... followed by the Structural Interaction Module (except the final block)."
            if i < Config.NUM_LAYERS - 1:
                interaction = DecoupledStructuralInteraction(
                    hidden_dim=Config.HIDDEN_DIM, dropout=Config.DROPOUT
                )
                self.interactions.append(interaction)
            else:
                # Placeholder for the last layer to maintain index alignment
                self.interactions.append(None)

        # 3. Output Head
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS)

    def forward(self, features, adjacency):
        """
        Forward pass of the model.

        Args:
            features (Tensor): Input features of shape (N, L, 14).
            adjacency (Tensor): Adjacency indices of shape (N, L).

        Returns:
            Tensor: Predictions of shape (N, L, 5).
        """
        # Pass through Stem
        x = self.stem(features)

        # Pass through Backbone
        for i, gru in enumerate(self.layers):
            # Apply BiGRU
            # GRU returns (output, h_n), we only use the sequence output
            x, _ = gru(x)

            # Apply Interaction or Final Stabilization
            interaction = self.interactions[i]
            if interaction is not None:
                # Layers 0 to N-2: Apply Decoupled Structural Interaction
                x = interaction(x, adjacency)
            elif i == Config.NUM_LAYERS - 1:
                # Layer N-1: Apply Final Norm since Interaction (and its post-norm) is skipped
                x = self.final_norm(x)

        # Pass through Output Head
        logits = self.head(x)

        return logits
