import torch
import torch.nn as nn
from library.config import Config
from library.layers import DeepResidualStem, StabilizedGLUInteraction


class HighCapacityRNAnet(nn.Module):
    """
    High-Capacity Hierarchical Synthesis Model.

    This architecture implements a deep hybrid model for RNA degradation prediction,
    featuring a Deep Residual Convolutional Stem, a High-Capacity BiGRU backbone,
    and Stabilized GLU-Decoupled Interaction modules.

    Architecture:
    1. Deep Residual Convolutional Stem (Input -> Hidden)
    2. 4-Layer Backbone:
       - BiGRU (High Capacity: 384 hidden per direction)
       - Stabilized GLU-Decoupled Interaction
       - Dropout
    3. Linear Output Head (Hidden -> 5 Targets)
    """

    def __init__(self):
        super(HighCapacityRNAnet, self).__init__()

        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.dropout_rate = Config.DROPOUT

        # 1. Deep Residual Stem
        # Projects 14 channels -> 768 channels via Conv/ResBlocks.
        # Explicitly models local k-mer motifs and secondary structure patterns
        # before temporal processing.
        self.stem = DeepResidualStem(
            input_dim=Config.INPUT_DIM,
            hidden_dim=self.hidden_dim,
            kernel_sizes=Config.STEM_KERNEL_SIZES,
        )

        # 2. Backbone Blocks
        # The backbone consists of 4 Blocks processed sequentially (Strict Hierarchy).
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            # Each block contains:
            # - BiGRU
            # - Interaction Module
            # - Dropout

            # BiGRU Layer
            # Hidden dimension of 384 per direction = 768 Total.
            # We explicitly maintain this high capacity to capture complex folding landscapes.
            gru = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim // 2,  # 384
                bidirectional=True,
                batch_first=True,
            )

            # Stabilized GLU-Decoupled Interaction Module
            # Synthesizes Decoupled Gating, GLU Messages, and Bias-Driven Refinement.
            interaction = StabilizedGLUInteraction(hidden_dim=self.hidden_dim)

            # Regularization
            dropout = nn.Dropout(self.dropout_rate)

            # Store the triplet as a ModuleList for easier iteration in forward
            self.blocks.append(nn.ModuleList([gru, interaction, dropout]))

        # 3. Output Head
        # Linear projection to 5 target values.
        self.head = nn.Linear(self.hidden_dim, 5)

    def forward(self, inputs, pair_indices, pair_masks):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input features of shape (Batch, Length, 14).
            pair_indices (torch.Tensor): Structural pair indices of shape (Batch, Length).
            pair_masks (torch.Tensor): Structural pair masks of shape (Batch, Length).

        Returns:
            torch.Tensor: Predicted degradation rates of shape (Batch, Length, 5).
        """
        # 1. Stem Processing
        # (Batch, Length, 14) -> (Batch, Length, 768)
        x = self.stem(inputs)

        # 2. Backbone Processing
        for gru, interaction, dropout in self.blocks:
            # BiGRU
            # Output shape: (Batch, Length, 2 * 384) -> (Batch, Length, 768)
            # We ignore the final hidden state (_)
            x, _ = gru(x)

            # Stabilized GLU-Decoupled Interaction
            # Injects structural context using the pair indices and masks.
            # Performs gather, zero-masking, GLU messaging, and residual injection.
            x = interaction(x, pair_indices, pair_masks)

            # Dropout
            x = dropout(x)

        # 3. Output Head
        # (Batch, Length, 768) -> (Batch, Length, 5)
        out = self.head(x)

        return out
