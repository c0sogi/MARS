import torch
import torch.nn as nn
from library.layers import StabilizedGLUInteraction


class HighCapacityBiGRU(nn.Module):
    """
    High-Capacity Stabilized GLU-Decoupled BiGRU Architecture.

    This model implements a deep hybrid architecture consisting of:
    1. A Convolutional Stem for local feature extraction.
    2. A 4-Layer Bidirectional GRU backbone for high-capacity sequence modeling.
    3. Interleaved Stabilized GLU-Decoupled Interaction Modules to inject
       secondary structure information without disrupting gradient flow.
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration object containing hyperparameters.
        """
        super().__init__()
        self.config = config

        # ----------------------------------------------------------------
        # 1. Convolutional Stem
        # ----------------------------------------------------------------
        # Projects sparse one-hot inputs (14 channels) to dense embeddings (256 channels).
        # Preserves sequence length L.
        self.conv = nn.Conv1d(
            in_channels=config.INPUT_CHANNELS,
            out_channels=config.CONV_FILTERS,
            kernel_size=3,
            padding=1,
        )
        self.act = nn.GELU()

        # ----------------------------------------------------------------
        # 2. High-Capacity Backbone (4 Layers)
        # ----------------------------------------------------------------
        self.blocks = nn.ModuleList()

        gru_input_dim = config.CONV_FILTERS  # 256
        gru_hidden = config.GRU_HIDDEN  # 384
        gru_out_dim = gru_hidden * 2  # 768 (Bidirectional)

        for i in range(config.NUM_LAYERS):
            # The first GRU layer takes the Conv output.
            # Subsequent layers take the output of the previous GRU (768 dim).
            input_size = gru_input_dim if i == 0 else gru_out_dim

            # Bidirectional GRU
            gru = nn.GRU(
                input_size=input_size,
                hidden_size=gru_hidden,
                batch_first=True,
                bidirectional=True,
            )

            # Structural Interaction Module
            # Applied to all blocks EXCEPT the final one to refine representations
            # before the final abstraction layer.
            interaction = None
            if i < config.NUM_LAYERS - 1:
                interaction = StabilizedGLUInteraction(gru_out_dim)

            self.blocks.append(nn.ModuleDict({"gru": gru, "interaction": interaction}))

        # Dropout for regularization between blocks
        self.dropout = nn.Dropout(config.DROPOUT)

        # ----------------------------------------------------------------
        # 3. Output Head
        # ----------------------------------------------------------------
        # Projects the final hidden state to the 5 target variables.
        self.head = nn.Linear(gru_out_dim, 5)

    def forward(self, x, adj, mask):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input features. Shape (Batch, Length, 14).
            adj (torch.Tensor): Adjacency indices for pairing. Shape (Batch, Length).
            mask (torch.Tensor): Pairing mask. Shape (Batch, Length).

        Returns:
            torch.Tensor: Predictions. Shape (Batch, Length, 5).
        """
        # 1. Stem
        # Permute to (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.act(x)
        # Permute back to (B, L, C)
        x = x.permute(0, 2, 1)

        h = x

        # 2. Backbone
        for i, block in enumerate(self.blocks):
            # GRU Forward
            # h shape: (B, L, Input_Dim) -> (B, L, Hidden*2)
            h, _ = block["gru"](h)

            # Interaction (if present)
            if block["interaction"] is not None:
                h = block["interaction"](h, adj, mask)

            # Dropout
            # Applied between blocks (i.e., not after the very last block's processing)
            if i < len(self.blocks) - 1:
                h = self.dropout(h)

        # 3. Head
        out = self.head(h)  # (B, L, 5)

        return out
