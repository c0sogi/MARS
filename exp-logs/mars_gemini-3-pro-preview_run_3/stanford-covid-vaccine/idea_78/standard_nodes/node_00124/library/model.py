import torch
import torch.nn as nn
from library.config import Config
from library.layers import ConvStem, StabilizedGLUInteraction


class DeepResidualBiGRU(nn.Module):
    """
    Deep Residual High-Capacity BiGRU with Stabilized GLU-Interaction.

    This model implements a deep 4-layer bidirectional GRU backbone designed for
    high capacity. To ensure trainability and stability, it employs:
    1. Vertical Residual Connections: Allowing gradients to bypass recurrent layers.
    2. Stabilized GLU-Decoupled Interaction: Injecting structural information
       (paired bases) via a gated mechanism with bias-driven refinement for
       unpaired loops.

    Architecture Flow:
    Input -> ConvStem -> [BiGRU -> Residual -> Interaction] x 4 -> Head
    """

    def __init__(self):
        super(DeepResidualBiGRU, self).__init__()

        # =====================================================================
        # Configuration
        # =====================================================================
        self.input_dim = Config.INPUT_DIM
        self.stem_dim = Config.STEM_FILTERS
        self.hidden_dim = Config.HIDDEN_DIM  # Dimension per direction
        self.num_layers = Config.NUM_LAYERS
        self.dropout_rate = Config.DROPOUT
        self.num_targets = Config.NUM_TARGETS

        # Architectural Flags
        self.use_residuals = Config.USE_VERTICAL_RESIDUALS
        self.use_interaction = Config.USE_GLU_INTERACTION

        # =====================================================================
        # 1. Convolutional Stem
        # =====================================================================
        self.stem = ConvStem(
            self.input_dim, self.stem_dim, kernel_size=Config.STEM_KERNEL_SIZE
        )

        # =====================================================================
        # 2. Deep Residual Backbone
        # =====================================================================
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # The output dimension of a BiGRU is hidden_dim * 2
        total_hidden_dim = self.hidden_dim * 2

        for i in range(self.num_layers):
            # Determine input size for the GRU
            # Layer 0: Input is Stem Output (256)
            # Layer >0: Input is Previous Layer Output (768)
            input_size = self.stem_dim if i == 0 else total_hidden_dim

            # Bidirectional GRU
            # We use num_layers=1 per block to manually handle residuals between blocks
            gru = nn.GRU(
                input_size=input_size,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(gru)

            # Dropout for the residual path
            self.dropouts.append(nn.Dropout(self.dropout_rate))

            # Stabilized GLU-Decoupled Interaction Module
            if self.use_interaction:
                interaction = StabilizedGLUInteraction(
                    hidden_dim=total_hidden_dim, dropout=self.dropout_rate
                )
                self.interaction_layers.append(interaction)

        # =====================================================================
        # 3. Output Head
        # =====================================================================
        self.head = nn.Linear(total_hidden_dim, self.num_targets)

    def forward(self, x, adjacency, bpp_mask):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input features. Shape (Batch, Seq_Len, 14).
            adjacency (torch.Tensor): Indices of paired bases. Shape (Batch, Seq_Len).
            bpp_mask (torch.Tensor): Mask (1.0 paired, 0.0 unpaired). Shape (Batch, Seq_Len).

        Returns:
            torch.Tensor: Predictions. Shape (Batch, Seq_Len, 5).
        """
        # 1. Stem Projection
        h = self.stem(x)  # (Batch, Seq_Len, Stem_Dim)

        # 2. Backbone Processing
        for i in range(self.num_layers):
            # A. BiGRU Layer
            # Output shape: (Batch, Seq_Len, Hidden_Dim * 2)
            gru_out, _ = self.gru_layers[i](h)

            # B. Vertical Residual Connection
            # For the first layer (i=0), dimensions change (Stem -> Hidden), so strictly feed-forward.
            # For subsequent layers (i>0), dimensions match, so we apply the residual:
            # h_l = h_{l-1} + Dropout(BiGRU(h_{l-1}))
            if i > 0 and self.use_residuals:
                gru_out = self.dropouts[i](gru_out)
                h = h + gru_out
            else:
                h = gru_out

            # C. Stabilized GLU-Decoupled Interaction
            # Injects structural context into the sequence representation
            if self.use_interaction:
                h = self.interaction_layers[i](h, adjacency, bpp_mask)

        # 3. Output Projection
        out = self.head(h)  # (Batch, Seq_Len, 5)

        return out
