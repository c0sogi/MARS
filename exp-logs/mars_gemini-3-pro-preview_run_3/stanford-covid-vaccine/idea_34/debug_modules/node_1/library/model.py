import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import config


class StructuralInteractionLayer(nn.Module):
    """
    Implements the Context-Aware Structural Interaction Module.

    Logic:
    1. Gather neighbor states h_j based on pair_indices.
    2. Zero-mask h_j if the base is unpaired.
    3. Construct context c_ij = [h_i; h_j].
    4. Compute message m_ij and gate g_ij from context.
    5. Update: h_res = h_i + g_ij * m_ij.
    6. Apply LayerNorm.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # Context is concatenation of self and neighbor: [h_i; h_j] -> 2 * hidden_dim
        self.msg_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gate_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices, pair_masks):
        """
        Args:
            x: Hidden states (Batch, Seq, HiddenDim)
            pair_indices: Indices of paired bases (Batch, Seq).
                          Unpaired bases point to index 0 (handled by mask).
            pair_masks: Mask indicating if base is paired (Batch, Seq).
                        1.0 for paired, 0.0 for unpaired.
        """
        B, L, D = x.shape

        # 1. Gather neighbor states
        # Expand indices to (B, L, D) to gather across the feature dimension
        idx = pair_indices.unsqueeze(-1).expand(-1, -1, D)
        # Gather along sequence dimension (dim=1)
        neighbor_x = torch.gather(x, dim=1, index=idx)

        # 2. Zero-masking for unpaired bases
        # If unpaired, pair_indices pointed to 0, so neighbor_x has x[0].
        # We multiply by 0.0 to force it to be a zero vector.
        mask = pair_masks.unsqueeze(-1)  # (B, L, 1)
        neighbor_x = neighbor_x * mask

        # 3. Context Construction
        # Concatenate current state and neighbor state
        context = torch.cat([x, neighbor_x], dim=-1)  # (B, L, 2*D)

        # 4. Context-Aware Message
        msg = F.gelu(self.msg_proj(context))

        # 5. Channel-Wise Gating
        gate = torch.sigmoid(self.gate_proj(context))

        # 6. Residual Injection
        h_res = x + gate * msg

        # 7. Post-Normalization
        out = self.norm(h_res)

        return out


class DCASGBiGRU(nn.Module):
    """
    Deep Context-Aware Structural-Gated BiGRU Architecture.

    Structure:
    1. 1D Convolutional Stem (Input -> Embedding)
    2. 4-Layer Backbone:
       - Layers 1-3: BiGRU -> StructuralInteraction -> Dropout
       - Layer 4: BiGRU -> Dropout
    3. Linear Output Head
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters from config
        self.input_dim = config.INPUT_DIM
        self.cnn_filters = config.CNN_FILTERS
        self.kernel_size = config.KERNEL_SIZE
        self.gru_hidden = config.HIDDEN_DIM
        self.num_layers = config.NUM_LAYERS
        self.dropout_rate = config.DROPOUT
        self.num_targets = config.NUM_TARGETS

        # 1. Convolutional Stem
        self.stem_conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.cnn_filters,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.stem_act = nn.GELU()

        # 2. Deep Stabilized Backbone
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # BiGRU is bidirectional, so output dimension is 2 * hidden_size
        gru_out_dim = 2 * self.gru_hidden

        for i in range(self.num_layers):
            # First layer input comes from CNN (cnn_filters)
            # Subsequent layers input comes from previous BiGRU (gru_out_dim)
            in_dim = self.cnn_filters if i == 0 else gru_out_dim

            gru = nn.GRU(
                input_size=in_dim,
                hidden_size=self.gru_hidden,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(gru)

            # Add Interaction Layer to all blocks EXCEPT the final one
            if i < self.num_layers - 1:
                self.interaction_layers.append(StructuralInteractionLayer(gru_out_dim))

            self.dropouts.append(nn.Dropout(self.dropout_rate))

        # 3. Output Head
        self.head = nn.Linear(gru_out_dim, self.num_targets)

    def forward(self, inputs, pair_indices, pair_masks):
        """
        Args:
            inputs: (Batch, Seq, InputDim)
            pair_indices: (Batch, Seq)
            pair_masks: (Batch, Seq)
        """
        # Permute for Conv1d: (N, L, C) -> (N, C, L)
        x = inputs.transpose(1, 2)

        # Stem
        x = self.stem_conv(x)
        x = self.stem_act(x)

        # Permute back for RNN: (N, C, L) -> (N, L, C)
        x = x.transpose(1, 2)

        # Backbone
        for i in range(self.num_layers):
            # BiGRU
            x, _ = self.gru_layers[i](x)

            # Structural Interaction (if applicable for this layer)
            if i < len(self.interaction_layers):
                x = self.interaction_layers[i](x, pair_indices, pair_masks)

            # Dropout
            x = self.dropouts[i](x)

        # Output Head
        out = self.head(x)

        return out
