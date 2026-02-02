import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralInteractionModule(nn.Module):
    """
    Decoupled Structural Interaction Module with Strict Output Masking.

    This module implements the structural injection mechanism described in the strategy:
    1. Point-to-Point Gathering of neighbor states.
    2. Decoupled Message computation (derived solely from neighbor state).
    3. Channel-Wise Gating (derived from joint context).
    4. Strict Output Masking (unpaired bases get exactly zero update).
    5. Post-LayerNorm stabilization.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message projection: derived solely from neighbor (Decoupled)
        # We do not concatenate h_i here to force the branch to learn the structural delta.
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gate projection: derived from joint context [h_i; h_j]
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, x, bpp_indices, bpp_mask):
        """
        Args:
            x: Input tensor (Batch, Seq, Hidden).
            bpp_indices: Adjacency indices (Batch, Seq).
                         indices[i] = j if paired, else i.
            bpp_mask: Binary mask (Batch, Seq, 1).
                      1.0 if paired, 0.0 if unpaired.
        """
        batch_size, seq_len, hidden_dim = x.size()

        # 1. Gather neighbor states h_j
        # Expand indices to (B, L, D) to gather across the hidden dimension
        indices_expanded = bpp_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        h_neighbor = torch.gather(x, 1, indices_expanded)

        # 2. Decoupled Message
        # m_ij = GELU(W_msg * h_j)
        msg = self.act(self.msg_proj(h_neighbor))

        # 3. Channel-Wise Gating
        # g_ij = Sigmoid(W_gate * [h_i; h_j])
        gate_input = torch.cat([x, h_neighbor], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))

        # 4. Compute Update with Strict Output Masking
        # u_ij = (g_ij * m_ij) * M_pair
        update = gate * msg

        # CRITICAL: Apply mask to the computed update.
        # This ensures unpaired bases (mask=0) receive a mathematically exact zero update,
        # preventing bias drift from the linear layers.
        update = update * bpp_mask

        update = self.dropout(update)

        # 5. Residual + Post-Norm
        # h_out = LayerNorm(h_i + u_ij)
        out = self.layer_norm(x + update)

        return out


class RNAModel(nn.Module):
    """
    Deep Decoupled Post-Norm BiGRU with Strict Output Masking.

    Architecture:
    - Input: Sequence-preserved tensor (Batch, 107, 14).
    - Stem: 1D Convolution projecting to dense embedding.
    - Backbone: 4 Layers.
        - Layers 0-2: BiGRU -> StructuralInteractionModule.
        - Layer 3: BiGRU only.
    - Head: Linear projection to targets.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Convolutional Stem
        # Projects sparse one-hot inputs (14 channels) into dense embedding space (256).
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=config.num_features,
                out_channels=config.cnn_filters,
                kernel_size=config.cnn_kernel_size,
                padding=config.cnn_kernel_size // 2,
            ),
            nn.GELU(),
        )

        self.layers = nn.ModuleList()

        # Dimensions
        input_dim = config.cnn_filters
        gru_hidden = config.hidden_dim
        # BiGRU output dimension is 2 * hidden_size
        gru_out_dim = gru_hidden * 2

        for i in range(config.num_layers):
            # BiGRU Layer
            # Note: input_dim is 256 for the first layer, 768 for subsequent layers
            gru = nn.GRU(
                input_size=input_dim,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

            # Structural Interaction Module
            # Applied to all layers except the final block, as per strategy.
            if i < config.num_layers - 1:
                interaction = StructuralInteractionModule(
                    hidden_dim=gru_out_dim, dropout=config.dropout
                )
            else:
                interaction = None

            self.layers.append(nn.ModuleDict({"gru": gru, "interaction": interaction}))

            # Next layer input is current layer output
            input_dim = gru_out_dim

        # Output Head
        self.head = nn.Linear(gru_out_dim, config.num_targets)

    def forward(self, inputs, bpp_indices, bpp_mask):
        """
        Args:
            inputs: (Batch, Seq, 14)
            bpp_indices: (Batch, Seq)
            bpp_mask: (Batch, Seq, 1)
        """
        # 1. Stem
        # Conv1d expects (Batch, Channels, Seq)
        x = inputs.transpose(1, 2)
        x = self.stem(x)
        x = x.transpose(1, 2)  # Back to (Batch, Seq, Channels)

        # 2. Deep Backbone
        for layer in self.layers:
            # BiGRU
            x, _ = layer["gru"](x)

            # Structural Interaction (if present in this block)
            if layer["interaction"] is not None:
                x = layer["interaction"](x, bpp_indices, bpp_mask)

        # 3. Head
        out = self.head(x)

        return out
