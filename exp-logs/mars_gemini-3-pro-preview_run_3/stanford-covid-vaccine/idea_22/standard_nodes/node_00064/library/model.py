import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InteractionModule(nn.Module):
    """
    Non-Linear Channel-Gated Structural Interaction Module.

    This module implements the channel-wise gating mechanism to selectively
    propagate structural information between paired bases.

    Logic:
    1. Gather hidden state h_j from paired neighbor.
    2. Compute Message: m_ij = GELU(W_msg * h_j)
    3. Compute Gate: g_ij = Sigmoid(W_gate * [h_i; h_j])
    4. Update: h'_i = h_i + g_ij * m_ij
    5. Stabilize: LayerNorm(h'_i)
    """

    def __init__(self, dim):
        super(InteractionModule, self).__init__()

        # Message transformation: Non-linear projection of neighbor state
        self.msg_proj = nn.Linear(dim, dim)

        # Gating mechanism: Projects concatenated states [h_i; h_j] to a gate vector
        self.gate_proj = nn.Linear(dim * 2, dim)

        # Stabilization
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x, pair_indices):
        """
        Args:
            x (torch.Tensor): Hidden states (Batch, Seq_Len, Dim).
            pair_indices (torch.Tensor): Indices of paired bases (Batch, Seq_Len).

        Returns:
            torch.Tensor: Updated hidden states.
        """
        B, L, D = x.shape

        # 1. Gather neighbor states h_j
        # Create batch indices grid: (B, L)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, L)

        # Gather: x[b, pair_indices[b, i], :]
        h_j = x[batch_idx, pair_indices, :]  # (B, L, D)
        h_i = x

        # 2. Compute Non-Linear Message
        # m_ij = GELU(W_msg * h_j)
        m_ij = self.act(self.msg_proj(h_j))

        # 3. Compute Channel-Wise Gate
        # Concatenate h_i and h_j: (B, L, 2*D)
        cat_h = torch.cat([h_i, h_j], dim=-1)
        # g_ij = Sigmoid(W_gate * [h_i; h_j])
        g_ij = torch.sigmoid(self.gate_proj(cat_h))

        # 4. Injection
        # Element-wise gating of the message
        update = g_ij * m_ij

        # Residual connection
        out = h_i + update

        # 5. Stabilization
        out = self.norm(out)

        return out


class NonLinearChannelGatedBiGRU(nn.Module):
    """
    Deep BiGRU with Interleaved Non-Linear Channel-Gated Structural Injection.

    Architecture:
    1. Convolutional Stem (1D Conv + GELU)
    2. Refinement Backbone (3 Blocks of BiGRU + Interaction)
       - Note: The final block does NOT have the interaction module.
    3. Output Head (Linear)
    """

    def __init__(self):
        super(NonLinearChannelGatedBiGRU, self).__init__()

        # ==============================
        # 1. Convolutional Stem
        # ==============================
        # Projects (N, 14, L) -> (N, 256, L)
        self.stem_conv = nn.Conv1d(
            in_channels=Config.INPUT_CHANNELS,
            out_channels=Config.CONV_FILTERS,
            kernel_size=Config.KERNEL_SIZE,
            padding=Config.KERNEL_SIZE // 2,
        )
        self.stem_act = nn.GELU()
        self.stem_dropout = nn.Dropout(Config.DROPOUT)

        # ==============================
        # 2. Refinement Backbone
        # ==============================
        self.blocks = nn.ModuleList()

        # Input dimension for the first GRU is the output of the stem
        current_input_dim = Config.CONV_FILTERS

        # The backbone consists of 3 blocks
        for i in range(Config.NUM_LAYERS):
            # Determine if this is the final block
            is_last_block = i == Config.NUM_LAYERS - 1

            # BiGRU Layer
            # Hidden size is strictly 384. Bidirectional=True implies output dim is 768.
            gru = nn.GRU(
                input_size=current_input_dim,
                hidden_size=Config.HIDDEN_DIM,
                batch_first=True,
                bidirectional=True,
            )

            # Calculate output dimension of the GRU (2 * hidden)
            gru_out_dim = Config.HIDDEN_DIM * 2

            # Dropout for regularization
            dropout = nn.Dropout(Config.DROPOUT)

            # Interaction Module
            # Applied in all blocks EXCEPT the final one
            interaction = None
            if not is_last_block:
                interaction = InteractionModule(gru_out_dim)

            # Store block components
            self.blocks.append(
                nn.ModuleDict(
                    {"gru": gru, "dropout": dropout, "interaction": interaction}
                )
            )

            # Update input dimension for the next block
            current_input_dim = gru_out_dim

        # ==============================
        # 3. Output Head
        # ==============================
        # Projects final hidden states to the 5 target classes
        self.head = nn.Linear(current_input_dim, Config.NUM_CLASSES)

    def forward(self, inputs, pair_indices):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input features. Shape (Batch, 107, 14).
            pair_indices (torch.Tensor): Pair indices. Shape (Batch, 107).

        Returns:
            torch.Tensor: Predictions. Shape (Batch, 107, 5).
        """
        # 1. Stem
        # Permute to (Batch, Channels, Seq_Len) for Conv1d
        x = inputs.permute(0, 2, 1)

        x = self.stem_conv(x)
        x = self.stem_act(x)
        x = self.stem_dropout(x)

        # Permute back to (Batch, Seq_Len, Channels) for GRU
        x = x.permute(0, 2, 1)

        # 2. Backbone
        for block in self.blocks:
            gru = block["gru"]
            dropout = block["dropout"]
            interaction = block["interaction"]

            # BiGRU Pass
            # x shape: (Batch, Seq_Len, Input_Dim)
            # output shape: (Batch, Seq_Len, 2*Hidden_Dim)
            x, _ = gru(x)

            # Dropout
            x = dropout(x)

            # Interaction Pass (if applicable)
            if interaction is not None:
                x = interaction(x, pair_indices)

        # 3. Head
        # x shape: (Batch, Seq_Len, 768)
        out = self.head(x)  # (Batch, Seq_Len, 5)

        return out
