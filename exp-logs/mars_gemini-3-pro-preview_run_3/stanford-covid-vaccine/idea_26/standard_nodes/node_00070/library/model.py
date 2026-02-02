import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_CHANNELS,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    CONV_FILTERS,
    CONV_KERNEL_SIZE,
    NUM_TARGETS,
)


class StructuralInteractionLayer(nn.Module):
    """
    Post-Norm Structural Interaction Module with Zero-Masked Channel-Gating.

    This layer enables point-to-point communication between paired bases in the RNA structure.
    It uses a gating mechanism to control information flow and applies Layer Normalization
    after the residual connection to stabilize deep network training.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(StructuralInteractionLayer, self).__init__()
        self.hidden_dim = hidden_dim

        # Message projection: Transform neighbor features
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gate projection: Determine how much of the message to accept based on self + neighbor
        # Input: [h_i; h_j] -> 2 * hidden_dim
        self.gate_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # Layer Normalization (Post-Norm)
        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Hidden_Dim)
            pair_indices: LongTensor of shape (Batch, Seq_Len) containing indices of paired bases.
            pair_mask: FloatTensor of shape (Batch, Seq_Len) containing 1.0 for paired, 0.0 for unpaired.
        """
        batch_size, seq_len, dim = x.shape

        # 1. Gather Neighbor Features
        # We need to gather x[b, pair_indices[b, i], :] for each i.
        # Flatten batch and sequence for easier indexing or use torch.gather

        # Expand indices to match feature dimension for gather
        # pair_indices shape: (B, L) -> (B, L, D)
        idx_expanded = pair_indices.unsqueeze(-1).expand(-1, -1, dim)

        # Gather: dim=1 is the sequence dimension
        neighbor_x = torch.gather(x, 1, idx_expanded)

        # 2. Zero-Masking
        # Force features of unpaired bases (which might point to index 0 or garbage) to strictly Zero.
        # pair_mask shape: (B, L) -> (B, L, 1)
        mask_expanded = pair_mask.unsqueeze(-1)
        neighbor_x = neighbor_x * mask_expanded

        # 3. Compute Message
        # m_ij = GELU(W_msg * h_j)
        msg = F.gelu(self.msg_proj(neighbor_x))

        # 4. Compute Gate
        # g_ij = Sigmoid(W_gate * [h_i; h_j])
        cat_features = torch.cat([x, neighbor_x], dim=-1)
        gate = torch.sigmoid(self.gate_proj(cat_features))

        # 5. Residual Update
        # h_res = h_i + g_ij * m_ij
        update = gate * msg
        x_res = x + self.dropout(update)

        # 6. Post-Normalization
        # h_out = LayerNorm(h_res)
        out = self.norm(x_res)

        return out


class DeepPostNormBiGRU(nn.Module):
    """
    Deep Post-Norm BiGRU with Zero-Masked Channel-Gating.

    Architecture:
    1. Conv1d Stem
    2. Stack of BiGRU blocks.
    3. Structural Interaction Layers interleaved between BiGRUs (except after the last one).
    4. Linear Head.
    """

    def __init__(self):
        super(DeepPostNormBiGRU, self).__init__()

        # --- Stem ---
        # Projects 14 input channels to CONV_FILTERS (256)
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=INPUT_CHANNELS,
                out_channels=CONV_FILTERS,
                kernel_size=CONV_KERNEL_SIZE,
                padding=CONV_KERNEL_SIZE // 2,
            ),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        # --- Backbone ---
        self.gru_blocks = nn.ModuleList()
        self.interaction_blocks = nn.ModuleList()

        # We want NUM_LAYERS of BiGRUs.
        # We want (NUM_LAYERS - 1) Interaction layers interleaved.

        current_dim = CONV_FILTERS

        for i in range(NUM_LAYERS):
            # BiGRU: Hidden size is HIDDEN_DIM // 2 so output is HIDDEN_DIM
            gru = nn.GRU(
                input_size=current_dim,
                hidden_size=HIDDEN_DIM // 2,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_blocks.append(gru)

            # Interaction Layer (only between GRUs, not after the last one)
            if i < NUM_LAYERS - 1:
                interaction = StructuralInteractionLayer(HIDDEN_DIM, dropout=DROPOUT)
                self.interaction_blocks.append(interaction)

            # Update current_dim for the next layer (output of BiGRU is HIDDEN_DIM)
            current_dim = HIDDEN_DIM

        # --- Head ---
        self.head = nn.Linear(HIDDEN_DIM, NUM_TARGETS)

    def forward(self, inputs, pair_indices, pair_mask):
        """
        Args:
            inputs: (Batch, Seq_Len, Input_Channels)
            pair_indices: (Batch, Seq_Len)
            pair_mask: (Batch, Seq_Len)
        """
        # 1. Stem
        # Conv1d expects (Batch, Channels, Seq_Len)
        x = inputs.permute(0, 2, 1)
        x = self.stem(x)
        # Permute back to (Batch, Seq_Len, Channels) for RNN
        x = x.permute(0, 2, 1)

        # 2. Backbone
        for i in range(NUM_LAYERS):
            # Apply BiGRU
            # GRU returns (output, h_n), we only need output
            x, _ = self.gru_blocks[i](x)

            # Apply Interaction if available (Layers 0 to N-2)
            if i < len(self.interaction_blocks):
                x = self.interaction_blocks[i](x, pair_indices, pair_mask)

        # 3. Head
        # x shape: (Batch, Seq_Len, Hidden_Dim)
        out = self.head(x)

        return out
