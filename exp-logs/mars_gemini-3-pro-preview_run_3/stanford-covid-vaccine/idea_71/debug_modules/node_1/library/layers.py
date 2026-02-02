import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledGLUInteraction(nn.Module):
    """
    Decoupled GLU-Structural Interaction Module.

    Implements the structural injection mechanism:
    1. Gathers paired hidden states (h_j) based on bpp_indices.
    2. Applies zero-masking for unpaired bases (h_j = 0 if unpaired).
    3. Computes a GLU message with bias-driven refinement for unpaired loops.
    4. Integrates the message via a Full-Rank Stabilized Gate.

    Reference: High-Capacity FFN-Augmented Synthesis (Idea 71)
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message Computation: Linear -> GLU
        # Projects to 2 * hidden_dim to allow for GLU splitting (Value, Gate)
        # If input is 0 (unpaired), the bias of this Linear layer acts as the "Loop Embedding".
        self.message_fc = nn.Linear(hidden_dim, hidden_dim * 2)

        # Full-Rank Stabilized Gate: 2-layer MLP
        # Structure: Linear -> LayerNorm -> GELU -> Linear -> Sigmoid
        # "Internal LayerNorm" prevents saturation. "No Logit Norm" allows switching.
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Output LayerNorm for the residual connection
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, bpp_indices, pair_mask):
        """
        Args:
            h (torch.Tensor): Input hidden states (Batch, Seq, Hidden).
            bpp_indices (torch.Tensor): Indices of paired bases (Batch, Seq).
                                        Values should be in range [0, Seq-1].
            pair_mask (torch.Tensor): Mask indicating paired status (Batch, Seq).
                                      1.0 if paired, 0.0 if unpaired.

        Returns:
            torch.Tensor: Structurally refined hidden states (Batch, Seq, Hidden).
        """
        batch_size, seq_len, _ = h.shape

        # 1. Gather Neighbor States (Point-to-Point)
        # We flatten the batch and sequence dimensions to use flat indices
        flat_h = h.reshape(batch_size * seq_len, -1)

        # Create batch offsets: [[0], [L], [2L], ...]
        batch_offsets = (torch.arange(batch_size, device=h.device) * seq_len).unsqueeze(
            1
        )

        # Adjust bpp_indices to point to the correct index in the flattened tensor
        flat_indices = (bpp_indices + batch_offsets).view(-1)

        # Gather
        h_neighbor = flat_h[flat_indices].view(batch_size, seq_len, -1)

        # 2. Input Zero-Masking
        # If base i is unpaired, pair_mask[i] is 0. We force h_neighbor to 0.
        # This ensures that for unpaired bases, the message comes purely from the bias terms.
        h_neighbor = h_neighbor * pair_mask.unsqueeze(-1)

        # 3. GLU Message
        # m_{ij} = Linear(h_j) * sigmoid(Linear(h_j))
        # Note: If h_j is 0, this becomes Bias_Val * sigmoid(Bias_Gate)
        msg_raw = self.message_fc(h_neighbor)
        msg_val, msg_gate = torch.chunk(msg_raw, 2, dim=-1)
        m_ij = msg_val * torch.sigmoid(msg_gate)

        # 4. Full-Rank Stabilized Gate
        # g_{ij} = MLP(h_i)
        g_ij = self.gate_mlp(h)

        # 5. Injection & Residual
        # h_{struct} = LayerNorm(h_in + g_{ij} * m_{ij})
        update = g_ij * m_ij
        update = self.dropout(update)

        h_struct = self.out_norm(h + update)

        return h_struct


class PointwiseFFN(nn.Module):
    """
    Pointwise Feed-Forward Network (FFN).

    Augments the recurrent backbone with additional non-linear processing depth.
    Structure: Linear -> GELU -> Dropout -> Linear
    Residual: h_out = LayerNorm(x + FFN(x))
    """

    def __init__(self, hidden_dim, dropout=0.1, expansion_factor=4):
        super().__init__()

        intermediate_dim = int(hidden_dim * expansion_factor)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, hidden_dim),
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (Batch, Seq, Hidden).

        Returns:
            torch.Tensor: Output tensor (Batch, Seq, Hidden).
        """
        # FFN transformation
        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)

        # Residual Connection + LayerNorm
        out = self.out_norm(x + ffn_out)

        return out
