import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StructuralInteractionModule(nn.Module):
    """
    Decoupled Structural Interaction Module with Bias-Driven Loop Refinement.

    Implements the logic:
    1. Gather neighbor features.
    2. Zero-Mask unpaired neighbors.
    3. Compute decoupled message (relying on bias for unpaired bases).
    4. Compute channel-wise gate using current state and masked neighbor.
    5. Residual update and Post-LayerNorm.
    """

    def __init__(self, dim):
        super().__init__()
        self.msg_proj = nn.Linear(dim, dim, bias=True)
        self.gate_proj = nn.Linear(dim * 2, dim, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x, pair_indices):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Dim)
            pair_indices: Tensor of shape (Batch, Seq_Len) with indices of paired bases (-1 for unpaired)
        """
        B, L, D = x.shape

        # 1. Prepare indices for gather
        # Create a mask for valid pairs (1 for paired, 0 for unpaired)
        mask = (pair_indices != -1).unsqueeze(-1).type_as(x)  # (B, L, 1)

        # Replace -1 with 0 to ensure valid indices for gather (masked out later)
        safe_indices = pair_indices.clone()
        safe_indices[safe_indices == -1] = 0

        # Expand indices to match feature dimension: (B, L, D)
        idx_expanded = safe_indices.unsqueeze(-1).expand(-1, -1, D)

        # Gather neighbor features: h_j
        x_neighbor = torch.gather(x, 1, idx_expanded.long())

        # 2. Zero-Masking
        # Force unpaired neighbors to be exactly 0 vector.
        # This prevents self-loops and noise injection from arbitrary indices.
        x_neighbor = x_neighbor * mask

        # 3. Decoupled Message & Bias-Driven Loop Refinement
        # m_ij = GELU(W_msg * h_j + b_msg)
        # For unpaired bases (h_j=0), m_ij = GELU(b_msg), serving as a learnable unpaired bias.
        msg = self.act(self.msg_proj(x_neighbor))

        # 4. Channel-Wise Gating
        # g_ij = sigma(W_gate * [h_i; h_j])
        # Uses the current state h_i to modulate how much of the neighbor/bias signal to accept.
        concat = torch.cat([x, x_neighbor], dim=-1)
        gate = torch.sigmoid(self.gate_proj(concat))

        # 5. Injection & Post-Normalization
        # h_res = h_i + g_ij * m_ij
        x_res = x + gate * msg
        x_out = self.norm(x_res)

        return x_out


class DeepDecoupledBiGRU(nn.Module):
    """
    4-Layer Bidirectional GRU with Interleaved Decoupled Post-Norm Structural Injection.
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters
        input_dim = Config.NUM_INPUT_CHANNELS
        stem_filters = Config.STEM_FILTERS
        hidden_dim = Config.HIDDEN_DIM
        num_layers = Config.NUM_LAYERS
        num_targets = Config.NUM_TARGETS
        kernel_size = Config.STEM_KERNEL_SIZE

        # 1. Convolutional Stem
        # Projects sparse one-hot inputs into dense embedding space
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_dim,
                stem_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
        )

        # 2. Deep Stabilized Backbone
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()

        # BiGRU output dimension is 2 * hidden_dim
        gru_out_dim = hidden_dim * 2

        for i in range(num_layers):
            # First layer takes stem output, others take previous block output
            in_size = stem_filters if i == 0 else gru_out_dim

            # BiGRU Layer
            self.gru_layers.append(
                nn.GRU(in_size, hidden_dim, batch_first=True, bidirectional=True)
            )

            # Structural Interaction Module
            # Applied after every block EXCEPT the final block
            if i < num_layers - 1:
                self.interaction_layers.append(StructuralInteractionModule(gru_out_dim))

        # 3. Output Head
        self.head = nn.Linear(gru_out_dim, num_targets)

    def forward(self, features, pair_indices):
        """
        Args:
            features: (Batch, Seq_Len, Channels)
            pair_indices: (Batch, Seq_Len)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = features.permute(0, 2, 1)

        # Stem
        x = self.stem(x)

        # Permute back: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # Backbone
        for i in range(len(self.gru_layers)):
            # BiGRU
            x, _ = self.gru_layers[i](x)

            # Interaction (if present for this layer)
            if i < len(self.interaction_layers):
                x = self.interaction_layers[i](x, pair_indices)

        # Output Head
        out = self.head(x)

        return out
