import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StructuralInteractionModule(nn.Module):
    """
    Decoupled Structural Interaction Module with Bias-Driven Loop Refinement.

    Implements the logic:
    1. Gather neighbor states h_j (Point-to-Point).
    2. Zero-mask h_j for unpaired bases (No self-loops).
    3. Message: m_ij = GELU(W_msg * h_j + b_msg).
       - For unpaired bases, h_j=0, so m_ij = GELU(b_msg). This acts as a learnable bias.
    4. Gate: g_ij = Sigmoid(W_gate * [h_i; h_j]).
    5. Injection: h_res = h_i + g_ij * m_ij.
    6. Post-Normalization: LayerNorm(h_res).
    """

    def __init__(self, hidden_dim):
        super(StructuralInteractionModule, self).__init__()
        self.hidden_dim = hidden_dim

        # Message projection: W_msg, b_msg
        # Projects from hidden_dim to hidden_dim
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gate projection: W_gate
        # Takes concatenation of h_i and h_j (2 * hidden_dim) -> hidden_dim
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Post-Normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Activation
        self.act = nn.GELU()

    def forward(self, x, pair_indices):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Hidden_Dim)
            pair_indices: LongTensor of shape (Batch, Seq_Len) where unpaired bases are -1.
        """
        B, L, D = x.shape

        # 1. Create mask for paired bases (True if paired, False if unpaired)
        # pair_indices is (B, L)
        paired_mask = (pair_indices != -1).unsqueeze(-1)  # (B, L, 1)

        # 2. Gather h_j safely
        # We need to gather x[b, pair_indices[b, i], :]
        # Since pair_indices contains -1, we clamp them to 0 to avoid indexing errors,
        # then multiply the result by the mask to zero out the invalid gathers.
        safe_indices = pair_indices.clone()
        safe_indices[pair_indices == -1] = 0

        # Expand indices for gather: (B, L, D)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, D)

        # Gather: h_j_raw
        h_j_raw = torch.gather(x, 1, gather_indices)

        # Apply mask: Force h_j to 0 where unpaired
        h_j = h_j_raw * paired_mask.float()

        # 3. Compute Message
        # m_ij = GELU(W_msg * h_j + b_msg)
        msg = self.act(self.msg_proj(h_j))

        # 4. Compute Gate
        # g_ij = Sigmoid(W_gate * [h_i; h_j])
        cat_input = torch.cat([x, h_j], dim=-1)
        gate = torch.sigmoid(self.gate_proj(cat_input))

        # 5. Residual Update
        h_res = x + gate * msg

        # 6. Post-Normalization
        out = self.norm(h_res)

        return out


class DeepBiasRefinedBiGRU(nn.Module):
    """
    4-Layer Bidirectional GRU with Interleaved Decoupled Post-Norm Structural Injection.

    Structure:
    - Input (One-Hot)
    - Conv1d Stem
    - Block 1: BiGRU + Interaction
    - Block 2: BiGRU + Interaction
    - Block 3: BiGRU + Interaction
    - Block 4: BiGRU (No Interaction)
    - Output Head
    """

    def __init__(self):
        super(DeepBiasRefinedBiGRU, self).__init__()

        # Dimensions from Config
        self.in_channels = Config.IN_CHANNELS
        self.stem_filters = Config.STEM_FILTERS
        self.hidden_dim = Config.HIDDEN_DIM  # 384
        self.gru_out_dim = self.hidden_dim * 2  # 768 (Bidirectional)

        # 1. Convolutional Stem
        # Projects sparse inputs to dense embedding space
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.in_channels,
                self.stem_filters,
                kernel_size=Config.STEM_KERNEL_SIZE,
                padding=1,
            ),
            nn.GELU(),
        )

        # 2. Backbone Blocks
        self.grus = nn.ModuleList()
        self.interactions = nn.ModuleList()

        # Layer 1
        # Input: stem_filters (256) -> Output: gru_out_dim (768)
        self.grus.append(
            nn.GRU(
                self.stem_filters, self.hidden_dim, batch_first=True, bidirectional=True
            )
        )
        self.interactions.append(StructuralInteractionModule(self.gru_out_dim))

        # Layer 2
        # Input: 768 -> Output: 768
        self.grus.append(
            nn.GRU(
                self.gru_out_dim, self.hidden_dim, batch_first=True, bidirectional=True
            )
        )
        self.interactions.append(StructuralInteractionModule(self.gru_out_dim))

        # Layer 3
        # Input: 768 -> Output: 768
        self.grus.append(
            nn.GRU(
                self.gru_out_dim, self.hidden_dim, batch_first=True, bidirectional=True
            )
        )
        self.interactions.append(StructuralInteractionModule(self.gru_out_dim))

        # Layer 4 (Final Block)
        # Input: 768 -> Output: 768
        # No interaction for the final block as per strategy
        self.grus.append(
            nn.GRU(
                self.gru_out_dim, self.hidden_dim, batch_first=True, bidirectional=True
            )
        )

        # Dropout
        self.dropout = nn.Dropout(Config.DROPOUT)

        # 3. Output Head
        # Projects final hidden state to 5 targets
        self.head = nn.Linear(self.gru_out_dim, 5)

    def forward(self, features, pair_indices, unpaired_mask=None):
        """
        Args:
            features: (N, L, 14)
            pair_indices: (N, L)
            unpaired_mask: (N, L) - Not explicitly used as pair_indices handles logic
        """
        # Permute for Conv1d: (N, L, C) -> (N, C, L)
        x = features.permute(0, 2, 1)

        # Stem
        x = self.stem(x)

        # Permute back for GRU: (N, C, L) -> (N, L, C)
        x = x.permute(0, 2, 1)

        # Backbone Execution

        # Block 1
        x, _ = self.grus[0](x)
        x = self.interactions[0](x, pair_indices)
        x = self.dropout(x)

        # Block 2
        x, _ = self.grus[1](x)
        x = self.interactions[1](x, pair_indices)
        x = self.dropout(x)

        # Block 3
        x, _ = self.grus[2](x)
        x = self.interactions[2](x, pair_indices)
        x = self.dropout(x)

        # Block 4 (Final - No Interaction)
        x, _ = self.grus[3](x)
        x = self.dropout(x)

        # Output Head
        out = self.head(x)

        return out
