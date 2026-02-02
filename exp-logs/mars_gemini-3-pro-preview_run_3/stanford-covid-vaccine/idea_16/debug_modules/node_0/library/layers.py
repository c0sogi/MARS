import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralInjectionLayer(nn.Module):
    """
    Implements the Structural Injection Layer for the Interleaved Gated-Structure BiGRU.

    This layer allows the network to dynamically incorporate long-range dependencies
    defined by the secondary structure. It uses a gating mechanism to decide how much
    information to accept from the paired base, allowing it to filter out noise from
    incorrect structure predictions.

    Mechanism:
    1. Identify paired neighbor j for each position i.
    2. Gather hidden state h_j.
    3. Compute gate g_ij = sigmoid(W_gate * [h_i; h_j]).
    4. Update h_i = h_i + g_ij * (W_proj * h_j).
    """

    def __init__(self, hidden_dim, dropout=0.1):
        """
        Args:
            hidden_dim (int): The dimensionality of the hidden states.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Projection layer for the paired feature (W_proj)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        # Gating layer (W_gate)
        # Inputs: Concatenation of current state and paired state
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # LayerNorm for residual block stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Hidden_Dim).
            pair_indices (torch.Tensor): Indices of paired bases of shape (Batch, Seq_Len).
                                         Values are indices [0, Seq_Len-1], with -1 denoting unpaired.

        Returns:
            torch.Tensor: Output tensor of shape (Batch, Seq_Len, Hidden_Dim).
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 1. Create validity mask
        # -1 indicates unpaired bases. We want a mask of shape (B, L, 1)
        # 1.0 where paired, 0.0 where unpaired.
        mask = (pair_indices != -1).unsqueeze(-1).type_as(x)

        # 2. Prepare indices for gathering
        # We replace -1 with 0 to make indices valid for gather.
        # The features gathered from index 0 for unpaired bases will be masked out later.
        # Ensure indices are LongTensor
        safe_indices = pair_indices.clone().long()
        safe_indices[safe_indices == -1] = 0

        # Expand indices to (B, L, H) to match x for gathering along dim 1
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)

        # 3. Gather paired features h_j
        # x: (B, L, H)
        # gather_indices: (B, L, H)
        # h_pair: (B, L, H)
        h_pair = torch.gather(x, dim=1, index=gather_indices)

        # 4. Mask invalid pairs
        # Zero out the vectors retrieved for unpaired positions
        h_pair = h_pair * mask

        # 5. Compute Gate
        # Concatenate h_i and h_j along the feature dimension
        concat_features = torch.cat([x, h_pair], dim=-1)  # (B, L, 2*H)
        g = torch.sigmoid(self.gate(concat_features))  # (B, L, H)

        # 6. Compute Structural Update
        # Project the paired features: W_proj * h_j
        projected_pair = self.proj(h_pair)  # (B, L, H)

        # Apply gate and mask
        # The mask is technically redundant if h_pair is already masked and bias is handled,
        # but applying it here ensures the update is strictly zero for unpaired bases.
        update = g * projected_pair * mask

        # 7. Residual Connection and Normalization
        out = x + self.dropout(update)
        out = self.norm(out)

        return out
