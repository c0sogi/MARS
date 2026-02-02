import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralInteractionLayer(nn.Module):
    """
    Implements the Stabilized Decoupled Interaction Module (SDIM).

    Key Features:
    1. Point-to-Point Gathering: Retrieves paired states directly.
    2. Input Zero-Masking: Explicitly forces unpaired neighbor states to zero.
    3. Bias-Refined Message: Unpaired positions generate messages from learnable bias (loop embedding).
    4. Stabilized MLP Gate: Uses internal LayerNorm to prevent saturation, without normalizing logits.
    5. Post-Normalization: Applies LayerNorm after the residual connection.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(StructuralInteractionLayer, self).__init__()

        self.hidden_dim = hidden_dim

        # ==========================================
        # Decoupled Message Pathway
        # ==========================================
        # Calculates m_ij based ONLY on h_j (neighbor).
        # If h_j is masked (0), the output is GELU(bias), serving as a loop embedding.
        self.w_msg = nn.Linear(hidden_dim, hidden_dim)

        # ==========================================
        # Stabilized Gating Pathway
        # ==========================================
        # Projects joint context [h_i; h_j] to determine gate value g_ij.
        # w_g1: Projects 2*hidden_dim -> hidden_dim
        self.w_g1 = nn.Linear(hidden_dim * 2, hidden_dim)

        # Internal Normalization: Stabilizes the hidden layer of the gate MLP.
        self.ln_gate = nn.LayerNorm(hidden_dim)

        # w_g2: Projects hidden_dim -> hidden_dim (Logits)
        self.w_g2 = nn.Linear(hidden_dim, hidden_dim)

        # ==========================================
        # Injection & Output
        # ==========================================
        self.dropout = nn.Dropout(dropout)

        # Post-Normalization layer
        self.ln_out = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x (torch.Tensor): Input embeddings of shape (Batch, Seq_Len, Hidden_Dim).
            pair_indices (torch.Tensor): Indices of paired bases (Batch, Seq_Len).
            pair_mask (torch.Tensor): Mask where 1.0 indicates paired, 0.0 unpaired (Batch, Seq_Len).

        Returns:
            torch.Tensor: Updated embeddings of shape (Batch, Seq_Len, Hidden_Dim).
        """
        batch_size, seq_len, _ = x.size()

        # 1. Gather Neighbor States (h_j)
        # Expand indices to match embedding dimension: (B, L) -> (B, L, D)
        idx_expanded = pair_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        # Gather: For each position i, get the vector at index pair_indices[i]
        h_j = torch.gather(x, 1, idx_expanded)

        # 2. Input Zero-Masking
        # Explicitly force h_j to 0 where the base is unpaired.
        # This prevents self-loops (if pair_indices pointed to self) and noise.
        # mask_expanded: (B, L, 1)
        mask_expanded = pair_mask.unsqueeze(-1)
        h_j = h_j * mask_expanded

        # 3. Compute Decoupled Message (Bias-Refined)
        # m_ij = GELU(W_msg * h_j + b_msg)
        # For unpaired bases (h_j=0), m_ij = GELU(b_msg).
        m_ij = F.gelu(self.w_msg(h_j))

        # 4. Compute Stabilized MLP Gate
        # Input: Concatenation of self (h_i) and neighbor (h_j)
        # Shape: (B, L, 2*D)
        cat_input = torch.cat([x, h_j], dim=-1)

        # Project Joint Context (z_raw)
        z_raw = self.w_g1(cat_input)

        # Internal Normalization
        z_norm = self.ln_gate(z_raw)

        # Activation
        z_act = F.gelu(z_norm)

        # Logit Projection (No normalization here)
        logits = self.w_g2(z_act)

        # Sigmoid Activation to get gate values [0, 1]
        g_ij = torch.sigmoid(logits)

        # 5. Injection
        # Update signal: Gate * Message
        update = g_ij * m_ij
        update = self.dropout(update)

        # Residual Connection
        h_res = x + update

        # 6. Post-Normalization
        # Stabilizes the backbone for deeper stacking
        h_out = self.ln_out(h_res)

        return h_out
