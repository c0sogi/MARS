import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class VerticalResidualBiGRU(nn.Module):
    """
    A Bidirectional GRU layer with an optional vertical residual connection.

    If the input dimension matches the output dimension (hidden_dim * 2),
    a residual connection is added: x = x + Dropout(GRU(x)).
    Otherwise, it acts as a standard BiGRU: x = Dropout(GRU(x)).

    This facilitates gradient flow in deep recurrent backbones.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim * 2  # Bidirectional
        self.dropout_p = dropout

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Determine if residual connection is geometrically possible
        self.use_residual = self.input_dim == self.output_dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, Input_Dim)

        Returns:
            torch.Tensor: Output tensor of shape (Batch, Seq_Len, Hidden_Dim * 2)
        """
        # GRU returns (output, h_n). We only need output.
        gru_out, _ = self.gru(x)

        # Apply dropout to the output of the GRU
        out = self.dropout(gru_out)

        if self.use_residual:
            return x + out
        else:
            return out


class UnifiedGLUInteraction(nn.Module):
    """
    Unified GLU-Decoupled Interaction Module.

    Synthesizes structural information by passing messages between paired bases.

    Mechanism:
    1. Gather: Retrieve features h_j for each position i based on pair_indices.
    2. Zero-Masking: Unpaired bases (index -1) get h_j = 0.
    3. GLU Message: m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g).
       For unpaired bases, this reduces to bias terms, acting as loop embeddings.
    4. Stabilized Gate: Computes an injection gate based on [h_i; h_j].
    5. Injection: h_out = LayerNorm(h_i + gate * message).
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- GLU Message Components ---
        # Content transformation
        self.w_c = nn.Linear(hidden_dim, hidden_dim)
        # Gate transformation for the message
        self.w_g = nn.Linear(hidden_dim, hidden_dim)

        # --- Stabilized Injection Gate ---
        # Input is concatenation of h_i and h_j
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.gate_out = nn.Linear(hidden_dim, hidden_dim)

        # --- Final Normalization ---
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Seq_Len, Hidden_Dim).
            pair_indices (torch.Tensor): Structural indices (Batch, Seq_Len).
                                         Values are indices of paired bases, or -1 if unpaired.

        Returns:
            torch.Tensor: Structurally enriched features (Batch, Seq_Len, Hidden_Dim).
        """
        batch_size, seq_len, _ = x.shape

        # 1. Gather paired features h_j
        # Replace -1 with 0 to allow valid gathering (we will mask later)
        # Clone to avoid modifying the input tensor in place if it's reused
        gather_indices = pair_indices.clone()
        mask_unpaired = gather_indices == -1
        gather_indices[mask_unpaired] = 0

        # Expand indices for gathering across the feature dimension
        # gather_indices shape: (B, L) -> (B, L, D)
        gather_indices_expanded = gather_indices.unsqueeze(-1).expand(
            -1, -1, self.hidden_dim
        )

        # Gather: h_pair[b, i, :] = x[b, gather_indices[b, i], :]
        h_pair = torch.gather(x, 1, gather_indices_expanded)

        # 2. Zero-Masking
        # Explicitly force h_pair to 0 where bases are unpaired
        # mask_unpaired shape: (B, L) -> (B, L, 1)
        mask_expanded = mask_unpaired.unsqueeze(-1)
        h_pair = h_pair.masked_fill(mask_expanded, 0.0)

        # 3. GLU Message Calculation
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # Note: When h_j is 0 (unpaired), this becomes b_c * sigmoid(b_g)
        content = self.w_c(h_pair)
        gate_msg = torch.sigmoid(self.w_g(h_pair))
        message = content * gate_msg

        # 4. Stabilized Injection Gate
        # Input: Concatenation of h_i (x) and h_j (h_pair)
        concat_input = torch.cat([x, h_pair], dim=-1)

        # Wide Projection -> LayerNorm -> GELU -> Linear -> Sigmoid
        z_raw = self.gate_proj(concat_input)
        z_norm = self.gate_norm(z_raw)
        z_act = F.gelu(z_norm)
        injection_gate = torch.sigmoid(self.gate_out(z_act))

        # 5. Injection and Post-Normalization
        # h_struct = h_i + gate * message
        h_struct = x + (injection_gate * message)

        # Final stability norm
        out = self.final_norm(h_struct)

        return out
