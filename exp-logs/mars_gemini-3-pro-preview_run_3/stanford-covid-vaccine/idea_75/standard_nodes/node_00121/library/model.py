import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GLUInteractionBlock(nn.Module):
    """
    Full-Rank GLU-Decoupled Structural Injection Module.

    Mechanisms:
    1. Point-to-Point Gathering: Retrieves neighbor states h_j based on pair indices.
    2. Input Zero-Masking: Explicitly forces h_j=0 for unpaired bases to prevent self-loops.
    3. Bias-Refined GLU Message: m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g).
       For unpaired bases (h_j=0), this acts as a learnable bias embedding.
    4. Full-Rank Stabilized Gate: Wide projection with internal LayerNorm to control signal flow.
    5. Residual Injection & Post-Normalization.
    """

    def __init__(self, hidden_dim):
        super(GLUInteractionBlock, self).__init__()
        self.hidden_dim = hidden_dim

        # GLU Message Components
        # W_c and W_g operate on h_j (neighbor state)
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        # Full-Rank Gate Components
        # Input: Concatenation of h_i (self) and h_j (neighbor) -> 2 * hidden_dim
        # Output: Full-Rank hidden_dim (768)
        self.W_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_gate = nn.LayerNorm(hidden_dim)
        self.W_out = nn.Linear(hidden_dim, hidden_dim)

        # Post-Normalization for the residual block
        self.ln_post = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Seq_Len, Hidden_Dim).
            pair_indices (torch.Tensor): Structural adjacency indices (Batch, Seq_Len).

        Returns:
            torch.Tensor: Refined features (Batch, Seq_Len, Hidden_Dim).
        """
        B, L, D = x.shape

        # 1. Gather Neighbors
        # Expand pair_indices to match feature dimension: (B, L, D)
        idx = pair_indices.unsqueeze(-1).expand(-1, -1, D)
        # Gather h_j from x using the indices
        h_j = torch.gather(x, 1, idx)

        # 2. Input Zero-Masking
        # Identify unpaired bases: where pair_indices[i] == i
        arange = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        # Mask shape: (B, L, 1)
        mask = (pair_indices == arange).unsqueeze(-1)
        # Force neighbor state to 0 if unpaired
        h_j = h_j.masked_fill(mask, 0.0)

        # 3. GLU Message Calculation
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # When h_j is 0, this becomes b_c * sigmoid(b_g) (Bias-Refined Loop Embedding)
        msg_content = self.W_c(h_j)
        msg_gate = torch.sigmoid(self.W_g(h_j))
        m_ij = msg_content * msg_gate

        # 4. Full-Rank Stabilized Gate
        # Concatenate self (x) and neighbor (h_j)
        cat_input = torch.cat([x, h_j], dim=-1)  # (B, L, 2*D)

        # Wide Projection -> LayerNorm -> GELU -> Linear -> Sigmoid
        z_wide = self.W_in(cat_input)
        z_norm = self.ln_gate(z_wide)  # Internal Normalization
        z_act = F.gelu(z_norm)
        logits = self.W_out(z_act)
        g_ij = torch.sigmoid(logits)  # No Logit Norm, allow saturation

        # 5. Injection
        # Gated addition of the message
        h_res = x + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.ln_post(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    High-Capacity Full-Rank GLU-Decoupled BiGRU Model.

    Architecture:
    1. Input: One-Hot encoded features (Sequence, Structure, Loop Type).
    2. Stem: 1D Convolution + GELU + Dropout.
    3. Backbone: 4 Layers of High-Capacity BiGRU (768 dim).
       - Layers 0, 1, 2 are followed by GLUInteractionBlock.
       - Layer 3 is BiGRU only.
    4. Head: Linear Projection to 5 targets.
    """

    def __init__(self, config: Config):
        super(RNAModel, self).__init__()

        # Dimensions
        # Input channels: 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
        self.input_dim = 14
        self.stem_dim = 256
        # BiGRU Hidden Dim: config.hidden_dim (384) * 2 = 768
        self.backbone_dim = config.hidden_dim * 2

        self.n_layers = config.n_layers
        self.dropout_rate = config.dropout

        # 1. Convolutional Stem
        # Projects sparse inputs into dense embedding space
        self.stem_conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.stem_dim,
            kernel_size=config.kernel_size,
            padding=config.kernel_size // 2,
        )
        self.stem_dropout = nn.Dropout(self.dropout_rate)

        # 2. Backbone Layers
        self.grus = nn.ModuleList()
        self.interactions = nn.ModuleList()

        current_input_dim = self.stem_dim

        for i in range(self.n_layers):
            # Bidirectional GRU
            # Maintains high capacity (384 per direction)
            gru = nn.GRU(
                input_size=current_input_dim,
                hidden_size=config.hidden_dim,  # 384
                batch_first=True,
                bidirectional=True,
            )
            self.grus.append(gru)

            # Update input dimension for next layer (output is 768)
            current_input_dim = self.backbone_dim

            # Add Interaction Block for first N-1 layers
            if i < self.n_layers - 1:
                self.interactions.append(GLUInteractionBlock(self.backbone_dim))

        # 3. Output Head
        self.head = nn.Linear(self.backbone_dim, config.num_targets)

    def forward(self, x, pair_indices):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Seq_Len, 14).
            pair_indices (torch.Tensor): Adjacency indices (Batch, Seq_Len).

        Returns:
            torch.Tensor: Predictions (Batch, Seq_Len, 5).
        """
        # Permute for Conv1d: (B, 14, L)
        x = x.permute(0, 2, 1)

        # Stem Processing
        x = self.stem_conv(x)
        x = F.gelu(x)

        # Permute back for RNN: (B, L, 256)
        x = x.permute(0, 2, 1)
        x = self.stem_dropout(x)

        # Backbone Processing
        # Iterate through layers
        for i in range(self.n_layers):
            # Apply BiGRU
            # x shape becomes (B, L, 768)
            x, _ = self.grus[i](x)

            # Apply Interaction Block if available for this layer
            if i < len(self.interactions):
                x = self.interactions[i](x, pair_indices)

        # Output Head
        out = self.head(x)

        return out
