import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StructuralInteractionModule(nn.Module):
    """
    Decoupled Structural Interaction Module with Bias-Driven Refinement.

    Implements:
    1. Decoupled Gather: Retrieve neighbor state h_j.
    2. Input Zero-Masking: Force h_j = 0 if unpaired.
    3. Bias-Refined Message: m_ij = GELU(W_msg * h_j + b_msg).
       If unpaired, m_ij = GELU(b_msg) (learnable loop embedding).
    4. Stabilized MLP Gate:
       z = LayerNorm(W_g1 * [h_i; h_j]) -> GELU -> W_g2 -> Sigmoid.
    5. Injection: h_new = h_i + g_ij * m_ij.
    6. Post-Normalization: LayerNorm(h_new).
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message projection
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Gate MLP
        # Input is [h_i; h_j] -> 2 * hidden_dim
        self.gate_proj1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.gate_norm = nn.LayerNorm(hidden_dim)
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Post-injection normalization
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x: (Batch, Seq_Len, Hidden_Dim)
            pair_indices: (Batch, Seq_Len) - Indices of paired bases.
            pair_mask: (Batch, Seq_Len) - 1.0 if paired, 0.0 if unpaired.
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 1. Gather h_j
        # Create flat indices for gathering
        # pair_indices contains values in [0, seq_len-1]
        # We need to offset them by batch index * seq_len to index into the flattened batch
        batch_offsets = torch.arange(batch_size, device=x.device).unsqueeze(1) * seq_len
        flat_indices = (pair_indices + batch_offsets).view(-1)

        # Flatten x to (B*L, H)
        x_flat = x.view(-1, hidden_dim)

        # Gather and reshape back to (B, L, H)
        x_j = x_flat[flat_indices].view(batch_size, seq_len, hidden_dim)

        # 2. Input Zero-Masking
        # pair_mask is (B, L). Expand to (B, L, 1) for broadcasting.
        mask = pair_mask.unsqueeze(-1)
        x_j = x_j * mask  # If unpaired, x_j becomes 0 vector

        # 3. Decoupled Message (Bias-Refined)
        # m_ij = GELU(W h_j + b)
        # If x_j is 0, this becomes GELU(b), a learnable constant vector per channel
        m_ij = F.gelu(self.msg_proj(x_j))

        # 4. Stabilized MLP Gate
        # Concatenate h_i (x) and h_j (x_j)
        cat_input = torch.cat([x, x_j], dim=-1)  # (B, L, 2H)

        # Project
        z_raw = self.gate_proj1(cat_input)

        # Internal Normalization (Stabilization)
        z_norm = self.gate_norm(z_raw)

        # Activation
        z_act = F.gelu(z_norm)

        # Logit Projection and Sigmoid (No Logit Norm)
        logits = self.gate_proj2(z_act)
        g_ij = torch.sigmoid(logits)

        # 5. Injection
        h_res = x + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class HC_DBR_BiGRU(nn.Module):
    """
    High-Capacity Decoupled Bias-Refined BiGRU.

    Structure:
    1. Conv1d Stem
    2. 4-Layer Backbone:
       - Layer 1: BiGRU -> Interaction
       - Layer 2: BiGRU -> Interaction
       - Layer 3: BiGRU -> Interaction
       - Layer 4: BiGRU -> (No Interaction)
    3. Linear Head
    """

    def __init__(self):
        super().__init__()

        # Dimensions
        self.num_features = Config.NUM_FEATURES
        self.conv_filters = Config.CONV_FILTERS
        self.hidden_dim = Config.HIDDEN_DIM * 2  # Bidirectional (384 * 2 = 768)
        self.num_layers = Config.NUM_LAYERS  # 4
        self.num_targets = Config.NUM_TARGETS

        # 1. Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.num_features,
                out_channels=self.conv_filters,
                kernel_size=Config.CONV_KERNEL,
                padding=Config.CONV_KERNEL // 2,
            ),
            nn.GELU(),
        )

        # 2. Backbone
        self.gru_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()

        current_dim = self.conv_filters

        for i in range(self.num_layers):
            # BiGRU Layer
            gru = nn.GRU(
                input_size=current_dim,
                hidden_size=Config.HIDDEN_DIM,  # 384
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(gru)

            # Interaction Layer
            # Applied to all blocks EXCEPT the final one
            if i < self.num_layers - 1:
                interaction = StructuralInteractionModule(self.hidden_dim)
                self.interaction_layers.append(interaction)

            # Update dimension for next layer (always 768 after first GRU)
            current_dim = self.hidden_dim

        # Dropout for regularization
        self.dropout = nn.Dropout(Config.DROPOUT)

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim, self.num_targets)

    def forward(self, features, pair_indices, pair_mask):
        """
        Args:
            features: (B, L, 14)
            pair_indices: (B, L)
            pair_mask: (B, L)
        """
        # 1. Stem
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = features.permute(0, 2, 1)
        x = self.stem(x)
        # Permute back: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # 2. Backbone
        for i in range(self.num_layers):
            # BiGRU
            # GRU returns (output, h_n). We only need output.
            x, _ = self.gru_layers[i](x)

            # Apply Dropout
            x = self.dropout(x)

            # Interaction Module (if exists for this layer)
            if i < len(self.interaction_layers):
                x = self.interaction_layers[i](x, pair_indices, pair_mask)

        # 3. Head
        out = self.head(x)

        return out
