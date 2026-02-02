import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StructuralInteractionModule(nn.Module):
    """
    Decoupled Structural Interaction Module.
    Implements Point-to-Point gathering, Bias-Driven Loop Refinement,
    Internal Gate Normalization, and Post-Normalization.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Decoupled Message: Projects h_j to message
        # If h_j is masked (0), this outputs the learned bias (Loop Embedding)
        # effectively refining the representation for unpaired bases.
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)

        # Stabilized MLP Gate
        # Input: Concatenation of h_i and h_j
        self.gate_proj1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)  # Internal Normalization
        self.gate_proj2 = nn.Linear(hidden_dim, hidden_dim)

        # Post-Normalization for the residual block
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, pair_indices, pair_mask):
        """
        Args:
            h: Tensor of shape (Batch, Seq_Len, Hidden_Dim)
            pair_indices: LongTensor of shape (Batch, Seq_Len)
            pair_mask: Tensor of shape (Batch, Seq_Len, 1)
        """
        B, L, D = h.shape

        # 1. Gather Context (Point-to-Point)
        # Create batch indices for gathering: [[0,0...], [1,1...], ...]
        batch_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(B, L)

        # Gather h_j: Select the feature vector of the paired base
        # h[b, i] corresponds to h_i
        # h[b, pair_indices[b, i]] corresponds to h_j
        h_j = h[batch_idx, pair_indices]  # (B, L, D)

        # 2. Input Zero-Masking
        # If unpaired, pair_mask is 0, forcing h_j to 0 vector.
        # This ensures unpaired bases rely purely on the bias of msg_proj.
        h_j = h_j * pair_mask

        # 3. Decoupled Message
        # m_ij = GELU(W_msg * h_j + b_msg)
        m_ij = F.gelu(self.msg_proj(h_j))

        # 4. Stabilized MLP Gate
        # Concatenate h_i (current) and h_j (paired context)
        cat = torch.cat([h, h_j], dim=-1)  # (B, L, 2D)

        # Project and Normalize internally (Stabilization)
        z_raw = self.gate_proj1(cat)
        z_norm = self.gate_norm(z_raw)
        z_act = F.gelu(z_norm)

        # Logit Projection and Sigmoid (No Logit Norm to allow saturation)
        logits = self.gate_proj2(z_act)
        g_ij = torch.sigmoid(logits)

        # 5. Injection
        # h_res = h_i + g_ij * m_ij
        h_res = h + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class DeepStabilizedBiGRU(nn.Module):
    """
    4-Layer Bidirectional GRU with Interleaved Decoupled Post-Norm Structural Injection.
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters
        self.input_dim = Config.INPUT_CHANNELS
        self.hidden_dim = Config.HIDDEN_DIM  # 384
        self.num_layers = Config.NUM_LAYERS  # 4
        self.kernel_size = Config.KERNEL_SIZE  # 3
        self.dropout_rate = Config.DROPOUT

        # Stem Filters fixed to 256 as per strategy
        self.stem_filters = 256

        # BiGRU Hidden Size
        # To maintain output dimension of 384, bidirectional GRU needs hidden_size = 192
        self.gru_hidden = self.hidden_dim // 2

        # 1. Convolutional Stem
        self.conv_stem = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.stem_filters,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.stem_act = nn.GELU()

        # 2. Deep Stabilized Backbone
        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            # Input dimension logic:
            # Layer 0: Takes stem output (256)
            # Layers 1-3: Takes previous block output (384)
            input_size = self.stem_filters if i == 0 else self.hidden_dim

            # BiGRU Layer
            gru = nn.GRU(
                input_size=input_size,
                hidden_size=self.gru_hidden,
                batch_first=True,
                bidirectional=True,
            )

            # Structural Interaction Module
            # Included in all blocks EXCEPT the final block
            interaction = None
            if i < self.num_layers - 1:
                interaction = StructuralInteractionModule(self.hidden_dim)

            self.layers.append(nn.ModuleList([gru, interaction]))

        self.dropout = nn.Dropout(self.dropout_rate)

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dim, Config.NUM_TARGETS)

    def forward(self, features, pair_indices, pair_mask):
        """
        Args:
            features: (B, L, 14)
            pair_indices: (B, L)
            pair_mask: (B, L, 1)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = features.transpose(1, 2)

        # Stem
        x = self.conv_stem(x)
        x = self.stem_act(x)

        # Permute back for GRU: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)

        # Backbone
        for gru, interaction in self.layers:
            # GRU Forward
            # Output shape: (B, L, 2 * gru_hidden) = (B, L, 384)
            x, _ = gru(x)

            # Interaction Module (if present)
            if interaction is not None:
                x = interaction(x, pair_indices, pair_mask)

            # Dropout applied after block
            x = self.dropout(x)

        # Output Head
        out = self.head(x)

        return out
