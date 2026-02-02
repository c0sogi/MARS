import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GLUDecoupledInteraction(nn.Module):
    """
    Stabilized GLU-Decoupled Interaction Module.

    Synthesizes Decoupled Gating, GLU Messages, and Bias-Driven Refinement.
    For unpaired bases, the message becomes a learnable bias vector (Loop Embedding).
    """

    def __init__(self, hidden_dim):
        super().__init__()

        # GLU Message Components (Decoupled: depends only on h_j)
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        # Wide Stabilized MLP Gate
        # Input: [h_i; h_j] -> Output: Gate score
        self.W_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm_gate = nn.LayerNorm(hidden_dim)
        self.W_out = nn.Linear(hidden_dim, hidden_dim)

        # Post-injection Normalization
        self.layer_norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, h, adj_indices, adj_mask):
        """
        Args:
            h: Hidden states (Batch, SeqLen, HiddenDim)
            adj_indices: Adjacency indices (Batch, SeqLen). Values in [0, SeqLen-1].
            adj_mask: Adjacency mask (Batch, SeqLen). 1.0 if paired, 0.0 if unpaired.
        """
        B, L, H = h.shape

        # 1. Gather h_j (Point-to-Point)
        # Expand indices to (B, L, H) to gather along the sequence dimension
        idx_expanded = adj_indices.unsqueeze(-1).expand(-1, -1, H)
        h_j = torch.gather(h, 1, idx_expanded)  # (B, L, H)

        # 2. Input Zero-Masking
        # If unpaired, force h_j = 0. This enables the bias-driven loop embedding.
        mask = adj_mask.unsqueeze(-1)  # (B, L, 1)
        h_j = h_j * mask

        # 3. GLU Message (Bias-Refined)
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # For unpaired bases (h_j=0), this becomes b_c * sigmoid(b_g)
        content = self.W_c(h_j)
        gate_signal = torch.sigmoid(self.W_g(h_j))
        m_ij = content * gate_signal

        # 4. Wide Stabilized MLP Gate
        # Concatenate h_i and h_j
        concat = torch.cat([h, h_j], dim=-1)  # (B, L, 2H)

        # Wide Projection -> LayerNorm -> GELU -> Sigmoid
        z_raw = self.W_in(concat)
        z_norm = self.layer_norm_gate(z_raw)
        z_act = F.gelu(z_norm)
        g_ij = torch.sigmoid(self.W_out(z_act))

        # 5. Injection
        h_res = h + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.layer_norm_out(h_res)

        return h_out


class RNARegressor(nn.Module):
    """
    High-Capacity Stabilized GLU-Decoupled BiGRU Architecture.

    Consists of:
    1. 1D Convolutional Stem
    2. 4-Layer BiGRU Backbone (768 units)
    3. Interleaved GLU-Decoupled Interaction Modules
    4. Linear Output Head
    """

    def __init__(self):
        super().__init__()

        # ==========================================
        # 1. Convolutional Stem
        # ==========================================
        self.conv = nn.Conv1d(
            in_channels=Config.INPUT_DIM,
            out_channels=Config.CONV_FILTERS,
            kernel_size=Config.CONV_KERNEL_SIZE,
            padding=Config.CONV_KERNEL_SIZE // 2,
        )
        self.act = nn.GELU()

        # ==========================================
        # 2. Backbone
        # ==========================================
        self.num_layers = Config.NUM_LAYERS
        # Hidden dim is doubled because of Bidirectional GRU
        self.hidden_dim = (
            Config.HIDDEN_DIM * 2 if Config.BIDIRECTIONAL else Config.HIDDEN_DIM
        )

        self.grus = nn.ModuleList()
        self.interactions = nn.ModuleList()
        self.dropout = nn.Dropout(Config.DROPOUT)

        input_dim = Config.CONV_FILTERS

        for i in range(self.num_layers):
            # BiGRU Layer
            self.grus.append(
                nn.GRU(
                    input_size=input_dim,
                    hidden_size=Config.HIDDEN_DIM,
                    batch_first=True,
                    bidirectional=Config.BIDIRECTIONAL,
                    num_layers=1,  # Stacked manually to interleave interactions
                )
            )

            # Update input dimension for next layer (output of BiGRU is hidden_dim)
            input_dim = self.hidden_dim

            # Add Interaction Module for all blocks except the final one
            if i < self.num_layers - 1:
                self.interactions.append(GLUDecoupledInteraction(self.hidden_dim))

        # ==========================================
        # 3. Output Head
        # ==========================================
        self.head = nn.Linear(self.hidden_dim, 5)

    def forward(self, inputs, adjacency_indices, adjacency_mask):
        """
        Args:
            inputs: (Batch, SeqLen, 14)
            adjacency_indices: (Batch, SeqLen)
            adjacency_mask: (Batch, SeqLen)
        """
        # 1. Convolutional Stem
        # Permute to (Batch, Channels, SeqLen) for Conv1d
        x = inputs.transpose(1, 2)
        x = self.conv(x)
        x = self.act(x)
        # Permute back to (Batch, SeqLen, Channels) for RNN
        x = x.transpose(1, 2)

        # 2. Backbone
        for i in range(self.num_layers):
            # Run BiGRU
            # x shape: (B, L, InputDim) -> (B, L, HiddenDim)
            x, _ = self.grus[i](x)

            # Apply Dropout
            x = self.dropout(x)

            # Apply Interaction (if present for this block)
            if i < len(self.interactions):
                x = self.interactions[i](x, adjacency_indices, adjacency_mask)

        # 3. Output Head
        out = self.head(x)
        return out
