import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualStem(nn.Module):
    """
    Dilated Residual Convolutional Stem.

    Hierarchically aggregates local context using a 3-stage dilated residual network.
    This allows the model to explicitly encode local secondary structure motifs
    (e.g., hairpins, internal loops) into the latent space before temporal processing.

    Architecture:
        - Stage 1: Conv1D(d=1) -> GELU -> LayerNorm
        - Stage 2: Conv1D(d=2) -> GELU -> LayerNorm (Residual + Stage 1)
        - Stage 3: Conv1D(d=4) -> GELU -> LayerNorm (Residual + Stage 2)
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        filters=Config.STEM_FILTERS,
        kernel_size=Config.STEM_KERNEL_SIZE,
        dilations=Config.STEM_DILATIONS,
    ):
        super().__init__()

        # Stage 1: Projection + Conv (d=1)
        # Handles the dimension change from input_dim (14) to filters (768)
        # Padding ensures sequence length is preserved
        pad1 = dilations[0] * (kernel_size - 1) // 2
        self.stage1_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=pad1,
            dilation=dilations[0],
        )
        self.stage1_norm = nn.LayerNorm(filters)

        # Stages 2 & 3: Residual Blocks with increasing dilation
        self.residual_stages = nn.ModuleList()
        for d in dilations[1:]:
            pad = d * (kernel_size - 1) // 2
            stage = nn.ModuleDict(
                {
                    "conv": nn.Conv1d(
                        in_channels=filters,
                        out_channels=filters,
                        kernel_size=kernel_size,
                        padding=pad,
                        dilation=d,
                    ),
                    "norm": nn.LayerNorm(filters),
                }
            )
            self.residual_stages.append(stage)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (N, L, input_dim).

        Returns:
            torch.Tensor: Encoded motifs of shape (N, L, filters).
        """
        # Transpose for Conv1d: (N, L, C) -> (N, C, L)
        x_in = x.transpose(1, 2)

        # --- Stage 1 ---
        out = self.stage1_conv(x_in)  # (N, filters, L)
        out = out.transpose(1, 2)  # (N, L, filters) for LayerNorm
        out = F.gelu(out)
        out = self.stage1_norm(out)

        # --- Stages 2 & 3 (Residual) ---
        for stage in self.residual_stages:
            residual = out

            # Conv Block
            out_conv = out.transpose(1, 2)  # (N, filters, L)
            out_conv = stage["conv"](out_conv)
            out_conv = out_conv.transpose(1, 2)  # (N, L, filters)

            out_conv = F.gelu(out_conv)
            out_conv = stage["norm"](out_conv)

            # Residual Connection
            out = out_conv + residual

        return out


class StabilizedGLUInteraction(nn.Module):
    """
    Stabilized GLU-Decoupled Interaction Module.

    Handles global folding constraints by gathering features from paired bases.

    Key Features:
        - Point-to-Point Gather: Retrieves h_j for every i paired with j.
        - Zero-Masking: Explicitly forces h_j=0 if i is unpaired.
        - GLU Message (Bias-Refined): m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g).
          For unpaired bases, this becomes a learnable bias vector representing 'unpaired' state.
        - Wide Stabilized MLP Gate: Projects [h_i; h_j] to wide dimension, normalizes internally,
          and produces a gating value.
        - Post-Normalization: Stabilizes the residual injection for deep stacks.
    """

    def __init__(self, hidden_dim=Config.RNN_HIDDEN_DIM * 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- GLU Message Components ---
        # Decoupled: Only depends on h_j
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        # --- Wide Stabilized MLP Gate ---
        # Input: [h_i; h_j] -> size 2 * hidden_dim
        self.gate_W_in = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_norm = nn.LayerNorm(hidden_dim)  # Internal Normalization
        self.gate_W_out = nn.Linear(hidden_dim, hidden_dim)

        # --- Post-Normalization ---
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adjacency):
        """
        Args:
            x (torch.Tensor): Sequence features (N, L, hidden_dim).
            adjacency (torch.Tensor): Pairing indices (N, L). -1 indicates unpaired.

        Returns:
            torch.Tensor: Updated features (N, L, hidden_dim).
        """
        B, L, C = x.shape

        # 1. Gather Paired Features (h_j)
        # --------------------------------
        # Prepare indices: replace -1 with 0 to allow gather (will mask later)
        adj_indices = adjacency.clone()
        mask = adj_indices != -1  # (N, L)
        adj_indices[~mask] = 0

        # Expand indices for gathering across feature dimension
        # gather_indices: (N, L, C)
        gather_indices = adj_indices.unsqueeze(-1).expand(-1, -1, C)

        # Gather: h_j[i] = x[adj[i]]
        h_j = torch.gather(x, 1, gather_indices)

        # Zero-Masking: Force h_j = 0 where unpaired
        mask_expanded = mask.unsqueeze(-1).float()  # (N, L, 1)
        h_j = h_j * mask_expanded

        # 2. GLU Message Calculation
        # --------------------------------
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # Note: When h_j is 0 (unpaired), this computes bias_c * sigmoid(bias_g),
        # effectively learning a specific embedding for unpaired/loop regions.
        val_c = self.W_c(h_j)
        val_g = torch.sigmoid(self.W_g(h_j))
        m_ij = val_c * val_g

        # 3. Wide Stabilized MLP Gate
        # --------------------------------
        # Concatenate [h_i; h_j]
        cat_input = torch.cat([x, h_j], dim=-1)  # (N, L, 2*C)

        # Wide Projection -> Internal Norm -> GELU -> Projection -> Sigmoid
        z_raw = self.gate_W_in(cat_input)
        z_norm = self.gate_norm(z_raw)
        z_act = F.gelu(z_norm)
        g_ij = torch.sigmoid(self.gate_W_out(z_act))

        # 4. Injection & Post-Normalization
        # --------------------------------
        # Residual Injection
        h_res = x + g_ij * m_ij

        # Post-Norm
        h_out = self.out_norm(h_res)

        return h_out
