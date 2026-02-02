import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DecoupledStructuralBlock(nn.Module):
    """
    Implements the Decoupled Channel-Gating logic with Bias-Driven Loop Refinement
    and Stabilized MLP Gating.
    """

    def __init__(self, input_dim):
        super(DecoupledStructuralBlock, self).__init__()

        # Message computation: m_ij = GELU(W_msg * h_j + b_msg)
        # When h_j is masked (0), this learns a bias embedding for loops.
        self.msg_proj = nn.Linear(input_dim, input_dim, bias=True)
        self.act = nn.GELU()

        # Stabilized MLP Gate
        # Input: [h_i; h_j] -> 2 * input_dim
        # We project to input_dim for the internal gate state
        self.gate_l1 = nn.Linear(input_dim * 2, input_dim)
        self.gate_norm = nn.LayerNorm(input_dim)
        self.gate_act = nn.GELU()
        self.gate_l2 = nn.Linear(input_dim, input_dim)

        # Post-Normalization
        self.out_norm = nn.LayerNorm(input_dim)

    def forward(self, x, bppm_indices):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Hidden_Dim)
            bppm_indices: Tensor of shape (Batch, Seq_Len) with indices of pairs.
                          -1 indicates unpaired.
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 1. Gather h_j (Point-to-Point)
        # Handle -1 indices by replacing with 0 temporarily, then masking
        # Create mask: 1 where paired, 0 where unpaired
        mask = (bppm_indices != -1).unsqueeze(-1).float()  # (B, L, 1)

        # Safe indices: replace -1 with 0 to avoid gather error
        safe_indices = bppm_indices.clone()
        safe_indices[safe_indices == -1] = 0

        # Expand indices for gather: (B, L, H)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)

        # Gather h_j
        h_j_raw = torch.gather(x, 1, gather_indices)

        # Apply Zero-Masking: Force h_j = 0 if unpaired
        h_j = h_j_raw * mask

        # 2. Decoupled Message: m_ij = GELU(W * h_j + b)
        # For unpaired (h_j=0), this becomes GELU(b), the loop embedding.
        m_ij = self.act(self.msg_proj(h_j))

        # 3. Stabilized MLP Gate
        # Joint Context z_raw_in
        z_raw_in = torch.cat([x, h_j], dim=-1)  # (B, L, 2*H)

        # Project -> Norm -> Act -> Project -> Sigmoid
        z_raw = self.gate_l1(z_raw_in)
        z_norm = self.gate_norm(z_raw)
        z_act = self.gate_act(z_norm)
        logits = self.gate_l2(z_act)
        g_ij = torch.sigmoid(logits)

        # 4. Injection: h_res = h_i + g_ij * m_ij
        h_res = x + g_ij * m_ij

        # 5. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class DeepStabilizedBiGRU(nn.Module):
    """
    4-Layer Bidirectional GRU with Interleaved Decoupled Post-Norm Structural Injection.
    """

    def __init__(self):
        super(DeepStabilizedBiGRU, self).__init__()

        # Dimensions
        self.input_channels = Config.INPUT_CHANNELS
        self.conv_filters = Config.CONV_FILTERS
        self.hidden_dim = Config.HIDDEN_DIM
        self.gru_output_dim = self.hidden_dim * 2  # BiGRU doubles dimension
        self.num_layers = Config.N_LAYERS
        self.num_targets = Config.NUM_TARGETS

        # 1. Convolutional Stem
        self.conv_stem = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_channels,
                out_channels=self.conv_filters,
                kernel_size=Config.CONV_KERNEL,
                padding=Config.CONV_KERNEL // 2,
            ),
            nn.GELU(),
        )

        # 2. Deep Backbone
        # We construct layers explicitly to handle the interaction module logic
        self.gru_layers = nn.ModuleList()
        self.interaction_modules = nn.ModuleList()

        current_dim = self.conv_filters

        for i in range(self.num_layers):
            # BiGRU Layer
            gru = nn.GRU(
                input_size=current_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.gru_layers.append(gru)

            # Update current dim to GRU output dim
            current_dim = self.gru_output_dim

            # Interaction Module (for all blocks except the final one)
            if i < self.num_layers - 1:
                self.interaction_modules.append(DecoupledStructuralBlock(current_dim))
            else:
                # Placeholder to keep indexing aligned if needed, though we won't use it
                self.interaction_modules.append(nn.Identity())

        # 3. Output Head
        self.head = nn.Linear(current_dim, self.num_targets)

        # Dropout
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x, bppm_indices):
        """
        Args:
            x: Input features (Batch, Seq_Len, Channels)
            bppm_indices: Pairing indices (Batch, Seq_Len)
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)

        # Stem
        x = self.conv_stem(x)

        # Permute back for GRU: (B, C, L) -> (B, L, C)
        x = x.permute(0, 2, 1)

        # Backbone
        for i in range(self.num_layers):
            # BiGRU
            x, _ = self.gru_layers[i](x)

            # Apply Dropout
            x = self.dropout(x)

            # Structural Interaction (except final block)
            if i < self.num_layers - 1:
                x = self.interaction_modules[i](x, bppm_indices)

        # Head
        out = self.head(x)

        return out
