import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StructuralInteractionModule(nn.Module):
    """
    Implements the Decoupled Structural Interaction with Strict Output Masking.

    Logic:
    1. Gather neighbor states h_j based on pair_indices.
    2. Compute Message m_ij derived SOLELY from neighbor h_j (Decoupled).
    3. Compute Gate g_ij derived from joint context [h_i; h_j].
    4. Apply Update u_ij = (g_ij * m_ij) * M_pair (Strict Masking).
    5. Residual Connection + LayerNorm.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # Message generation: depends only on neighbor h_j
        self.msg_net = nn.Linear(hidden_dim, hidden_dim)

        # Gating: depends on h_i and h_j
        self.gate_net = nn.Linear(hidden_dim * 2, hidden_dim)

        # Post-Normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x (torch.Tensor): Hidden states (B, L, D).
            pair_indices (torch.Tensor): Indices of paired bases (B, L).
            pair_mask (torch.Tensor): Mask indicating paired status (B, L). 1.0 if paired, 0.0 otherwise.

        Returns:
            torch.Tensor: Updated hidden states (B, L, D).
        """
        B, L, D = x.shape

        # 1. Gather neighbor states h_j
        # pair_indices contains the index j for each i.
        # We need to expand indices to match the hidden dimension for gathering.
        idx_expanded = pair_indices.unsqueeze(-1).expand(-1, -1, D)

        # Gather: out[b, i, k] = input[b, index[b, i, k], k]
        x_neighbor = torch.gather(x, 1, idx_expanded)

        # 2. Decoupled Message: m_ij = GELU(W_msg * h_j)
        # Note: We do NOT concatenate h_i here, forcing the branch to learn the structural delta.
        m = F.gelu(self.msg_net(x_neighbor))

        # 3. Channel-Wise Gating: g_ij = sigmoid(W_gate * [h_i; h_j])
        # Gating uses the full context.
        cat_input = torch.cat([x, x_neighbor], dim=-1)
        g = torch.sigmoid(self.gate_net(cat_input))

        # 4. Strict Output Masking
        # u_ij = (g * m) * mask
        # Ensure mask is broadcastable: (B, L) -> (B, L, 1)
        mask_expanded = pair_mask.unsqueeze(-1)
        u = (g * m) * mask_expanded

        # 5. Residual + Post-Norm
        # Unpaired bases receive u=0, so they just pass through the residual (and get normalized).
        out = self.norm(x + u)

        return out


class DeepDecoupledModel(nn.Module):
    """
    Deep Decoupled Post-Norm BiGRU with Strict Output Masking.

    Architecture:
    - Input: One-hot encoded features (Seq, Struct, Loop).
    - Stem: 1D Convolution.
    - Backbone: 4 Blocks of BiGRU.
      - Blocks 1-3: BiGRU -> StructuralInteractionModule.
      - Block 4: BiGRU.
    - Head: Linear projection to targets.
    """

    def __init__(self):
        super().__init__()
        self.config = Config()

        # Dimensions
        self.num_features = self.config.num_features  # 14
        self.num_targets = self.config.num_targets  # 5

        # Architecture Hyperparameters
        self.conv_filters = self.config.conv_filters  # 256
        self.conv_kernel = self.config.conv_kernel  # 3
        self.hidden_dim = self.config.hidden_dim  # 384
        self.num_layers = self.config.num_layers  # 4
        self.dropout_p = self.config.dropout  # 0.1

        # 1. Convolutional Stem
        # Projects sparse inputs to dense embedding and aggregates local k-mers.
        # Padding is kernel // 2 to maintain sequence length.
        self.stem_conv = nn.Conv1d(
            in_channels=self.num_features,
            out_channels=self.conv_filters,
            kernel_size=self.conv_kernel,
            padding=self.conv_kernel // 2,
        )

        # 2. Deep Backbone
        # BiGRU output dimension is hidden_dim * 2 (because bidirectional)
        self.gru_output_dim = self.hidden_dim * 2

        self.grus = nn.ModuleList()
        self.interactions = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(self.num_layers):
            # First layer takes conv output, subsequent layers take previous GRU output
            input_dim = self.conv_filters if i == 0 else self.gru_output_dim

            # BiGRU Layer
            gru = nn.GRU(
                input_size=input_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.grus.append(gru)
            self.dropouts.append(nn.Dropout(self.dropout_p))

            # Interaction Module
            # Applied after blocks 0, 1, 2. Not applied after the final block (3).
            if i < self.num_layers - 1:
                self.interactions.append(
                    StructuralInteractionModule(self.gru_output_dim)
                )

        # 3. Output Head
        self.head = nn.Linear(self.gru_output_dim, self.num_targets)

    def forward(self, x, pair_indices, pair_mask):
        """
        Args:
            x (torch.Tensor): Input features (B, L, 14).
            pair_indices (torch.Tensor): (B, L).
            pair_mask (torch.Tensor): (B, L).

        Returns:
            torch.Tensor: Predictions (B, L, 5).
        """
        # Permute for Conv1d: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)

        # Stem
        x = self.stem_conv(x)
        x = F.gelu(x)

        # Permute back for GRU: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)

        # Backbone
        for i in range(self.num_layers):
            # Apply BiGRU
            # gru returns (out, h_n), we only need out
            x, _ = self.grus[i](x)

            # Apply Dropout
            x = self.dropouts[i](x)

            # Apply Structural Interaction (if not the last layer)
            if i < len(self.interactions):
                x = self.interactions[i](x, pair_indices, pair_mask)

        # Head
        out = self.head(x)

        return out
