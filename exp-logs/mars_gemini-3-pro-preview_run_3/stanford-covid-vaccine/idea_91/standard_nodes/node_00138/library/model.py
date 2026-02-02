import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedStem(nn.Module):
    """
    Dilated Residual Motif-Encoding Stem.
    Hierarchically aggregates local context using dilated convolutions.

    Structure:
    - Stage 1: Conv1D(k=3, d=1) -> GELU -> LayerNorm
    - Stage 2: Conv1D(k=3, d=2) -> GELU -> LayerNorm (Residual to Stage 1)
    - Stage 3: Conv1D(k=3, d=4) -> GELU -> LayerNorm (Residual to Stage 2)
    """

    def __init__(self, input_dim, filters, kernel_size, dilations):
        super(DilatedStem, self).__init__()
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Stage 1
        self.stages.append(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=filters,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * dilations[0] // 2,
                dilation=dilations[0],
            )
        )
        self.norms.append(nn.LayerNorm(filters))

        # Stage 2
        self.stages.append(
            nn.Conv1d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * dilations[1] // 2,
                dilation=dilations[1],
            )
        )
        self.norms.append(nn.LayerNorm(filters))

        # Stage 3
        self.stages.append(
            nn.Conv1d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * dilations[2] // 2,
                dilation=dilations[2],
            )
        )
        self.norms.append(nn.LayerNorm(filters))

    def forward(self, x):
        # x: (N, L, input_dim)

        # Stage 1
        # Permute for Conv1d: (N, C, L)
        out = x.permute(0, 2, 1)
        out = self.stages[0](out)
        out = out.permute(0, 2, 1)  # Back to (N, L, C)
        out = F.gelu(out)
        out = self.norms[0](out)

        x_prev = out

        # Stage 2 (Residual)
        out = x_prev.permute(0, 2, 1)
        out = self.stages[1](out)
        out = out.permute(0, 2, 1)
        out = F.gelu(out)
        out = self.norms[1](out)
        out = out + x_prev  # Residual

        x_prev = out

        # Stage 3 (Residual)
        out = x_prev.permute(0, 2, 1)
        out = self.stages[2](out)
        out = out.permute(0, 2, 1)
        out = F.gelu(out)
        out = self.norms[2](out)
        out = out + x_prev  # Residual

        return out


class InteractionModule(nn.Module):
    """
    Stabilized GLU-Decoupled Interaction Module.
    Handles long-range interactions via point-to-point gathering and gating.
    """

    def __init__(self, dim):
        super(InteractionModule, self).__init__()

        # GLU Message components
        self.w_c = nn.Linear(dim, dim)
        self.w_g = nn.Linear(dim, dim)

        # Wide Stabilized MLP Gate
        # Input is [h_i; h_j] -> 2 * dim
        self.w_in = nn.Linear(dim * 2, dim)
        self.gate_norm = nn.LayerNorm(dim)
        self.w_out = nn.Linear(dim, dim)

        # Post-Normalization
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, h, pair_indices, pair_masks):
        """
        Args:
            h: (N, L, D) Hidden states
            pair_indices: (N, L) Indices of paired bases
            pair_masks: (N, L) 1.0 if paired, 0.0 if unpaired
        """
        batch_size, seq_len, dim = h.shape

        # 1. Gather h_j
        # Create batch indices for gathering
        # batch_indices: (N, L)
        batch_indices = (
            torch.arange(batch_size, device=h.device).unsqueeze(1).expand(-1, seq_len)
        )

        # Gather: h[b, pair_indices[b, i], :]
        # pair_indices has 0 for unpaired (handled by mask later), so this is safe
        h_j = h[batch_indices, pair_indices]  # (N, L, D)

        # 2. Input Zero-Masking
        # If unpaired, force h_j = 0
        mask = pair_masks.unsqueeze(-1)  # (N, L, 1)
        h_j = h_j * mask

        # 3. GLU Message (Bias-Refined)
        # m_ij = (W_c h_j + b_c) * sigmoid(W_g h_j + b_g)
        # For unpaired (h_j=0), this becomes b_c * sigmoid(b_g)
        m_ij = self.w_c(h_j) * torch.sigmoid(self.w_g(h_j))

        # 4. Wide Stabilized MLP Gate
        # Concatenate h_i and h_j
        z_raw = torch.cat([h, h_j], dim=-1)  # (N, L, 2D)

        # Wide Projection
        z_proj = self.w_in(z_raw)  # (N, L, D)

        # Internal Normalization & Activation
        z_act = F.gelu(self.gate_norm(z_proj))

        # Gate Output (No Logit Norm)
        g_ij = torch.sigmoid(self.w_out(z_act))

        # 5. Injection
        # Additive injection
        h_res = h + g_ij * m_ij

        # 6. Post-Normalization
        h_out = self.out_norm(h_res)

        return h_out


class RNAModel(nn.Module):
    """
    High-Capacity BiGRU with Dilated Motif-Encoding Stem.
    """

    def __init__(self):
        super(RNAModel, self).__init__()

        # 1. Dilated Stem
        self.stem = DilatedStem(
            input_dim=Config.INPUT_DIM,
            filters=Config.STEM_FILTERS,
            kernel_size=Config.STEM_KERNEL_SIZE,
            dilations=Config.STEM_DILATIONS,
        )

        # 2. Backbone
        self.rnn_layers = nn.ModuleList()
        self.interaction_layers = nn.ModuleList()

        # Layer 1 Input comes from Stem (dim = STEM_FILTERS)
        # Subsequent layers input comes from Interaction Module (dim = 2 * RNN_HIDDEN_DIM)

        rnn_input_dim = Config.STEM_FILTERS
        rnn_hidden_dim = Config.RNN_HIDDEN_DIM
        total_dim = rnn_hidden_dim * 2 if Config.BIDIRECTIONAL else rnn_hidden_dim

        for i in range(Config.RNN_LAYERS):
            # BiGRU Layer
            self.rnn_layers.append(
                nn.GRU(
                    input_size=rnn_input_dim,
                    hidden_size=rnn_hidden_dim,
                    batch_first=True,
                    bidirectional=Config.BIDIRECTIONAL,
                    dropout=Config.DROPOUT if i < Config.RNN_LAYERS - 1 else 0,
                )
            )

            # Interaction Module
            self.interaction_layers.append(InteractionModule(dim=total_dim))

            # Update input dim for next layer
            rnn_input_dim = total_dim

        # 3. Output Head
        self.head = nn.Linear(total_dim, 5)

    def forward(self, features, pair_indices, pair_masks):
        """
        Args:
            features: (N, 107, 14)
            pair_indices: (N, 107)
            pair_masks: (N, 107)
        Returns:
            (N, 107, 5)
        """
        # Pass through Stem
        x = self.stem(features)  # (N, 107, STEM_FILTERS)

        # Pass through Backbone Layers
        for rnn, interaction in zip(self.rnn_layers, self.interaction_layers):
            # RNN
            # GRU returns (output, h_n). We only need output.
            x, _ = rnn(x)  # (N, 107, 768)

            # Interaction
            x = interaction(x, pair_indices, pair_masks)  # (N, 107, 768)

        # Head
        out = self.head(x)  # (N, 107, 5)

        return out
