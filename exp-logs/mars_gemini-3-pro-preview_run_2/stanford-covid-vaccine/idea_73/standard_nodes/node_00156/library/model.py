import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class PermuteLayerNorm(nn.Module):
    """
    Layer Normalization for (N, C, L) tensors.
    Permutes to (N, L, C) for LN, then back to (N, C, L).
    """

    def __init__(self, normalized_shape):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = x.transpose(1, 2)
        return x


class AugmentedStem(nn.Module):
    """
    Parallel Input Stem:
    - Branch A: Raw Identity (preserved)
    - Branch B: Spatial Context (Conv1d -> LN -> SiLU)
    """

    def __init__(self, in_channels, context_channels=32):
        super().__init__()
        self.branch_b = nn.Sequential(
            nn.Conv1d(in_channels, context_channels, kernel_size=3, padding=1),
            PermuteLayerNorm(context_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        # x: (N, C, L)
        out_a = x
        out_b = self.branch_b(x)
        return torch.cat([out_a, out_b], dim=1)


class DenseDilatedBlock(nn.Module):
    """
    Post-Activation Dilated Convolution Block.
    Structure: Conv(k=3, d=D) -> LN -> SiLU -> Conv(k=1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                growth_rate,
                kernel_size,
                padding=dilation,
                dilation=dilation,
            ),
            PermuteLayerNorm(growth_rate),
            nn.SiLU(),
            nn.Conv1d(growth_rate, growth_rate, 1),
            PermuteLayerNorm(growth_rate),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DenseBackbone(nn.Module):
    """
    Stack of DenseDilatedBlocks with Dense Connectivity.
    """

    def __init__(
        self, in_channels, growth_rate, kernel_size, dilations, dropout, out_dim
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_dim = in_channels

        for d in dilations:
            block = DenseDilatedBlock(current_dim, growth_rate, kernel_size, d, dropout)
            self.blocks.append(block)
            current_dim += growth_rate

        self.project = nn.Conv1d(current_dim, out_dim, 1)

    def forward(self, x):
        # Dense connectivity: concatenate block output to input for next layers
        for block in self.blocks:
            out = block(x)
            x = torch.cat([x, out], dim=1)
        return self.project(x)


class FeedbackModule(nn.Module):
    """
    Lightweight Dense TCN for processing recycled predictions.
    Includes channel masking for unscored targets.
    """

    def __init__(
        self, in_channels, growth_rate, kernel_size, dilations, dropout, out_dim
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_dim = in_channels

        for d in dilations:
            block = DenseDilatedBlock(current_dim, growth_rate, kernel_size, d, dropout)
            self.blocks.append(block)
            current_dim += growth_rate

        self.project = nn.Conv1d(current_dim, out_dim, 1)

    def forward(self, x):
        # x: (N, 5, L)
        # Mask unscored channels (indices 2 and 4) if Configured
        # Scored indices are [0, 1, 3]
        if Config.CHANNEL_MASKING:
            mask = torch.zeros_like(x)
            mask[:, Config.SCORED_INDICES, :] = 1.0
            x = x * mask

        for block in self.blocks:
            out = block(x)
            x = torch.cat([x, out], dim=1)
        return self.project(x)


class AS_DFRN(nn.Module):
    """
    Augmented-Stem Dense-Feedback Recurrent Network.
    """

    def __init__(self):
        super().__init__()

        # --- Dimensions ---
        # Seq(4) + Struct(3) + Loop(7) + Partner(4) = 18
        self.raw_in_dim = 18
        self.stem_context_dim = 32
        self.backbone_in_dim = self.raw_in_dim + self.stem_context_dim

        self.backbone_growth = Config.GROWTH_RATE
        self.feedback_growth = Config.FEEDBACK_GROWTH_RATE
        self.latent_dim = Config.DIM
        self.feedback_dim = 32

        # --- Modules ---

        # 1. Augmented Stem
        self.stem = AugmentedStem(self.raw_in_dim, self.stem_context_dim)

        # 2. Static Backbone
        self.backbone = DenseBackbone(
            in_channels=self.backbone_in_dim,
            growth_rate=self.backbone_growth,
            kernel_size=Config.KERNEL_SIZE,
            dilations=Config.DILATIONS,
            dropout=Config.DROPOUT,
            out_dim=self.latent_dim,
        )

        # 3. Feedback Module
        self.feedback_net = FeedbackModule(
            in_channels=Config.NUM_TARGETS,
            growth_rate=self.feedback_growth,
            kernel_size=Config.KERNEL_SIZE,
            dilations=Config.DILATIONS,
            dropout=Config.DROPOUT,
            out_dim=self.feedback_dim,
        )

        # 4. Interaction & Aggregation
        # Input to interaction: Z (64) + E_fb (32) = 96
        self.inter_dim = self.latent_dim + self.feedback_dim

        # GRU Input: Self(96) + Partner(96) = 192
        self.gru_in_dim = self.inter_dim * 2
        self.gru_hidden = Config.DIM  # 64

        self.gru = nn.GRU(
            input_size=self.gru_in_dim,
            hidden_size=self.gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Head: 2 * hidden -> 5 targets
        self.head = nn.Linear(self.gru_hidden * 2, Config.NUM_TARGETS)

    def forward(self, x, pair_indices):
        # x: (N, L, 18) -> Permute to (N, 18, L) for Conv1d
        x = x.transpose(1, 2)

        # 1. Stem & Backbone (Static)
        x_aug = self.stem(x)  # (N, 50, L)
        z = self.backbone(x_aug)  # (N, 64, L)
        z = z.transpose(1, 2)  # (N, L, 64)

        batch_size, seq_len, _ = z.shape

        # --- Pass 1: Zero Feedback ---
        y_prev = torch.zeros(batch_size, Config.NUM_TARGETS, seq_len, device=x.device)

        e_fb_1 = self.feedback_net(y_prev)  # (N, 32, L)
        e_fb_1 = e_fb_1.transpose(1, 2)  # (N, L, 32)

        y_pred_1 = self._interact_and_predict(z, e_fb_1, pair_indices)

        # --- Pass 2: Dense Feedback ---
        # Detach gradients from Pass 1 to treat it as a fixed signal
        y_prev_2 = y_pred_1.detach().transpose(1, 2)  # (N, 5, L)

        e_fb_2 = self.feedback_net(y_prev_2)
        e_fb_2 = e_fb_2.transpose(1, 2)

        y_pred_2 = self._interact_and_predict(z, e_fb_2, pair_indices)

        return y_pred_1, y_pred_2

    def _interact_and_predict(self, z, e_fb, pair_indices):
        """
        Fuses static (z) and dynamic (e_fb) features, gathers partner features,
        and runs the global RNN aggregation.
        """
        # Concatenate self features
        h_self = torch.cat([z, e_fb], dim=-1)  # (N, L, 96)

        batch_size, seq_len, dim = h_self.shape

        # --- Partner Gathering ---
        # pair_indices: (N, L). -1 indicates unpaired.
        # Replace -1 with 0 for safe gathering, then mask result.
        valid_mask = (pair_indices != -1).unsqueeze(-1)  # (N, L, 1)
        safe_indices = pair_indices.clone()
        safe_indices[pair_indices == -1] = 0

        # Expand indices for gather: (N, L, D)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, dim)

        # Gather partner vectors
        h_partner = torch.gather(h_self, 1, gather_indices)

        # Apply mask to zero out unpaired partners
        h_partner = h_partner * valid_mask.float()

        # --- Fusion & Aggregation ---
        # Concatenate Self and Partner
        gru_in = torch.cat([h_self, h_partner], dim=-1)  # (N, L, 192)

        # Bidirectional GRU
        gru_out, _ = self.gru(gru_in)  # (N, L, 128)

        # Prediction Head
        logits = self.head(gru_out)  # (N, L, 5)

        return logits
