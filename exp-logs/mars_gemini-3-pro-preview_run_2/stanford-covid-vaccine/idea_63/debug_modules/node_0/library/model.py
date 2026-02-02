import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    SEQ_LENGTH,
    GROWTH_RATE,
    LATENT_DIM,
    KERNEL_SIZE,
    DILATIONS,
    DROPOUT,
    FEEDBACK_GROWTH_RATE,
    FEEDBACK_CHANNELS,
    RNN_HIDDEN_SIZE,
    NUM_TARGETS,
)


class LayerNormChannels(nn.Module):
    """
    Applies LayerNorm along the channel dimension for (N, C, L) tensors.
    """

    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (N, C, L) -> (N, L, C) -> LN -> (N, C, L)
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        return x


class HybridStem(nn.Module):
    """
    Hybrid Input Stem:
    Branch A: Raw Identity (Input)
    Branch B: Spatial Context (Conv -> LN -> SiLU)
    Output: Concat(A, B)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch_b = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=KERNEL_SIZE,
                padding=KERNEL_SIZE // 2,
            ),
            LayerNormChannels(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        # x: (N, C, L)
        context = self.branch_b(x)
        return torch.cat([x, context], dim=1)


class PostActDenseBlock(nn.Module):
    """
    Post-Activation Dense Block:
    Input (Dense) -> Conv(3x3, d) -> LN -> SiLU -> Conv(1x1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                growth_rate,
                kernel_size=KERNEL_SIZE,
                padding=(KERNEL_SIZE // 2) * dilation,
                dilation=dilation,
            ),
            LayerNormChannels(growth_rate),
            nn.SiLU(),
            nn.Conv1d(growth_rate, growth_rate, kernel_size=1),
            LayerNormChannels(growth_rate),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class DenseDilatedTCN(nn.Module):
    """
    Backbone: Stack of Dense Blocks with increasing dilation.
    """

    def __init__(self, in_channels, growth_rate, dilations, out_dim):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for d in dilations:
            blk = PostActDenseBlock(current_channels, growth_rate, d)
            self.blocks.append(blk)
            current_channels += growth_rate

        self.projection = nn.Conv1d(current_channels, out_dim, kernel_size=1)

    def forward(self, x):
        # Dense connection management
        features = [x]
        for block in self.blocks:
            # Concatenate all previous features
            dense_input = torch.cat(features, dim=1)
            out = block(dense_input)
            features.append(out)

        # Final projection on all accumulated features
        total_features = torch.cat(features, dim=1)
        return self.projection(total_features)


class FeedbackModule(nn.Module):
    """
    Global-Context Pure-Feedback Module.
    Input: Recycled Predictions (5 channels).
    Logic: Mask unscored channels -> Spatial Stem -> Lightweight Dense TCN.
    """

    def __init__(self):
        super().__init__()
        # Unscored indices in [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
        # Scored: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
        # Unscored: 2 (deg_pH10), 4 (deg_50C)
        self.register_buffer(
            "channel_mask",
            torch.tensor([1, 1, 0, 1, 0], dtype=torch.float32).view(1, 5, 1),
        )

        # Spatial Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                5,
                FEEDBACK_GROWTH_RATE,
                kernel_size=KERNEL_SIZE,
                padding=KERNEL_SIZE // 2,
            ),
            LayerNormChannels(FEEDBACK_GROWTH_RATE),
            nn.SiLU(),
        )

        # Lightweight Backbone
        # Use same dilations but smaller growth rate
        self.backbone = DenseDilatedTCN(
            in_channels=FEEDBACK_GROWTH_RATE,
            growth_rate=FEEDBACK_GROWTH_RATE,
            dilations=DILATIONS,
            out_dim=FEEDBACK_CHANNELS,
        )

    def forward(self, y_prev):
        # y_prev: (N, L, 5) -> permute to (N, 5, L)
        x = y_prev.permute(0, 2, 1)

        # Channel Masking
        x = x * self.channel_mask

        # Stem
        x = self.stem(x)

        # Backbone
        out = self.backbone(x)  # (N, FEEDBACK_CHANNELS, L)
        return out


class InteractionHead(nn.Module):
    """
    Interaction & Aggregation:
    Augmented Gather (Self + Partner) -> Null Masking -> Fusion -> BiGRU -> Linear
    """

    def __init__(self, input_dim):
        super().__init__()
        # Input dim is (Latent + Feedback)
        # We concat Self (dim) + Partner (dim) = 2 * dim
        self.fusion_dim = input_dim * 2

        self.gru = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=RNN_HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(RNN_HIDDEN_SIZE * 2, NUM_TARGETS)

    def forward(self, z, e_fb, partner_indices):
        """
        z: (N, Latent, L)
        e_fb: (N, Feedback, L)
        partner_indices: (N, L) with -1 for unpaired
        """
        # 1. Concatenate Z and E_fb -> Combined Representation
        # Shape: (N, C_total, L)
        combined = torch.cat([z, e_fb], dim=1)
        N, C, L = combined.shape

        # 2. Prepare for Gather
        # partner_indices has -1. We replace -1 with 0 for valid gather, then mask result.
        # We need indices in shape (N, C, L) to gather along dim 2 (Length)

        # Mask for unpaired bases
        unpaired_mask = partner_indices == -1  # (N, L)

        # Safe indices: replace -1 with 0
        safe_indices = partner_indices.clone()
        safe_indices[unpaired_mask] = 0

        # Expand indices for gathering: (N, L) -> (N, 1, L) -> (N, C, L)
        gather_indices = safe_indices.unsqueeze(1).expand(-1, C, -1)

        # 3. Gather Partner Vectors
        partner_vecs = torch.gather(combined, 2, gather_indices)  # (N, C, L)

        # 4. Null-Masking
        # Apply zero mask where partner was -1
        # mask shape (N, L) -> (N, 1, L) -> (N, C, L)
        mask_expanded = unpaired_mask.unsqueeze(1).expand(-1, C, -1)
        partner_vecs = partner_vecs.masked_fill(mask_expanded, 0.0)

        # 5. Fusion
        # Self: combined, Partner: partner_vecs
        # Permute to (N, L, C) for RNN
        self_vecs_t = combined.permute(0, 2, 1)
        partner_vecs_t = partner_vecs.permute(0, 2, 1)

        fused = torch.cat([self_vecs_t, partner_vecs_t], dim=2)  # (N, L, 2*C)

        # 6. Aggregation (BiGRU)
        gru_out, _ = self.gru(fused)  # (N, L, 2*Hidden)

        # 7. Output
        logits = self.classifier(gru_out)  # (N, L, 5)

        return logits


class HS_GFDN(nn.Module):
    """
    Hybrid-Stem Global-Feedback Dense Network.
    """

    def __init__(self):
        super().__init__()

        # Input Channels: 18 (4 Seq + 3 Struct + 7 Loop + 4 Partner)
        self.input_channels = 18

        # 1. Hybrid Stem
        # Branch B produces GROWTH_RATE channels
        self.stem = HybridStem(self.input_channels, GROWTH_RATE)

        # Stem Output Channels = Input (18) + Context (GROWTH_RATE)
        self.stem_out_channels = self.input_channels + GROWTH_RATE

        # 2. Main Backbone
        self.backbone = DenseDilatedTCN(
            in_channels=self.stem_out_channels,
            growth_rate=GROWTH_RATE,
            dilations=DILATIONS,
            out_dim=LATENT_DIM,
        )

        # 3. Feedback Module
        self.feedback_module = FeedbackModule()

        # 4. Interaction Head
        # Input to head is Latent (64) + Feedback (32)
        self.head = InteractionHead(LATENT_DIM + FEEDBACK_CHANNELS)

    def forward(self, inputs, partner_indices, feedback=None):
        """
        inputs: (N, L, 18)
        partner_indices: (N, L)
        feedback: (N, L, 5) or None
        """
        # Permute inputs for Conv1d: (N, 18, L)
        x = inputs.permute(0, 2, 1)

        # 1. Static Path
        x = self.stem(x)
        z = self.backbone(x)  # (N, Latent, L)

        # 2. Feedback Path
        if feedback is None:
            # First pass: Feedback is zero
            N, _, L = z.shape
            e_fb = torch.zeros(
                (N, FEEDBACK_CHANNELS, L), device=z.device, dtype=z.dtype
            )
        else:
            # Recycled pass
            e_fb = self.feedback_module(feedback)  # (N, Feedback, L)

        # 3. Interaction & Output
        y_hat = self.head(z, e_fb, partner_indices)  # (N, L, 5)

        return y_hat
