import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    SEQ_LEN,
    NUM_TARGETS,
    HIDDEN_DIM,
    GROWTH_RATE,
    KERNEL_SIZE,
    DILATIONS,
    DROPOUT,
    FEEDBACK_GROWTH_RATE,
    FEEDBACK_LATENT_DIM,
)


class HybridInputStem(nn.Module):
    """
    Combines raw input features (Identity) with spatially processed features (Context).
    Branch A: Identity (Raw One-Hot)
    Branch B: Context (Conv3 -> LayerNorm -> SiLU)
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.branch_b_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.branch_b_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, C, L)

        # Branch A: Identity
        branch_a = x

        # Branch B: Context
        branch_b = self.branch_b_conv(x)
        # Permute for LayerNorm: (B, C, L) -> (B, L, C)
        branch_b = branch_b.permute(0, 2, 1)
        branch_b = self.branch_b_norm(branch_b)
        branch_b = self.act(branch_b)
        # Permute back: (B, L, C) -> (B, C, L)
        branch_b = branch_b.permute(0, 2, 1)

        # Concatenate along channel dimension
        out = torch.cat([branch_a, branch_b], dim=1)
        return out


class DenseDilatedBlock(nn.Module):
    """
    Post-Activation Dense Block with Decoupled Spatial/Channel Mixing.
    Structure: Conv(k=3, d=D) -> LN -> SiLU -> Conv(k=1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super().__init__()
        # Spatial Mixing (Dilated Conv)
        self.conv1 = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.norm1 = nn.LayerNorm(growth_rate)
        self.act1 = nn.SiLU()

        # Channel Mixing (Pointwise Conv)
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.norm2 = nn.LayerNorm(growth_rate)
        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C_in, L)

        out = self.conv1(x)
        out = out.permute(0, 2, 1)
        out = self.norm1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)

        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        out = self.norm2(out)
        out = self.act2(out)
        out = out.permute(0, 2, 1)

        out = self.dropout(out)
        return out


class DenseBackbone(nn.Module):
    """
    Stack of DenseDilatedBlocks.
    Uses dense connections (concatenating all previous block outputs).
    """

    def __init__(self, in_channels, growth_rate, dilations, dropout, latent_dim):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_dim = in_channels

        for d in dilations:
            block = DenseDilatedBlock(current_dim, growth_rate, KERNEL_SIZE, d, dropout)
            self.blocks.append(block)
            current_dim += growth_rate

        # Final projection to latent dimension
        self.project = nn.Conv1d(current_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, C, L)
        features = [x]

        for block in self.blocks:
            # Concatenate all previous features
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Final concatenation of all levels
        final_concat = torch.cat(features, dim=1)
        z = self.project(final_concat)
        return z


class FeedbackModule(nn.Module):
    """
    Processes recycled predictions.
    Includes a Spatial Stem and a Lightweight Dense Backbone.
    """

    def __init__(self, input_dim, growth_rate, latent_dim):
        super().__init__()
        # Spatial Feedback Stem
        self.stem_conv = nn.Conv1d(input_dim, growth_rate, kernel_size=3, padding=1)
        self.stem_norm = nn.LayerNorm(growth_rate)
        self.stem_act = nn.SiLU()

        # Lightweight Backbone (fewer dilations, same growth rate logic)
        dilations = [1, 2, 4, 8]
        self.backbone = DenseBackbone(
            growth_rate, growth_rate, dilations, 0.1, latent_dim
        )

    def forward(self, x):
        # x: (B, 5, L)

        # Stem
        out = self.stem_conv(x)
        out = out.permute(0, 2, 1)
        out = self.stem_norm(out)
        out = self.stem_act(out)
        out = out.permute(0, 2, 1)

        # Backbone
        out = self.backbone(out)
        return out


class AHS_DFN(nn.Module):
    """
    Anchored Hybrid-Stem Dense-Feedback Network.
    """

    def __init__(self):
        super().__init__()

        # Input channels: 4 (seq) + 3 (struct) + 7 (loop) + 4 (partner) = 18
        self.input_dim = 18

        # 1. Hybrid Input Stem
        self.hybrid_stem = HybridInputStem(self.input_dim, HIDDEN_DIM)
        # Output dim of hybrid stem = 18 + HIDDEN_DIM

        # 2. Main Backbone
        backbone_input_dim = self.input_dim + HIDDEN_DIM
        self.main_backbone = DenseBackbone(
            backbone_input_dim,
            GROWTH_RATE,
            DILATIONS,
            DROPOUT,
            HIDDEN_DIM,  # Latent Z dim
        )

        # 3. Feedback Module
        self.feedback_module = FeedbackModule(
            NUM_TARGETS, FEEDBACK_GROWTH_RATE, FEEDBACK_LATENT_DIM
        )

        # 4. Interaction & Aggregation
        # Input to GRU: (Z_self + E_fb_self) + (Z_partner + E_fb_partner)
        # Dim = (HIDDEN_DIM + FEEDBACK_LATENT_DIM) * 2
        gru_input_dim = (HIDDEN_DIM + FEEDBACK_LATENT_DIM) * 2
        self.gru = nn.GRU(
            gru_input_dim, HIDDEN_DIM, batch_first=True, bidirectional=True
        )

        # 5. Head
        self.head = nn.Linear(HIDDEN_DIM * 2, NUM_TARGETS)

    def compute_features(self, x):
        """
        Computes static backbone features Z.
        """
        # x: (B, L, 18) -> Permute to (B, 18, L) for Conv1d
        x = x.permute(0, 2, 1)

        stem_out = self.hybrid_stem(x)
        z = self.main_backbone(stem_out)  # (B, 64, L)

        return z.permute(0, 2, 1)  # Return as (B, L, 64)

    def compute_output(self, z, partner_indices, feedback_preds):
        """
        Computes predictions given static features Z and feedback predictions.
        """
        B, L, _ = z.shape

        # 1. Process Feedback
        if feedback_preds is None:
            # First pass: Zero feedback
            e_fb = torch.zeros(B, L, FEEDBACK_LATENT_DIM, device=z.device)
        else:
            # Apply Channel Masking (zero out indices 2 and 4: deg_pH10, deg_50C)
            masked_preds = feedback_preds.clone()
            masked_preds[:, :, 2] = 0.0
            masked_preds[:, :, 4] = 0.0

            # Permute to (B, 5, L)
            masked_preds = masked_preds.permute(0, 2, 1)

            # Process via Feedback Module
            e_fb = self.feedback_module(masked_preds)  # (B, 32, L)
            e_fb = e_fb.permute(0, 2, 1)  # (B, L, 32)

        # 2. Interaction (Augmented Gather)
        # Self vector: [Z_i, E_fb_i]
        self_vec = torch.cat([z, e_fb], dim=2)  # (B, L, 96)

        # Partner Gathering
        # partner_indices has -1 for unpaired. Replace -1 with 0 for safe gathering.
        safe_indices = partner_indices.clone()
        mask_unpaired = safe_indices == -1
        safe_indices[mask_unpaired] = 0

        # Expand indices for gathering: (B, L, feature_dim)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, self_vec.size(2))

        # Gather partner vectors
        partner_vec = torch.gather(self_vec, 1, gather_indices)

        # Apply Zero-Mask to partner vectors where bases are unpaired
        partner_vec[mask_unpaired] = 0.0

        # Fusion
        rnn_input = torch.cat([self_vec, partner_vec], dim=2)  # (B, L, 192)

        # 3. Global Aggregation
        rnn_out, _ = self.gru(rnn_input)  # (B, L, 128)

        # 4. Head
        logits = self.head(rnn_out)  # (B, L, 5)

        return logits

    def forward(self, x, partner_indices):
        # x: (B, L, 18)
        # partner_indices: (B, L)

        # Step 1: Compute Static Features (Once)
        z = self.compute_features(x)

        # Step 2: Pass 1 (Zero Feedback)
        y_hat_1 = self.compute_output(z, partner_indices, None)

        # Step 3: Pass 2 (With Feedback)
        # Detach gradients from Pass 1 to stop gradient flow through feedback input
        feedback_input = y_hat_1.detach()
        y_hat_2 = self.compute_output(z, partner_indices, feedback_input)

        return y_hat_1, y_hat_2
