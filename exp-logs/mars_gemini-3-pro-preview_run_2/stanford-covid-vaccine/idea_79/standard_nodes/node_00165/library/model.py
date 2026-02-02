import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    SEQ_LENGTH,
    SCORED_SEQ_LENGTH,
    LATENT_DIM,
    GROWTH_RATE,
    DILATION_RATES,
    KERNEL_SIZE_STEM,
    KERNEL_SIZE_POINTWISE,
    DROPOUT,
    FEEDBACK_DIM,
    FEEDBACK_GROWTH_RATE,
    RNN_HIDDEN_DIM,
    BIDIRECTIONAL,
    SCORED_INDICES,
)


class SpatialStem(nn.Module):
    """
    Processes raw inputs with a spatial convolution to extract local context
    before entering the dense backbone.
    Structure: Conv1d(k=3) -> LayerNorm -> SiLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (N, C, L)
        out = self.conv(x)

        # LayerNorm expects (N, L, C)
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = self.act(out)

        # Return to (N, C, L)
        out = out.permute(0, 2, 1)
        return out


class DenseBlock(nn.Module):
    """
    Post-Activation Dense Block with Decoupled Spatial/Channel mixing.
    Structure: Conv(Dilated) -> LN -> SiLU -> Conv(1x1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, dilation):
        super().__init__()

        # Spatial Mixing (Dilated Conv)
        self.conv1 = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm1 = nn.LayerNorm(growth_rate)
        self.act1 = nn.SiLU()

        # Channel Mixing (Pointwise Conv)
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.norm2 = nn.LayerNorm(growth_rate)
        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # x: (N, C_in, L)

        # 1. Spatial Conv
        out = self.conv1(x)

        # LN/Act 1
        out = out.permute(0, 2, 1)
        out = self.norm1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)

        # 2. Pointwise Conv
        out = self.conv2(out)

        # LN/Act 2
        out = out.permute(0, 2, 1)
        out = self.norm2(out)
        out = self.act2(out)
        out = out.permute(0, 2, 1)

        out = self.dropout(out)
        return out


class RawInjectingBackbone(nn.Module):
    """
    Dense Dilated TCN that injects Raw Features at every block input.
    Input to block k = Concat(All Previous Outputs, Raw Features).
    """

    def __init__(self, in_channels, raw_channels, growth_rate, dilation_rates):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_dim = in_channels

        for d in dilation_rates:
            # Input dim = Accumulated Dense Features + Raw Features
            block_in_dim = current_dim + raw_channels

            blk = DenseBlock(block_in_dim, growth_rate, d)
            self.blocks.append(blk)

            # Accumulate growth rate
            current_dim += growth_rate

        self.out_channels = current_dim

    def forward(self, x, raw_x):
        # x: (N, C_stem, L) - Initial stem features
        # raw_x: (N, C_raw, L) - Raw features for recursive injection

        features = [x]

        for block in self.blocks:
            # 1. Concatenate all prior dense features
            dense_accum = torch.cat(features, dim=1)

            # 2. Inject Raw Features
            block_input = torch.cat([dense_accum, raw_x], dim=1)

            # 3. Process
            new_features = block(block_input)
            features.append(new_features)

        # Return concatenation of all features (DenseNet style)
        return torch.cat(features, dim=1)


class FeedbackModule(nn.Module):
    """
    Processes recycled predictions.
    Structure: Spatial Stem -> Lightweight Dense TCN -> Projection
    """

    def __init__(self):
        super().__init__()
        # Input: 5 channels (predictions)
        self.stem = SpatialStem(5, FEEDBACK_GROWTH_RATE, kernel_size=3)

        self.backbone = nn.ModuleList()
        current_dim = FEEDBACK_GROWTH_RATE

        # Standard DenseNet for feedback (no raw injection)
        for d in DILATION_RATES:
            blk = DenseBlock(current_dim, FEEDBACK_GROWTH_RATE, d)
            self.backbone.append(blk)
            current_dim += FEEDBACK_GROWTH_RATE

        self.project = nn.Conv1d(current_dim, FEEDBACK_DIM, 1)

    def forward(self, y_pred):
        # y_pred: (N, L, 5) -> Permute to (N, 5, L)
        x = y_pred.permute(0, 2, 1)

        x = self.stem(x)

        features = [x]
        for block in self.backbone:
            dense_input = torch.cat(features, dim=1)
            new_features = block(dense_input)
            features.append(new_features)

        out = torch.cat(features, dim=1)
        out = self.project(out)  # (N, FeedbackDim, L)
        return out


class InteractionModule(nn.Module):
    """
    Handles Partner Gathering and Global Aggregation.
    Gather([Z, E_fb]) -> Concat -> Bidirectional GRU -> Linear
    """

    def __init__(self):
        super().__init__()
        # Input = Latent (Backbone) + Feedback
        self.input_dim = LATENT_DIM + FEEDBACK_DIM

        # Fusion = Self + Partner
        self.fusion_dim = self.input_dim * 2

        self.gru = nn.GRU(
            self.fusion_dim,
            RNN_HIDDEN_DIM,
            batch_first=True,
            bidirectional=BIDIRECTIONAL,
        )

        rnn_out_dim = RNN_HIDDEN_DIM * 2 if BIDIRECTIONAL else RNN_HIDDEN_DIM
        self.head = nn.Linear(rnn_out_dim, 5)

    def forward(self, z, e_fb, partner_map):
        # z: (N, 64, L)
        # e_fb: (N, 32, L)
        # partner_map: (N, L) - LongTensor of indices

        # 1. Concatenate Self Vectors
        combined = torch.cat([z, e_fb], dim=1)  # (N, 96, L)

        # Permute to (N, L, 96) for gathering
        combined = combined.permute(0, 2, 1)
        N, L, C = combined.shape

        # 2. Gather Partner Vectors
        # Handle -1 (unpaired) by replacing with 0 temporarily, then masking
        p_indices = partner_map.long()
        unpaired_mask = p_indices == -1

        safe_indices = p_indices.clone()
        safe_indices[unpaired_mask] = 0

        # Expand indices to (N, L, C)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, C)

        # Gather
        partner_vecs = torch.gather(combined, 1, gather_indices)

        # Apply Zero-Mask to unpaired positions
        partner_vecs[unpaired_mask.unsqueeze(-1).expand_as(partner_vecs)] = 0.0

        # 3. Fusion
        fusion = torch.cat([combined, partner_vecs], dim=2)  # (N, L, 192)

        # 4. Global Aggregation
        rnn_out, _ = self.gru(fusion)

        # 5. Projection
        logits = self.head(rnn_out)  # (N, L, 5)

        return logits


class RIS_DRN(nn.Module):
    """
    Raw-Injecting Spatial-Dense Recurrent Network (RIS-DRN).
    Combines recursive raw feature injection, dense dilated convolutions,
    and a global-context feedback loop.
    """

    def __init__(self):
        super().__init__()

        # Raw Channels: 4(Seq) + 3(Struct) + 7(Loop) + 4(Partner) = 18
        self.raw_channels = 18

        # 1. Spatial Input Stem
        self.input_stem = SpatialStem(self.raw_channels, GROWTH_RATE, KERNEL_SIZE_STEM)

        # 2. Main Backbone (Raw-Injecting)
        self.backbone = RawInjectingBackbone(
            in_channels=GROWTH_RATE,
            raw_channels=self.raw_channels,
            growth_rate=GROWTH_RATE,
            dilation_rates=DILATION_RATES,
        )

        # Latent Projection
        self.latent_proj = nn.Conv1d(self.backbone.out_channels, LATENT_DIM, 1)

        # 3. Feedback Module
        self.feedback_module = FeedbackModule()

        # 4. Interaction & Aggregation
        self.interaction = InteractionModule()

        # Scored Mask for Feedback
        # Indexes: 0=reactivity, 1=deg_Mg_pH10, 2=deg_pH10, 3=deg_Mg_50C, 4=deg_50C
        # Scored: 0, 1, 3. Unscored: 2, 4.
        self.register_buffer(
            "scored_mask", torch.tensor([1, 1, 0, 1, 0], dtype=torch.float32)
        )

    def forward(self, inputs, partner_map, targets=None):
        # inputs: (N, L, 18)
        # partner_map: (N, L)

        # Prepare Inputs: (N, 18, L)
        raw_x = inputs.permute(0, 2, 1)

        # --- Step 1: Static Backbone ---
        stem_out = self.input_stem(raw_x)
        backbone_out = self.backbone(stem_out, raw_x)
        z = self.latent_proj(backbone_out)  # (N, 64, L)

        # --- Step 2: Iterative Refinement ---

        # Pass 1: Zero Feedback
        batch_size, _, seq_len = z.shape
        y_pred_0 = torch.zeros(batch_size, seq_len, 5, device=z.device)

        # Process Pass 1
        e_fb_0 = self.feedback_module(y_pred_0)
        logits_1 = self.interaction(z, e_fb_0, partner_map)

        # Pass 2: Feedback from Logits 1
        r = logits_1.detach()

        # Apply Channel Masking (Zero out unscored columns)
        r_masked = r * self.scored_mask.view(1, 1, 5)

        # Process Pass 2
        e_fb_1 = self.feedback_module(r_masked)
        logits_2 = self.interaction(z, e_fb_1, partner_map)

        return logits_1, logits_2


def loss_fn(logits_1, logits_2, targets):
    """
    Calculates the MCRMSE loss with strict masking.

    Args:
        logits_1: Output from Pass 1 (N, L, 5)
        logits_2: Output from Pass 2 (N, L, 5)
        targets: Ground truth (N, L, 5)

    Returns:
        Weighted combined loss.
    """
    # 1. Sequence Masking
    # Only calculate loss on the scored sequence length (0 to SCORED_SEQ_LENGTH)
    # We ignore the zero-padded tail (68-107) for loss calculation.
    pred_1 = logits_1[:, :SCORED_SEQ_LENGTH, :]
    pred_2 = logits_2[:, :SCORED_SEQ_LENGTH, :]
    true_y = targets[:, :SCORED_SEQ_LENGTH, :]

    # 2. Target Masking
    # Only calculate loss on scored columns
    pred_1 = pred_1[:, :, SCORED_INDICES]
    pred_2 = pred_2[:, :, SCORED_INDICES]
    true_y = true_y[:, :, SCORED_INDICES]

    # 3. MCRMSE Calculation (Global Average of RMSE per column)
    # Mean Squared Error per column (averaged over batch and sequence)
    mse_1 = torch.mean((pred_1 - true_y) ** 2, dim=(0, 1))
    mcrmse_1 = torch.mean(torch.sqrt(mse_1))

    mse_2 = torch.mean((pred_2 - true_y) ** 2, dim=(0, 1))
    mcrmse_2 = torch.mean(torch.sqrt(mse_2))

    # Total Loss: Pass 2 + 0.5 * Pass 1
    loss = mcrmse_2 + 0.5 * mcrmse_1

    return loss
