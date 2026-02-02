import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class LayerNormChannels(nn.Module):
    """
    Applies LayerNorm along the channel dimension for (N, C, L) tensors.
    """

    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (N, C, L) -> (N, L, C)
        x = x.transpose(1, 2)
        x = self.ln(x)
        # (N, L, C) -> (N, C, L)
        return x.transpose(1, 2)


class HybridInputStem(nn.Module):
    """
    Splits input into Identity (Raw) and Context (Conv processed) branches.
    """

    def __init__(self, in_channels, context_channels):
        super().__init__()
        self.context_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                context_channels,
                kernel_size=config.KERNEL_SIZE,
                padding=config.KERNEL_SIZE // 2,
            ),
            LayerNormChannels(context_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        # x: (N, C, L)
        # Branch A: Identity
        identity = x
        # Branch B: Context
        context = self.context_conv(x)
        # Concatenate
        return torch.cat([identity, context], dim=1)


class PostActDenseBlock(nn.Module):
    """
    Post-Activation Dense Block:
    Conv(k=3) -> LN -> SiLU -> Conv(k=1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                growth_rate,
                kernel_size=config.KERNEL_SIZE,
                padding=dilation * (config.KERNEL_SIZE // 2),
                dilation=dilation,
            ),
            LayerNormChannels(growth_rate),
            nn.SiLU(),
            nn.Conv1d(growth_rate, growth_rate, kernel_size=1),
            LayerNormChannels(growth_rate),
            nn.SiLU(),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class DenseBackbone(nn.Module):
    """
    Stack of Dilated Dense Blocks.
    """

    def __init__(self, in_channels, growth_rate, dilations, out_dim):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for d in dilations:
            blk = PostActDenseBlock(current_channels, growth_rate, dilation=d)
            self.blocks.append(blk)
            current_channels += growth_rate

        self.project_out = nn.Conv1d(current_channels, out_dim, kernel_size=1)

    def forward(self, x):
        # x: (N, C_in, L)
        features = [x]
        for block in self.blocks:
            # Dense connection: concatenate all previous features
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Concatenate everything for final projection
        total_features = torch.cat(features, dim=1)
        z = self.project_out(total_features)
        return z


class FeedbackProcessor(nn.Module):
    """
    Processes recycled predictions.
    Masks unscored channels, applies lightweight DenseNet.
    """

    def __init__(self, num_targets, feedback_dim):
        super().__init__()
        self.scored_indices = torch.tensor(
            config.SCORED_TARGET_INDICES, dtype=torch.long
        )

        # Stem
        stem_dim = 32
        self.stem = nn.Sequential(
            nn.Conv1d(
                num_targets,
                stem_dim,
                kernel_size=config.KERNEL_SIZE,
                padding=config.KERNEL_SIZE // 2,
            ),
            LayerNormChannels(stem_dim),
            nn.SiLU(),
        )

        # Lightweight Backbone (Growth Rate 16)
        self.backbone = DenseBackbone(
            in_channels=stem_dim,
            growth_rate=16,
            dilations=config.DILATIONS,
            out_dim=feedback_dim,
        )

    def forward(self, y_pred):
        # y_pred: (N, L, 5) -> permute to (N, 5, L)
        x = y_pred.permute(0, 2, 1)

        # Channel Masking: Zero out unscored channels
        # Create a mask of zeros
        mask = torch.zeros_like(x)
        # Set scored channels to 1
        mask[:, self.scored_indices.to(x.device), :] = 1.0

        # Apply mask
        x_masked = x * mask

        # Process
        x_stem = self.stem(x_masked)
        e_fb = self.backbone(x_stem)  # (N, Feedback_Dim, L)

        return e_fb


class InteractionModule(nn.Module):
    """
    Gathers partner features and fuses with self features.
    """

    def __init__(self, dim):
        super().__init__()
        # Input dim is (Z_dim + Feedback_dim) * 2 (Self + Partner)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, z, e_fb, partner_indices):
        # z: (N, Z_dim, L)
        # e_fb: (N, FB_dim, L)
        # partner_indices: (N, L)

        # Concatenate self features
        self_feat = torch.cat([z, e_fb], dim=1)  # (N, Dim, L)

        # Prepare for gathering: (N, L, Dim)
        self_feat_t = self_feat.permute(0, 2, 1)
        B, L, C = self_feat_t.shape

        # Handle -1 indices (unpaired)
        # Replace -1 with 0 for gather, then mask result
        mask_paired = (partner_indices != -1).unsqueeze(-1).float()  # (N, L, 1)
        p_idx_safe = partner_indices.clone()
        p_idx_safe[partner_indices == -1] = 0

        # Expand indices for gather: (N, L, C)
        idx_expanded = p_idx_safe.unsqueeze(-1).expand(-1, -1, C)

        # Gather partner features
        partner_feat_t = torch.gather(self_feat_t, 1, idx_expanded)

        # Apply mask (zero out features for unpaired bases)
        partner_feat_t = partner_feat_t * mask_paired

        # Concatenate Self and Partner
        combined = torch.cat([self_feat_t, partner_feat_t], dim=2)  # (N, L, 2*Dim)

        # Fuse
        out = self.fusion(combined)  # (N, L, Dim)

        return out


class HC_HIDN(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Input Dimensions ---
        # Seq(4) + Struct(3) + Loop(7) + PartnerSeq(4) = 18
        in_channels = 18

        # --- Hybrid Stem ---
        # Branch B projects to GROWTH_RATE (64)
        self.stem = HybridInputStem(in_channels, config.GROWTH_RATE)
        stem_out_channels = in_channels + config.GROWTH_RATE

        # --- Main Backbone ---
        self.backbone = DenseBackbone(
            in_channels=stem_out_channels,
            growth_rate=config.GROWTH_RATE,
            dilations=config.DILATIONS,
            out_dim=config.HIDDEN_DIM,
        )

        # --- Feedback Processor ---
        self.feedback_net = FeedbackProcessor(
            num_targets=config.NUM_TARGETS, feedback_dim=config.FEEDBACK_DIM
        )

        # --- Interaction & Aggregation ---
        interaction_dim = config.HIDDEN_DIM + config.FEEDBACK_DIM
        self.interaction = InteractionModule(interaction_dim)

        self.gru = nn.GRU(
            input_size=interaction_dim,
            hidden_size=config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Bidirectional output -> 2 * hidden
        self.head = nn.Linear(config.HIDDEN_DIM * 2, config.NUM_TARGETS)

    def forward_pass(self, z, e_fb, partner_indices):
        # Interaction
        fused = self.interaction(z, e_fb, partner_indices)  # (N, L, Dim)

        # Global Aggregation
        gru_out, _ = self.gru(fused)  # (N, L, 2*Hidden)

        # Head
        logits = self.head(gru_out)  # (N, L, 5)
        return logits

    def forward(self, x, partner_indices):
        # x: (N, L, C) -> permute to (N, C, L) for CNNs
        x = x.permute(0, 2, 1)

        # 1. Static Feature Extraction
        x_stem = self.stem(x)
        z = self.backbone(x_stem)  # (N, Z_dim, L)

        # 2. Pass 1 (Init Feedback with Zeros)
        B, _, L = z.shape
        # Initial prediction is zero
        y_prev = torch.zeros((B, L, config.NUM_TARGETS), device=z.device)

        # Generate initial feedback embedding
        e_fb_0 = self.feedback_net(y_prev)  # (N, FB_dim, L)

        # Predict Pass 1
        y1 = self.forward_pass(z, e_fb_0, partner_indices)

        # 3. Pass 2 (Recycle)
        # Detach gradients from Pass 1 to stop gradient explosion/drift
        y1_detached = y1.detach()

        # Generate refined feedback embedding
        e_fb_1 = self.feedback_net(y1_detached)

        # Predict Pass 2
        y2 = self.forward_pass(z, e_fb_1, partner_indices)

        return y1, y2
