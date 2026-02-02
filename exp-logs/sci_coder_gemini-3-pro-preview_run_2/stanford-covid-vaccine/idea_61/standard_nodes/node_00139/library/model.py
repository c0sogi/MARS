import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridInputStem(nn.Module):
    """
    Processes inputs via two parallel branches to resolve the Input Representation Conflict.
    Branch A: Preserves raw discrete identity (One-Hot).
    Branch B: Extracts local context via Spatial Convolution.
    """

    def __init__(self, input_channels, context_channels):
        super(HybridInputStem, self).__init__()
        # Branch B: Context extraction
        self.branch_b_conv = nn.Conv1d(
            input_channels, context_channels, kernel_size=3, padding=1
        )
        self.branch_b_norm = nn.LayerNorm(context_channels)
        self.branch_b_act = nn.SiLU()

    def forward(self, x):
        # x: (B, C, L)

        # Branch A: Identity (Raw Features)
        branch_a = x

        # Branch B: Context
        branch_b = self.branch_b_conv(x)
        # Permute for LayerNorm: (B, C, L) -> (B, L, C)
        branch_b = branch_b.permute(0, 2, 1)
        branch_b = self.branch_b_norm(branch_b)
        branch_b = self.branch_b_act(branch_b)
        # Permute back: (B, L, C) -> (B, C, L)
        branch_b = branch_b.permute(0, 2, 1)

        # Concatenate branches
        out = torch.cat([branch_a, branch_b], dim=1)
        return out


class PostActDenseBlock(nn.Module):
    """
    A specific Post-Activation Dense Block structure:
    Conv(k=3) -> LN -> SiLU -> Conv(k=1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout):
        super(PostActDenseBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.ln1 = nn.LayerNorm(growth_rate)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.ln2 = nn.LayerNorm(growth_rate)
        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, In_C, L)
        out = self.conv1(x)

        # LN/Act 1
        out = out.permute(0, 2, 1)
        out = self.ln1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)

        out = self.conv2(out)

        # LN/Act 2
        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = self.act2(out)
        out = out.permute(0, 2, 1)

        out = self.dropout(out)
        return out


class DenseTCN(nn.Module):
    """
    A stack of dilated dense blocks.
    Uses dense connections: input to block i is concatenation of inputs to block 0..i-1.
    """

    def __init__(self, in_channels, growth_rate, dilations, dropout, out_channels):
        super(DenseTCN, self).__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for d in dilations:
            blk = PostActDenseBlock(current_channels, growth_rate, d, dropout)
            self.blocks.append(blk)
            current_channels += growth_rate

        # Projection to latent dimension
        self.projection = nn.Conv1d(current_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, L)
        features = [x]
        for blk in self.blocks:
            # Dense connection: Concatenate all previous features
            inp = torch.cat(features, dim=1)
            out = blk(inp)
            features.append(out)

        # Concatenate everything for final projection
        total_features = torch.cat(features, dim=1)
        out = self.projection(total_features)
        return out


class SpatialFeedbackStem(nn.Module):
    """
    Processes recycled predictions.
    Applies channel masking to zero out unscored targets.
    Uses a Spatial Convolution before the dense backbone.
    """

    def __init__(
        self, input_channels, hidden_dim, growth_rate, dilations, dropout, out_dim
    ):
        super(SpatialFeedbackStem, self).__init__()

        # Spatial Stem: Conv -> LN -> SiLU
        self.spatial_conv = nn.Conv1d(
            input_channels, hidden_dim, kernel_size=3, padding=1
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

        # Feedback Backbone (Lightweight)
        self.backbone = DenseTCN(
            in_channels=hidden_dim,
            growth_rate=growth_rate,
            dilations=dilations,
            dropout=dropout,
            out_channels=out_dim,
        )

    def forward(self, preds):
        # preds: (B, L, 5)

        # Channel Masking: Zero out indices 2 (deg_pH10) and 4 (deg_50C)
        # Keep 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
        mask = torch.tensor([1, 1, 0, 1, 0], device=preds.device, dtype=preds.dtype)
        masked_preds = preds * mask.view(1, 1, 5)

        # Permute to (B, C, L) for Conv1d
        x = masked_preds.permute(0, 2, 1)

        # Spatial Stem
        x = self.spatial_conv(x)

        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)

        # Backbone
        out = self.backbone(x)  # (B, Out_Dim, L)
        return out


class ADSRN(nn.Module):
    """
    Anchored Dual-Stem Recurrent Network.
    """

    def __init__(self):
        super(ADSRN, self).__init__()

        # 1. Hybrid Input Stem
        # Branch B context dim set to HIDDEN_DIM (64)
        self.hybrid_stem = HybridInputStem(Config.INPUT_CHANNELS, Config.HIDDEN_DIM)

        # Input to backbone is Input Channels (18) + Context Channels (64) = 82
        backbone_in_channels = Config.INPUT_CHANNELS + Config.HIDDEN_DIM

        # 2. Main Backbone
        self.main_backbone = DenseTCN(
            in_channels=backbone_in_channels,
            growth_rate=Config.GROWTH_RATE,
            dilations=Config.DILATIONS,
            dropout=Config.DROPOUT,
            out_channels=Config.LATENT_DIM,
        )

        # 3. Feedback Module
        self.feedback_module = SpatialFeedbackStem(
            input_channels=Config.FEEDBACK_INPUT_CHANNELS,
            hidden_dim=32,  # Intermediate dim for spatial stem
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            dilations=Config.DILATIONS,
            dropout=Config.DROPOUT,
            out_dim=Config.FEEDBACK_OUT_DIM,
        )

        # 4. Aggregator
        # Input to RNN: (Z + E_fb)_self + (Z + E_fb)_partner
        # Z dim = 64, E_fb dim = 32 -> Total per site = 96
        # Paired = 96 * 2 = 192
        rnn_input_dim = (Config.LATENT_DIM + Config.FEEDBACK_OUT_DIM) * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT,
        )

        self.head = nn.Linear(Config.HIDDEN_DIM * 2, 5)

    def forward(self, inputs, partner_indices):
        """
        Args:
            inputs: (B, L, 18)
            partner_indices: (B, L) with -1 for unpaired
        Returns:
            y_2: Final predictions (B, L, 5)
            y_1: First pass predictions (B, L, 5) for auxiliary loss
        """
        B, L, _ = inputs.shape

        # Permute inputs for CNNs: (B, 18, L)
        x = inputs.permute(0, 2, 1)

        # 1. Hybrid Stem
        x = self.hybrid_stem(x)

        # 2. Main Backbone -> Z
        z = self.main_backbone(x)  # (B, 64, L)
        z = z.permute(0, 2, 1)  # (B, L, 64)

        # 3. Iterative Refinement Loop

        # Step 2: Init Y_0 = 0
        y_current = torch.zeros((B, L, 5), device=inputs.device, dtype=inputs.dtype)

        outputs = []

        # Run 2 passes
        for i in range(2):
            # Detach gradients if it's the second pass (input is Y_1)
            # This prevents backprop through the feedback loop generation itself,
            # treating Y_1 as a fixed input for the second pass.
            if i == 1:
                y_in = y_current.detach()
            else:
                y_in = y_current

            # Feedback Module
            e_fb = self.feedback_module(y_in)  # (B, 32, L)
            e_fb = e_fb.permute(0, 2, 1)  # (B, L, 32)

            # Combine Z and E_fb
            node_feat = torch.cat([z, e_fb], dim=2)  # (B, L, 96)

            # Gather Partner Features
            # partner_indices has -1 for unpaired.
            # We need to handle -1. Replace -1 with 0 for gathering, then mask.
            p_idx = partner_indices.long()
            mask_unpaired = p_idx == -1
            p_idx_clamped = p_idx.clone()
            p_idx_clamped[mask_unpaired] = 0

            # Gather: (B, L, 96)
            # Create batch indices for advanced indexing
            batch_idx = torch.arange(B, device=inputs.device).view(B, 1).expand(B, L)
            partner_feat = node_feat[batch_idx, p_idx_clamped, :]

            # Apply Null-Masking for unpaired bases
            partner_feat[mask_unpaired] = 0.0

            # Fusion
            rnn_in = torch.cat([node_feat, partner_feat], dim=2)  # (B, L, 192)

            # Global Aggregation (BiGRU)
            rnn_out, _ = self.rnn(rnn_in)

            # Head
            y_next = self.head(rnn_out)
            outputs.append(y_next)
            y_current = y_next

        # Return Y_2 (final) and Y_1 (auxiliary)
        return outputs[1], outputs[0]
