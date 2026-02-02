import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridInputStem(nn.Module):
    """
    Splits input into two branches:
    1. Identity: Preserves raw one-hot features.
    2. Context: Applies spatial convolution to extract local motifs.
    Concatenates them to form the input to the backbone.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.context_conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(input_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (Batch, Length, Channels)

        # Branch A: Identity
        branch_a = x

        # Branch B: Context
        # Permute for Conv1d: (B, C, L)
        branch_b = x.transpose(1, 2)
        branch_b = self.context_conv(branch_b)
        branch_b = branch_b.transpose(1, 2)  # Back to (B, L, C) for LN
        branch_b = self.norm(branch_b)
        branch_b = self.act(branch_b)

        # Concatenate: Output dim is 2 * input_dim
        return torch.cat([branch_a, branch_b], dim=-1)


class PostActDenseBlock(nn.Module):
    """
    A single block for the DenseTCN.
    Structure: Conv(k=3, d=d) -> LN -> SiLU -> Conv(k=1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super().__init__()
        # Depthwise-separable or standard? Prompt says "Standard Dilated Conv".
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_channels, growth_rate, kernel_size, padding=padding, dilation=dilation
        )
        self.norm1 = nn.LayerNorm(growth_rate)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.norm2 = nn.LayerNorm(growth_rate)
        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C_in, L)

        out = self.conv1(x)

        # LN/Act
        out = out.transpose(1, 2)  # (B, L, C)
        out = self.norm1(out)
        out = self.act1(out)
        out = out.transpose(1, 2)  # (B, C, L)

        out = self.conv2(out)

        # LN/Act/Drop
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)

        return out


class DenseTCN(nn.Module):
    """
    Dilated Dense Network Backbone.
    Maintains dense connections by concatenating inputs to all subsequent layers.
    """

    def __init__(
        self, in_channels, growth_rate, kernel_size, dilations, dropout, out_channels
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        current_dim = in_channels
        for d in dilations:
            blk = PostActDenseBlock(current_dim, growth_rate, kernel_size, d, dropout)
            self.blocks.append(blk)
            current_dim += growth_rate

        # Final projection to latent dimension
        self.projection = nn.Conv1d(current_dim, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, L, C) -> Permute to (B, C, L) for Conv operations
        x = x.transpose(1, 2)

        features = [x]

        for block in self.blocks:
            # Dense connection: Concatenate all previous features along channel dim
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Final concatenation
        total = torch.cat(features, dim=1)

        # Project to latent dim
        out = self.projection(total)

        # Return to (B, L, C)
        return out.transpose(1, 2)


class FeedbackModule(nn.Module):
    """
    Processes recycled predictions.
    1. Masks unscored channels.
    2. Spatial Stem.
    3. Lightweight DenseTCN.
    """

    def __init__(self, num_targets, scored_indices, feedback_dim):
        super().__init__()
        self.num_targets = num_targets
        self.scored_indices = scored_indices

        # Stem
        self.stem_conv = nn.Conv1d(num_targets, 16, kernel_size=3, padding=1)
        self.stem_norm = nn.LayerNorm(16)
        self.stem_act = nn.SiLU()

        # Backbone (Lightweight: Growth Rate 16)
        # Using same dilations as main backbone for consistent receptive field growth
        self.backbone = DenseTCN(
            in_channels=16,
            growth_rate=16,
            kernel_size=3,
            dilations=Config.DILATIONS,
            dropout=Config.DROPOUT,
            out_channels=feedback_dim,
        )

    def forward(self, preds):
        # preds: (B, L, 5)

        # 1. Channel Masking
        # Create a mask or manually zero out
        mask = torch.zeros_like(preds)
        mask[:, :, self.scored_indices] = 1.0
        masked_preds = preds * mask

        # 2. Spatial Stem
        x = masked_preds.transpose(1, 2)  # (B, C, L)
        x = self.stem_conv(x)
        x = x.transpose(1, 2)  # (B, L, C)
        x = self.stem_norm(x)
        x = self.stem_act(x)

        # 3. Backbone
        out = self.backbone(x)  # (B, L, Feedback_Dim)

        return out


class InteractionAggregator(nn.Module):
    """
    Handles fusion of static and dynamic features, partner gathering, and RNN aggregation.
    """

    def __init__(self, latent_dim, feedback_dim, rnn_hidden, num_targets):
        super().__init__()

        input_dim = latent_dim + feedback_dim
        self.fusion_dim = input_dim * 2  # Self + Partner

        # Bidirectional GRU
        self.rnn = nn.GRU(
            input_size=self.fusion_dim,
            hidden_size=rnn_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # Final Projection
        self.head = nn.Linear(rnn_hidden * 2, num_targets)

    def forward(self, z, e_fb, partner_indices):
        # z: (B, L, Latent)
        # e_fb: (B, L, Feedback)
        # partner_indices: (B, L)

        # 1. Concatenate Self Vectors
        h_self = torch.cat([z, e_fb], dim=-1)  # (B, L, Latent+Feedback)

        # 2. Gather Partner Vectors
        batch_size, seq_len, _ = h_self.shape

        # Handle -1 in partner_indices by clamping to 0 and masking later
        # partner_indices is (B, L)
        # We need to gather from h_self which is (B, L, C)

        # Create batch indices: (B, L)
        batch_idx = (
            torch.arange(batch_size, device=h_self.device)
            .unsqueeze(1)
            .expand(-1, seq_len)
        )

        # Safe indices: replace -1 with 0
        safe_partner_idx = partner_indices.clone()
        mask_unpaired = safe_partner_idx == -1
        safe_partner_idx[mask_unpaired] = 0

        # Gather
        # h_self[b, p_idx, :]
        h_partner = h_self[batch_idx, safe_partner_idx, :]

        # Null-Masking: Zero out vectors where partner was -1
        h_partner[mask_unpaired] = 0.0

        # 3. Fusion
        h_combined = torch.cat(
            [h_self, h_partner], dim=-1
        )  # (B, L, (Latent+Feedback)*2)

        # 4. Global Aggregation (RNN)
        rnn_out, _ = self.rnn(h_combined)  # (B, L, Hidden*2)

        # 5. Projection
        logits = self.head(rnn_out)

        return logits


class RHIDFN(nn.Module):
    """
    Robust Hybrid-Input Dense-Feedback Network.
    """

    def __init__(self):
        super().__init__()

        # Dimensions
        self.raw_input_dim = 18  # 4(Seq) + 3(Struct) + 7(Loop) + 4(Partner)
        self.hybrid_dim = self.raw_input_dim * 2

        # 1. Hybrid Input Stem
        self.input_stem = HybridInputStem(self.raw_input_dim)

        # 2. Main Backbone
        self.backbone = DenseTCN(
            in_channels=self.hybrid_dim,
            growth_rate=Config.GROWTH_RATE,
            kernel_size=Config.KERNEL_SIZE,
            dilations=Config.DILATIONS,
            dropout=Config.DROPOUT,
            out_channels=Config.LATENT_DIM,
        )

        # 3. Feedback Module
        self.feedback_module = FeedbackModule(
            num_targets=Config.NUM_TARGETS,
            scored_indices=Config.SCORED_INDICES,
            feedback_dim=Config.FEEDBACK_DIM,
        )

        # 4. Interaction & Aggregation
        self.aggregator = InteractionAggregator(
            latent_dim=Config.LATENT_DIM,
            feedback_dim=Config.FEEDBACK_DIM,
            rnn_hidden=Config.RNN_HIDDEN_SIZE,
            num_targets=Config.NUM_TARGETS,
        )

    def forward(self, inputs, partner_indices):
        """
        Args:
            inputs: (B, L, 18)
            partner_indices: (B, L)
        Returns:
            y_1: Prediction from Pass 1 (No Feedback)
            y_2: Prediction from Pass 2 (With Feedback)
        """
        # Step 1: Compute Static Features (Z)
        x = self.input_stem(inputs)
        z = self.backbone(x)  # (B, L, Latent_Dim)

        # Step 2: Pass 1 (Zero Feedback)
        batch_size, seq_len, _ = z.shape
        device = z.device

        # Initialize zero feedback embeddings directly (skipping module forward for 0 input optimization)
        # Or strictly follow logic: feed zeros to feedback module.
        # Let's feed zeros to ensure BN/LN stats if any (though we use LN which is instance based)
        # Ideally, e_fb_0 should be computed from zero predictions.

        y_0 = torch.zeros(batch_size, seq_len, Config.NUM_TARGETS, device=device)
        e_fb_0 = self.feedback_module(y_0)

        y_1 = self.aggregator(z, e_fb_0, partner_indices)

        # Step 3: Pass 2 (Recycled Feedback)
        # Detach gradients from Pass 1 to stop gradient flow through the feedback loop target
        r = y_1.detach()

        e_fb_1 = self.feedback_module(r)
        y_2 = self.aggregator(z, e_fb_1, partner_indices)

        return y_1, y_2
