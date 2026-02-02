import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridStem(nn.Module):
    """
    Splits inputs into two branches:
    1. Identity Branch: Preserves raw sparse features (including Partner Identity).
    2. Context Branch: Extracts local spatial context via Conv1d -> LN -> SiLU.
    Concatenates both to form the input to the backbone.
    """

    def __init__(self, in_channels=18, out_channels=32, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.ln = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # Branch A: Identity (Raw features)
        branch_a = x

        # Branch B: Context (Spatial aggregation)
        # Permute for LayerNorm (B, C, L) -> (B, L, C)
        branch_b = self.conv(x)
        branch_b = self.ln(branch_b.permute(0, 2, 1)).permute(0, 2, 1)
        branch_b = self.act(branch_b)

        # Concatenate along channel dimension
        return torch.cat([branch_a, branch_b], dim=1)


class DenseDilatedBlock(nn.Module):
    """
    Post-Activation Dense Block:
    Conv(k=3, d=D) -> LN -> SiLU -> Conv(k=1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.ln1 = nn.LayerNorm(growth_rate)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.ln2 = nn.LayerNorm(growth_rate)
        self.act2 = nn.SiLU()

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Spatial Aggregation
        out = self.conv1(x)
        out = self.ln1(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.act1(out)

        # Channel Mixing
        out = self.conv2(out)
        out = self.ln2(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.act2(out)

        return self.drop(out)


class DenseTCNBackbone(nn.Module):
    """
    Stack of DenseDilatedBlocks with dense connections.
    Each block receives the concatenation of ALL prior feature maps.
    """

    def __init__(self, in_dim, growth_rate, dilations, latent_dim):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_dim = in_dim

        for d in dilations:
            blk = DenseDilatedBlock(current_dim, growth_rate, d, Config.DROPOUT)
            self.blocks.append(blk)
            # In a dense network, the input to the next layer grows by the growth_rate
            current_dim += growth_rate

        # Project the final dense concatenation to latent dimension
        self.latent_proj = nn.Conv1d(current_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        features = [x]
        for block in self.blocks:
            # Concatenate all previous features
            dense_input = torch.cat(features, dim=1)
            out = block(dense_input)
            features.append(out)

        # Final projection of all features
        total_concat = torch.cat(features, dim=1)
        return self.latent_proj(total_concat)


class FeedbackModule(nn.Module):
    """
    Processes recycled predictions.
    1. Masks unscored channels.
    2. Applies Spatial Stem.
    3. Applies Lightweight Dense TCN.
    """

    def __init__(self, in_dim=5, growth_rate=16, layers=4, out_dim=32):
        super().__init__()
        self.stem = nn.Conv1d(in_dim, growth_rate, kernel_size=3, padding=1)
        self.stem_ln = nn.LayerNorm(growth_rate)
        self.stem_act = nn.SiLU()

        self.blocks = nn.ModuleList()
        current_dim = growth_rate

        # Lightweight Dense TCN (Dilation 1 assumed for local refinement)
        for _ in range(layers):
            blk = DenseDilatedBlock(current_dim, growth_rate, dilation=1, dropout=0.0)
            self.blocks.append(blk)
            current_dim += growth_rate

        self.out_proj = nn.Conv1d(current_dim, out_dim, kernel_size=1)
        self.scored_indices = Config.SCORED_TARGET_INDICES

    def forward(self, y_prev):
        # Channel Masking: Zero out unscored channels (indices 2, 4)
        # Keep sequence length intact (global context)
        mask = torch.zeros_like(y_prev)
        mask[:, self.scored_indices, :] = 1.0
        y_masked = y_prev * mask

        # Stem
        out = self.stem(y_masked)
        out = self.stem_ln(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.stem_act(out)

        # Dense Blocks
        features = [out]
        for block in self.blocks:
            dense_input = torch.cat(features, dim=1)
            out = block(dense_input)
            features.append(out)

        total_concat = torch.cat(features, dim=1)
        return self.out_proj(total_concat)


class InteractionModule(nn.Module):
    """
    Performs Augmented Gather (Self + Partner) and Global Aggregation.
    """

    def __init__(self, input_dim, rnn_hidden_dim):
        super().__init__()
        # Input to RNN is Self (dim) + Partner (dim)
        self.rnn = nn.GRU(
            input_dim * 2,
            rnn_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(rnn_hidden_dim * 2, 5)

    def forward(self, z, e_fb, pairs):
        # z: (B, Latent, L)
        # e_fb: (B, Feedback, L)
        # pairs: (B, L)

        B, _, L = z.shape

        # 1. Construct Self Vector: [Z_i, E_fb_i]
        self_vec = torch.cat([z, e_fb], dim=1)  # (B, 96, L)
        self_vec_t = self_vec.permute(0, 2, 1)  # (B, L, 96)

        # 2. Gather Partner Vector: [Z_j, E_fb_j]
        # Create batch indices grid
        batch_idx = torch.arange(B, device=z.device).unsqueeze(1).expand(B, L)

        # Handle unpaired bases (-1)
        valid_mask = pairs != -1
        safe_pairs = pairs.clone()
        safe_pairs[~valid_mask] = (
            0  # Point to index 0 temporarily to avoid gather error
        )

        partner_vec_t = self_vec_t[batch_idx, safe_pairs]  # (B, L, 96)

        # Null-Masking: Zero out vectors for unpaired bases
        partner_vec_t[~valid_mask] = 0.0

        # 3. Fusion: Concatenate Self and Partner
        combined = torch.cat([self_vec_t, partner_vec_t], dim=2)  # (B, L, 192)

        # 4. Global Aggregation (BiGRU)
        rnn_out, _ = self.rnn(combined)  # (B, L, 128)

        # 5. Projection
        logits = self.head(rnn_out)  # (B, L, 5)
        return logits  # (B, L, 5)


class HCHSGFN(nn.Module):
    """
    High-Capacity Hybrid-Stem Global-Feedback Network.
    Integrates all sub-modules for iterative refinement.
    """

    def __init__(self):
        super().__init__()

        # 1. Hybrid Stem (18 -> 50)
        self.stem = HybridStem(
            in_channels=18, out_channels=32, kernel_size=Config.STEM_KERNEL
        )

        # 2. Dense Backbone (50 -> 64)
        # Input dim is 18 (Branch A) + 32 (Branch B) = 50
        self.backbone = DenseTCNBackbone(
            in_dim=50,
            growth_rate=Config.BACKBONE_GROWTH_RATE,
            dilations=Config.BACKBONE_DILATIONS,
            latent_dim=Config.LATENT_DIM,
        )

        # 3. Feedback Module (5 -> 32)
        self.feedback = FeedbackModule(
            in_dim=5,
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            layers=Config.FEEDBACK_LAYERS,
            out_dim=Config.FEEDBACK_OUT_DIM,
        )

        # 4. Interaction Module (64 + 32 -> 5)
        self.interaction = InteractionModule(
            input_dim=Config.LATENT_DIM + Config.FEEDBACK_OUT_DIM,
            rnn_hidden_dim=Config.RNN_HIDDEN_DIM,
        )

    def forward(self, x, pairs, y_prev=None):
        # x: (B, 18, L)
        # pairs: (B, L)
        # y_prev: (B, 5, L) or None

        # Static Pass (computed once per sample)
        stem_out = self.stem(x)
        z = self.backbone(stem_out)  # (B, 64, L)

        # Feedback Pass
        if y_prev is None:
            # Initialize with zeros if no previous prediction
            y_prev = torch.zeros((x.shape[0], 5, x.shape[2]), device=x.device)

        e_fb = self.feedback(y_prev)  # (B, 32, L)

        # Interaction & Head
        preds = self.interaction(z, e_fb, pairs)  # (B, 5, L)

        return preds, z
