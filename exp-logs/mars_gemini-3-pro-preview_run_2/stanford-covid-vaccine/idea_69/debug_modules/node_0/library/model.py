import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridStem(nn.Module):
    """
    Hybrid Input Stem that combines raw identity features with spatially convolved context.
    Branch A: Identity (Raw Features)
    Branch B: Conv1d(k=3) -> LayerNorm -> SiLU
    """

    def __init__(self, in_channels, context_channels):
        super().__init__()
        self.branch_identity = nn.Identity()
        self.branch_context = nn.Sequential(
            nn.Conv1d(in_channels, context_channels, kernel_size=3, padding=1),
            # LayerNorm and SiLU are applied in forward to handle shape permutations
        )
        self.ln = nn.LayerNorm(context_channels)
        self.silu = nn.SiLU()
        self.out_channels = in_channels + context_channels

    def forward(self, x):
        # x: (B, C, L)

        # Branch A: Identity
        out_a = x

        # Branch B: Context
        b = self.branch_context[0](x)  # Conv
        b = b.permute(0, 2, 1)  # (B, L, C) for LN
        b = self.ln(b)
        b = self.silu(b)
        out_b = b.permute(0, 2, 1)  # (B, C, L)

        return torch.cat([out_a, out_b], dim=1)


class DilatedBlock(nn.Module):
    """
    Post-Activation Dilated Convolution Block.
    Structure: Conv(k=3, d) -> LN -> SiLU -> Conv(k=1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout):
        super().__init__()
        self.spatial_conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.pointwise_conv = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)

        self.ln1 = nn.LayerNorm(growth_rate)
        self.silu1 = nn.SiLU()
        self.ln2 = nn.LayerNorm(growth_rate)
        self.silu2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, Cin, L)

        # 1. Spatial Aggregation
        out = self.spatial_conv(x)

        # 2. LN + SiLU
        out = out.permute(0, 2, 1)  # (B, L, C)
        out = self.ln1(out)
        out = self.silu1(out)
        out = out.permute(0, 2, 1)  # (B, C, L)

        # 3. Channel Mixing
        out = self.pointwise_conv(out)

        # 4. LN + SiLU + Dropout
        out = out.permute(0, 2, 1)  # (B, L, C)
        out = self.ln2(out)
        out = self.silu2(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)  # (B, C, L)

        return out


class DenseTCN(nn.Module):
    """
    Dense Temporal Convolutional Network.
    Uses dense connections where the input to block i is the concatenation of
    inputs to all previous blocks and their outputs (accumulated features).
    """

    def __init__(self, in_channels, growth_rate, dilations, dropout, out_dim):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_channels = in_channels

        for d in dilations:
            blk = DilatedBlock(current_channels, growth_rate, d, dropout)
            self.blocks.append(blk)
            current_channels += growth_rate

        self.project = nn.Conv1d(current_channels, out_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, Cin, L)
        features = [x]

        for block in self.blocks:
            # Concatenate all previous features (Dense Connection)
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Final concatenation of all features
        final_concat = torch.cat(features, dim=1)
        return self.project(final_concat)


class FeedbackModule(nn.Module):
    """
    Global-Context Pure-Feedback Module.
    Processes recycled predictions to extract feedback embeddings.
    """

    def __init__(self, growth_rate, out_dim):
        super().__init__()
        # Input is 5 channels (predictions)

        # Spatial Stem
        self.stem_conv = nn.Conv1d(5, 16, kernel_size=3, padding=1)
        self.stem_ln = nn.LayerNorm(16)
        self.stem_silu = nn.SiLU()

        # Backbone (Lightweight Dense TCN)
        # Using same dilations as main backbone but smaller growth rate
        dilations = Config.BACKBONE_DILATIONS
        self.backbone = DenseTCN(
            in_channels=16,
            growth_rate=growth_rate,
            dilations=dilations,
            dropout=0.0,  # No dropout in feedback loop typically
            out_dim=out_dim,
        )

    def forward(self, y_pred):
        # y_pred: (B, L, 5)

        # 1. Channel Masking
        # Mask unscored channels to prevent noise injection
        mask_vec = torch.zeros(5, device=y_pred.device)
        mask_vec[Config.SCORED_INDICES] = 1.0
        y_masked = y_pred * mask_vec.view(1, 1, 5)

        # Permute to (B, 5, L) for Conv
        x = y_masked.permute(0, 2, 1)

        # 2. Stem
        x = self.stem_conv(x)
        x = x.permute(0, 2, 1)
        x = self.stem_ln(x)
        x = self.stem_silu(x)
        x = x.permute(0, 2, 1)

        # 3. Backbone
        out = self.backbone(x)  # (B, out_dim, L)

        return out.permute(0, 2, 1)  # Return (B, L, out_dim)


class RHS_GFN(nn.Module):
    """
    Robust Hybrid-Stem Global-Feedback Network.
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------
        # 1. Static Encoder Components
        # ----------------------------------------------------------------
        # Input: 18 channels (4 seq + 3 struct + 7 loop + 4 partner_id)
        self.input_dim = 18

        # Hybrid Stem
        # Context branch output size = Backbone Growth Rate
        self.stem = HybridStem(self.input_dim, Config.BACKBONE_GROWTH_RATE)
        stem_out_dim = self.input_dim + Config.BACKBONE_GROWTH_RATE

        # Main Backbone
        self.backbone = DenseTCN(
            in_channels=stem_out_dim,
            growth_rate=Config.BACKBONE_GROWTH_RATE,
            dilations=Config.BACKBONE_DILATIONS,
            dropout=Config.BACKBONE_DROPOUT,
            out_dim=Config.LATENT_DIM,
        )

        # ----------------------------------------------------------------
        # 2. Feedback Components
        # ----------------------------------------------------------------
        self.feedback_module = FeedbackModule(
            growth_rate=Config.FEEDBACK_GROWTH_RATE, out_dim=Config.FEEDBACK_DIM
        )

        # ----------------------------------------------------------------
        # 3. Interaction & Aggregation
        # ----------------------------------------------------------------
        # Feature dim per site = Z (64) + E_fb (32) = 96
        self.site_dim = Config.LATENT_DIM + Config.FEEDBACK_DIM

        # RNN Input = Self (96) + Partner (96) = 192
        self.rnn_input_dim = self.site_dim * 2

        self.rnn = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=Config.RNN_HIDDEN_DIM,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=Config.RNN_BIDIRECTIONAL,
        )

        rnn_out_dim = (
            Config.RNN_HIDDEN_DIM * 2
            if Config.RNN_BIDIRECTIONAL
            else Config.RNN_HIDDEN_DIM
        )
        self.head = nn.Linear(rnn_out_dim, Config.NUM_TARGETS)

    def gather_partner_features(self, features, partner_indices):
        """
        Gathers features from partner positions.
        features: (B, L, C)
        partner_indices: (B, L) with -1 for unpaired
        """
        B, L, C = features.shape

        # Replace -1 with 0 for valid gathering
        safe_indices = partner_indices.clone()
        mask = (safe_indices != -1).unsqueeze(-1).float()  # (B, L, 1)
        safe_indices[safe_indices == -1] = 0

        # Expand indices for gather: (B, L, C)
        # We gather along dim 1 (sequence length)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, C)

        partner_features = torch.gather(features, 1, gather_indices)

        # Apply mask (zero out features where partner_index was -1)
        partner_features = partner_features * mask

        return partner_features

    def forward_pass(self, z, e_fb, partner_indices):
        """
        Runs the Interaction + RNN + Head block.
        z: Static latent (B, L, Z_dim)
        e_fb: Feedback embedding (B, L, FB_dim)
        partner_indices: (B, L)
        """
        # 1. Concatenate Self Features
        # (B, L, 96)
        self_features = torch.cat([z, e_fb], dim=2)

        # 2. Gather Partner Features
        partner_features = self.gather_partner_features(self_features, partner_indices)

        # 3. Fuse
        # (B, L, 192)
        rnn_input = torch.cat([self_features, partner_features], dim=2)

        # 4. RNN
        rnn_out, _ = self.rnn(rnn_input)

        # 5. Head
        preds = self.head(rnn_out)

        return preds

    def forward(self, inputs, partner_indices, targets=None):
        """
        Forward method implementing the 2-pass iterative refinement loop.
        inputs: (B, L, 18)
        partner_indices: (B, L)
        targets: Optional, for compatibility
        """
        B, L, _ = inputs.shape

        # ----------------------------------------------------------------
        # Step 1: Static Encoding
        # ----------------------------------------------------------------
        # Permute for CNN: (B, 18, L)
        x = inputs.permute(0, 2, 1)

        # Stem
        x = self.stem(x)

        # Backbone
        z_conv = self.backbone(x)  # (B, Z_dim, L)
        z = z_conv.permute(0, 2, 1)  # (B, L, Z_dim)

        # ----------------------------------------------------------------
        # Step 2: Iterative Refinement
        # ----------------------------------------------------------------

        # --- Pass 1 ---
        # Initialize y_0 as zeros
        y_0 = torch.zeros((B, L, Config.NUM_TARGETS), device=inputs.device)

        # Feedback Module
        e_fb_0 = self.feedback_module(y_0)  # (B, L, FB_dim)

        # Predict
        y_1 = self.forward_pass(z, e_fb_0, partner_indices)

        # --- Pass 2 ---
        # Detach gradients from Pass 1 predictions for the input of Pass 2
        r = y_1.detach()

        # Feedback Module
        e_fb_1 = self.feedback_module(r)

        # Predict
        y_2 = self.forward_pass(z, e_fb_1, partner_indices)

        # Return both predictions for loss calculation (y_2 + 0.5 * y_1)
        return y_2, y_1
