import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridInputStem(nn.Module):
    """
    Splits input into two parallel branches:
    1. Identity: Preserves raw one-hot features.
    2. Context: Extracts local n-gram context via Spatial Conv -> LN -> SiLU.
    """

    def __init__(self, in_channels=18, context_channels=32):
        super().__init__()
        self.context_conv = nn.Conv1d(
            in_channels, context_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        # x: (N, C, L)

        # Branch B: Context
        ctx = self.context_conv(x)
        # LayerNorm expects (N, L, C)
        ctx = ctx.permute(0, 2, 1)
        ctx = F.layer_norm(ctx, ctx.shape[2:])
        ctx = F.silu(ctx)
        ctx = ctx.permute(0, 2, 1)

        # Concatenate Identity (Branch A) and Context (Branch B)
        out = torch.cat([x, ctx], dim=1)
        return out


class DenseDilatedBlock(nn.Module):
    """
    Single-Layer Dilated Block with Post-Activation structure:
    Conv3x3 -> LN -> SiLU -> Conv1x1 -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout):
        super().__init__()
        self.conv3x3 = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv1x1 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. Spatial Aggregation
        out = self.conv3x3(x)

        # LN + SiLU (Post-Activation)
        out = out.permute(0, 2, 1)
        out = F.layer_norm(out, out.shape[2:])
        out = F.silu(out)
        out = out.permute(0, 2, 1)

        # 2. Channel Mixing
        out = self.conv1x1(out)

        # LN + SiLU
        out = out.permute(0, 2, 1)
        out = F.layer_norm(out, out.shape[2:])
        out = F.silu(out)
        out = out.permute(0, 2, 1)

        out = self.dropout(out)
        return out


class DenseTCN(nn.Module):
    """
    Stack of dilated blocks with Dense Connections.
    Input to block i is the concatenation of inputs/outputs of blocks 0...i-1.
    """

    def __init__(self, in_channels, growth_rate, dilations, out_channels, dropout):
        super().__init__()
        self.blocks = nn.ModuleList()

        # Track current input dimension as features accumulate
        curr_dim = in_channels
        for d in dilations:
            blk = DenseDilatedBlock(curr_dim, growth_rate, d, dropout)
            self.blocks.append(blk)
            curr_dim += growth_rate

        # Final projection to latent dimension
        self.projection = nn.Conv1d(curr_dim, out_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for blk in self.blocks:
            # Dense connection: concatenate all previous features
            inp = torch.cat(features, dim=1)
            out = blk(inp)
            features.append(out)

        # Project the final accumulated state
        total_features = torch.cat(features, dim=1)
        z = self.projection(total_features)
        return z


class FeedbackProcessor(nn.Module):
    """
    Processes recycled predictions.
    Spatial Stem -> Lightweight Dense TCN.
    """

    def __init__(
        self,
        in_channels=5,
        stem_channels=16,
        growth_rate=16,
        dilations=[1, 2, 4, 8],
        out_channels=32,
        dropout=0.1,
    ):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, stem_channels, kernel_size=3, padding=1)
        self.backbone = DenseTCN(
            in_channels=stem_channels,
            growth_rate=growth_rate,
            dilations=dilations,
            out_channels=out_channels,
            dropout=dropout,
        )

    def forward(self, y):
        # y: (N, 5, L) - already masked

        # Spatial Stem
        h = self.stem(y)
        h = h.permute(0, 2, 1)
        h = F.layer_norm(h, h.shape[2:])
        h = F.silu(h)
        h = h.permute(0, 2, 1)

        # Backbone
        e_fb = self.backbone(h)
        return e_fb


class InteractionHead(nn.Module):
    """
    Fuses Self and Partner features, processes with Bi-GRU, and projects to targets.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Input to RNN is (Self + Partner) -> input_dim * 2
        self.rnn = nn.GRU(
            input_size=input_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # Bidirectional output -> hidden_dim * 2
        self.head = nn.Linear(hidden_dim * 2, 5)

    def forward(self, h_self, pair_map):
        # h_self: (N, C, L) -> permute to (N, L, C) for RNN/Gather
        h_self = h_self.permute(0, 2, 1)
        N, L, C = h_self.shape

        # Gather partner features
        # pair_map: (N, L), values are indices, -1 is unpaired
        safe_indices = pair_map.clone()
        mask = (safe_indices != -1).unsqueeze(-1).float()  # (N, L, 1)
        safe_indices[safe_indices == -1] = 0  # Safe index for gather

        # Create batch indices for advanced indexing
        batch_indices = torch.arange(N, device=h_self.device).unsqueeze(1).expand(-1, L)

        # Gather: h_partner[b, i] = h_self[b, pair_map[b, i]]
        h_partner = h_self[batch_indices, safe_indices]

        # Mask unpaired positions
        h_partner = h_partner * mask

        # Concatenate Self + Partner
        rnn_in = torch.cat([h_self, h_partner], dim=2)  # (N, L, C*2)

        # Global Aggregation via Bi-GRU
        rnn_out, _ = self.rnn(rnn_in)

        # Final Projection
        preds = self.head(rnn_out)  # (N, L, 5)

        return preds


class AHCHDN(nn.Module):
    """
    Anchored High-Capacity Hybrid-Dense Network.
    Combines Static Backbone, Feedback Loop, and Interaction Head.
    """

    def __init__(self):
        super().__init__()

        # 1. Hybrid Input Stem
        # Input features: Seq(4) + Struct(3) + Loop(7) + Partner(4) = 18
        self.input_stem = HybridInputStem(in_channels=18, context_channels=32)

        # 2. Main Backbone (High Capacity)
        # Input to backbone: 18 (Identity) + 32 (Context) = 50
        self.backbone = DenseTCN(
            in_channels=50,
            growth_rate=Config.GROWTH_RATE,
            dilations=[1, 2, 4, 8, 16, 32],
            out_channels=Config.LATENT_DIM,
            dropout=Config.DROPOUT,
        )

        # 3. Feedback Module
        self.feedback_processor = FeedbackProcessor(
            in_channels=5,
            stem_channels=16,
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            dilations=[1, 2, 4, 8],
            out_channels=Config.FEEDBACK_DIM,
            dropout=Config.DROPOUT,
        )

        # 4. Interaction Head
        self.interaction_head = InteractionHead(
            input_dim=Config.LATENT_DIM + Config.FEEDBACK_DIM,
            hidden_dim=Config.HIDDEN_DIM,
        )

        # Feedback Mask: Keep channels 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
        # Zero out channels 2 (deg_pH10) and 4 (deg_50C)
        self.register_buffer(
            "feedback_mask",
            torch.tensor([1, 1, 0, 1, 0], dtype=torch.float32).view(1, 5, 1),
        )

    def forward(self, x, pair_map, y_prev=None):
        # x: (N, L, 18) -> Permute to (N, 18, L) for Conv1d
        x = x.permute(0, 2, 1)

        # pair_map: (N, L)
        # y_prev: (N, L, 5) or None

        # --- Static Path ---
        h_start = self.input_stem(x)
        z = self.backbone(h_start)  # (N, Latent, L)

        # --- Feedback Path ---
        if y_prev is None:
            # Initialize with zeros if no previous prediction
            y_prev = torch.zeros(
                (x.shape[0], 5, x.shape[2]), device=x.device, dtype=x.dtype
            )
        else:
            # y_prev comes in as (N, L, 5) -> Permute to (N, 5, L)
            y_prev = y_prev.permute(0, 2, 1)

        # Strictly mask unscored channels to prevent noise injection
        y_masked = y_prev * self.feedback_mask

        # Process feedback
        e_fb = self.feedback_processor(y_masked)  # (N, Feedback, L)

        # --- Interaction & Output ---
        # Combine Static Latent and Dynamic Feedback
        h_combined = torch.cat([z, e_fb], dim=1)  # (N, Latent+Feedback, L)

        # Run Interaction Head (Gather + RNN + Linear)
        preds = self.interaction_head(h_combined, pair_map)  # (N, L, 5)

        return preds
