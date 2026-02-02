import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class LayerNormChannels(nn.Module):
    """
    Applies LayerNorm to a tensor of shape (N, C, L).
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (N, C, L) -> (N, L, C)
        x = x.transpose(1, 2)
        x = self.norm(x)
        # (N, L, C) -> (N, C, L)
        return x.transpose(1, 2)


class HybridInputStem(nn.Module):
    """
    Hybrid Input Stem that preserves raw identity while extracting spatial context.
    Branch A: Identity (Raw Features)
    Branch B: Spatial Convolution -> LayerNorm -> SiLU
    """

    def __init__(self, in_channels, context_channels):
        super().__init__()
        self.branch_b_conv = nn.Conv1d(
            in_channels, context_channels, kernel_size=3, padding=1
        )
        self.norm = LayerNormChannels(context_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (N, C_in, L)

        # Branch B: Context
        ctx = self.branch_b_conv(x)
        ctx = self.norm(ctx)
        ctx = self.act(ctx)

        # Concatenate Branch A (x) and Branch B (ctx)
        # Output dim: C_in + context_channels
        return torch.cat([x, ctx], dim=1)


class PostActDenseBlock(nn.Module):
    """
    Single-Layer Dilated Block with Dense Connections.
    Structure: LN -> SiLU -> Dilated Conv (k=3) -> LN -> SiLU -> Pointwise Conv (k=1) -> Dropout
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super().__init__()
        # 1. Spatial Aggregation: Compress input stack to growth_rate
        self.norm1 = LayerNormChannels(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )

        # 2. Channel Mixing
        self.norm2 = LayerNormChannels(growth_rate)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, in_channels, L)
        out = self.norm1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out = self.dropout(out)

        # Dense Connection: Concatenate input history with new features
        return torch.cat([x, out], dim=1)


class DenseDilatedBackbone(nn.Module):
    """
    Stack of PostActDenseBlocks with exponentially increasing dilation.
    """

    def __init__(self, in_channels, growth_rate, dilations, latent_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList()
        curr_channels = in_channels

        for d in dilations:
            block = PostActDenseBlock(
                curr_channels, growth_rate, kernel_size=3, dilation=d, dropout=dropout
            )
            self.blocks.append(block)
            curr_channels += growth_rate

        # Latent Projection to Z
        self.final_conv = nn.Conv1d(curr_channels, latent_dim, kernel_size=1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_conv(x)


class FeedbackStem(nn.Module):
    """
    Processes recycled predictions.
    Applies strict channel masking and extracts features via a lightweight DenseNet.
    """

    def __init__(self, in_channels=5, hidden_dim=32, growth_rate=12, layers=4):
        super().__init__()
        # Spatial Stem
        self.stem_conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.stem_norm = LayerNormChannels(hidden_dim)
        self.stem_act = nn.SiLU()

        # Lightweight Dense TCN
        self.blocks = nn.ModuleList()
        curr_channels = hidden_dim
        dilations = [2**i for i in range(layers)]

        for d in dilations:
            block = PostActDenseBlock(
                curr_channels, growth_rate, kernel_size=3, dilation=d, dropout=0.1
            )
            self.blocks.append(block)
            curr_channels += growth_rate

        # Output Projection
        self.out_conv = nn.Conv1d(
            curr_channels, 32, kernel_size=1
        )  # Fixed to 32 per spec

    def forward(self, y):
        # y: (N, 5, L)

        # 1. Strict Channel Masking
        # Keep indices 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
        # Zero out indices 2 (deg_pH10), 4 (deg_50C)
        mask = torch.tensor([1, 1, 0, 1, 0], device=y.device, dtype=y.dtype).view(
            1, 5, 1
        )
        y_masked = y * mask

        # 2. Spatial Stem
        x = self.stem_conv(y_masked)
        x = self.stem_norm(x)
        x = self.stem_act(x)

        # 3. Dense Blocks
        for block in self.blocks:
            x = block(x)

        # 4. Projection to E_fb
        return self.out_conv(x)


class InteractionModule(nn.Module):
    """
    Fuses Static (Z) and Dynamic (E_fb) features.
    Performs Augmented Gather for partners and Global Aggregation via RNN.
    """

    def __init__(self, dim_z=64, dim_fb=32, rnn_hidden=64, num_targets=5):
        super().__init__()
        # Input to RNN is (Z + E_fb) for Self + (Z + E_fb) for Partner
        input_dim = (dim_z + dim_fb) * 2

        self.rnn = nn.GRU(
            input_dim, rnn_hidden, num_layers=1, batch_first=True, bidirectional=True
        )

        # Output of Bidirectional GRU is 2 * rnn_hidden
        self.proj = nn.Linear(rnn_hidden * 2, num_targets)

    def forward(self, z, e_fb, partner_indices):
        # z: (N, 64, L)
        # e_fb: (N, 32, L)
        # partner_indices: (N, L)

        # 1. Concatenate Self: [Z, E_fb]
        h = torch.cat([z, e_fb], dim=1)  # (N, 96, L)

        # 2. Gather Partner
        batch_size, channels, length = h.shape

        # Handle -1 in partner_indices (unpaired) by replacing with 0 temporarily
        p_idx_safe = partner_indices.clone()
        mask_unpaired = p_idx_safe == -1
        p_idx_safe[mask_unpaired] = 0

        # Expand indices for gather: (N, L) -> (N, C, L)
        p_idx_expanded = p_idx_safe.unsqueeze(1).expand(-1, channels, -1)

        # Gather partner features
        h_partner = torch.gather(h, 2, p_idx_expanded)

        # Apply Zero-Mask to unpaired bases
        mask_expanded = mask_unpaired.unsqueeze(1).expand(-1, channels, -1)
        h_partner[mask_expanded] = 0.0

        # 3. Fusion: [Self, Partner]
        h_combined = torch.cat([h, h_partner], dim=1)  # (N, 192, L)

        # 4. Global Aggregation (RNN)
        # RNN expects (N, L, C)
        h_combined = h_combined.permute(0, 2, 1)

        rnn_out, _ = self.rnn(h_combined)  # (N, L, 2*hidden)

        # 5. Projection
        out = self.proj(rnn_out)  # (N, L, 5)

        # Permute back to (N, 5, L)
        return out.permute(0, 2, 1)


class HI_GFDN(nn.Module):
    """
    Hybrid-Input Global-Feedback Dense Network.
    Assembles all components.
    """

    def __init__(self):
        super().__init__()

        # Hyperparameters from config
        self.seq_len = config.SEQ_LEN
        self.num_targets = config.NUM_TARGETS  # 3 scored, but we predict 5
        self.total_targets = 5

        # 1. Hybrid Input Stem
        # Input channels = 18 (4 Seq + 3 Struct + 7 Loop + 4 Partner)
        # Context channels = 32 (Arbitrary choice for Branch B, matching growth rate)
        self.input_stem = HybridInputStem(
            in_channels=18, context_channels=config.GROWTH_RATE
        )

        # 2. Main Backbone
        # Input to backbone is 18 + 32 = 50
        self.backbone = DenseDilatedBackbone(
            in_channels=18 + config.GROWTH_RATE,
            growth_rate=config.GROWTH_RATE,
            dilations=config.DILATIONS,
            latent_dim=config.LATENT_DIM,
            dropout=config.DROPOUT,
        )

        # 3. Feedback Stem
        self.feedback_stem = FeedbackStem(
            in_channels=5, hidden_dim=32, growth_rate=12, layers=4
        )

        # 4. Interaction Module
        self.interaction = InteractionModule(
            dim_z=config.LATENT_DIM,
            dim_fb=config.FEEDBACK_DIM,  # 32
            rnn_hidden=config.RNN_HIDDEN,
            num_targets=5,  # We predict all 5 columns
        )

    def forward(self, x, partner_indices, y_prev=None):
        """
        Args:
            x: (N, 18, L) Input features
            partner_indices: (N, L) Pairing map
            y_prev: (N, 5, L) Previous predictions for feedback loop.
                    If None, initializes with zeros.
        """
        # 1. Static Feature Extraction
        # Run Hybrid Stem
        x_hybrid = self.input_stem(x)

        # Run Backbone -> Z
        z = self.backbone(x_hybrid)

        # 2. Feedback Processing
        if y_prev is None:
            batch_size, _, length = x.shape
            y_prev = torch.zeros(
                (batch_size, 5, length), device=x.device, dtype=x.dtype
            )

        e_fb = self.feedback_stem(y_prev)

        # 3. Interaction & Prediction
        y_pred = self.interaction(z, e_fb, partner_indices)

        return y_pred
