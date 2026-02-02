import torch
import torch.nn as nn
from library import config


class Permute(nn.Module):
    """
    Helper module to permute tensor dimensions.
    Useful for switching between (B, L, C) and (B, C, L) for Conv1d/LayerNorm.
    """

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class HybridStem(nn.Module):
    """
    Hybrid Input Stem.
    Branch A: Identity (Raw features).
    Branch B: Spatial Context (Conv3x3 -> LN -> SiLU).
    Output: Concatenation of Branch A and Branch B.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch_b = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            Permute((0, 2, 1)),  # (B, C, L) -> (B, L, C) for LN
            nn.LayerNorm(out_channels),
            Permute((0, 2, 1)),  # (B, L, C) -> (B, C, L)
            nn.SiLU(),
        )

    def forward(self, x):
        # x: (B, C, L)
        out_b = self.branch_b(x)
        # Concatenate along channel dimension
        return torch.cat([x, out_b], dim=1)


class DenseDilatedBlock(nn.Module):
    """
    Single-Layer Dilated Block with Post-Activation and Dense Connectivity structure.
    Structure: Conv(k=3, d=d) -> LN -> SiLU -> Conv(k=1) -> LN -> SiLU -> Dropout.
    """

    def __init__(self, in_channels, growth_rate, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                growth_rate,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            Permute((0, 2, 1)),
            nn.LayerNorm(growth_rate),
            Permute((0, 2, 1)),
            nn.SiLU(),
            nn.Conv1d(growth_rate, growth_rate, kernel_size=1),
            Permute((0, 2, 1)),
            nn.LayerNorm(growth_rate),
            Permute((0, 2, 1)),
            nn.SiLU(),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class DenseTCN(nn.Module):
    """
    High-Capacity Dense Dilated TCN.
    Manages a stack of DenseDilatedBlocks.
    Input to each block is the concatenation of ALL prior block outputs (and input).
    """

    def __init__(self, in_channels, growth_rate, dilations, latent_dim):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_in_channels = in_channels

        for d in dilations:
            blk = DenseDilatedBlock(current_in_channels, growth_rate, dilation=d)
            self.blocks.append(blk)
            # In DenseNet, the next block receives [input, out1, out2...]
            # So input channels increase by growth_rate at each step
            current_in_channels += growth_rate

        self.project = nn.Conv1d(current_in_channels, latent_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, C, L)
        features = [x]
        for block in self.blocks:
            # Concatenate all previous features along channel dim
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Final projection on concatenation of all features
        total_concat = torch.cat(features, dim=1)
        z = self.project(total_concat)
        return z


class FeedbackModule(nn.Module):
    """
    Global-Context Pure-Feedback Module.
    Processes recycled predictions.
    Applies strict channel masking to unscored targets.
    """

    def __init__(self):
        super().__init__()
        # Input is 5 channels (predictions)

        # Spatial Feedback Stem
        self.stem = nn.Sequential(
            nn.Conv1d(5, config.FEEDBACK_GROWTH_RATE, kernel_size=3, padding=1),
            Permute((0, 2, 1)),
            nn.LayerNorm(config.FEEDBACK_GROWTH_RATE),
            Permute((0, 2, 1)),
            nn.SiLU(),
        )

        # Lightweight Dense TCN Backbone
        self.backbone = DenseTCN(
            in_channels=config.FEEDBACK_GROWTH_RATE,
            growth_rate=config.FEEDBACK_GROWTH_RATE,
            dilations=config.DILATIONS,
            latent_dim=config.FEEDBACK_DIM,
        )

        # Indices of scored columns to keep: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        self.register_buffer(
            "keep_indices", torch.tensor(config.SCORED_COLS_INDICES, dtype=torch.long)
        )

    def forward(self, y_prev):
        # y_prev: (B, L, 5)

        # Masking: Zero out unscored channels
        y_masked = torch.zeros_like(y_prev)
        y_masked[:, :, self.keep_indices] = y_prev[:, :, self.keep_indices]

        # Permute to (B, C, L) for Conv1d
        x = y_masked.permute(0, 2, 1)

        x = self.stem(x)
        e_fb = self.backbone(x)  # (B, Feedback_Dim, L)
        return e_fb


class HCHSGFN(nn.Module):
    """
    High-Capacity Hybrid-Stem Global-Feedback Network.
    Integrates Static Backbone, Feedback Module, and Interaction Aggregation.
    """

    def __init__(self):
        super().__init__()

        # Input features: 18 (4 seq + 3 struct + 7 loop + 4 partner)
        self.input_dim = 18

        # --- 1. Static Path Components ---
        # Hybrid Stem
        self.hybrid_stem = HybridStem(self.input_dim, config.GROWTH_RATE)
        # Stem output dim = Input (18) + Branch B (Growth Rate)
        stem_out_dim = self.input_dim + config.GROWTH_RATE

        # Main Backbone
        self.main_backbone = DenseTCN(
            in_channels=stem_out_dim,
            growth_rate=config.GROWTH_RATE,
            dilations=config.DILATIONS,
            latent_dim=config.LATENT_DIM,
        )

        # --- 2. Feedback Path Components ---
        self.feedback_module = FeedbackModule()

        # --- 3. Interaction & Aggregation ---
        # Self Vector = Z (Latent) + E_fb (Feedback)
        self.pair_dim = config.LATENT_DIM + config.FEEDBACK_DIM

        # RNN Input = Self + Partner
        rnn_input_dim = self.pair_dim * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=config.RNN_HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Final Projection: Bi-GRU outputs 2 * hidden_size
        self.head = nn.Linear(config.RNN_HIDDEN_SIZE * 2, 5)

    def forward(self, x, p_idx, y_prev=None):
        """
        Args:
            x: (B, L, 18) Input features
            p_idx: (B, L) Partner indices (-1 for unpaired)
            y_prev: (B, L, 5) Previous predictions for feedback loop (optional)
        """
        B, L, _ = x.shape

        # --- 1. Static Path ---
        # Permute x to (B, C, L)
        x_t = x.permute(0, 2, 1)

        stem_out = self.hybrid_stem(x_t)  # (B, 82, L)
        z = self.main_backbone(stem_out)  # (B, 64, L)

        # --- 2. Feedback Path ---
        if y_prev is None:
            # Zero feedback for first pass
            e_fb = torch.zeros(
                B, config.FEEDBACK_DIM, L, device=x.device, dtype=x.dtype
            )
        else:
            e_fb = self.feedback_module(y_prev)  # (B, 32, L)

        # --- 3. Interaction & Aggregation ---
        # Concatenate Z and E_fb to form Self Vector
        # z: (B, 64, L), e_fb: (B, 32, L) -> self_feat_t: (B, 96, L)
        self_feat_t = torch.cat([z, e_fb], dim=1)

        # Permute back to (B, L, C) for gathering
        self_feat = self_feat_t.permute(0, 2, 1)  # (B, L, 96)

        # Augmented Gather (Partner Vector)
        # Handle -1 in p_idx by replacing with 0 temporarily, then masking result
        p_idx_safe = p_idx.clone()
        mask_unpaired = p_idx == -1
        p_idx_safe[mask_unpaired] = 0

        # Expand indices for gather: (B, L, C)
        idx_expanded = p_idx_safe.unsqueeze(-1).expand(-1, -1, self.pair_dim)

        # Gather partner features
        partner_feat = torch.gather(self_feat, 1, idx_expanded)

        # Null-Masking: Zero out features for unpaired bases
        partner_feat[mask_unpaired] = 0.0

        # Concatenate Self and Partner
        combined = torch.cat([self_feat, partner_feat], dim=2)  # (B, L, 192)

        # Global Aggregation (Bi-GRU)
        rnn_out, _ = self.rnn(combined)  # (B, L, 128)

        # Final Projection
        logits = self.head(rnn_out)  # (B, L, 5)

        return logits
