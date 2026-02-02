import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class LayerNorm1d(nn.Module):
    """
    Applies Layer Normalization to 1D data (N, C, L).
    """

    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: (N, C, L) -> (N, L, C)
        x = x.transpose(1, 2)
        x = self.norm(x)
        # (N, L, C) -> (N, C, L)
        return x.transpose(1, 2)


class DenseBlock(nn.Module):
    """
    Post-Activation Dense Block:
    Conv(k=3, d=d) -> LN -> SiLU -> Conv(k=1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilation, dropout):
        super().__init__()

        padding = (dilation * (kernel_size - 1)) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,  # Bias handled by LN usually, but keeping false for standard practice with Norm
        )
        self.ln1 = LayerNorm1d(growth_rate)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1, bias=False)
        self.ln2 = LayerNorm1d(growth_rate)
        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.ln2(out)
        out = self.act2(out)

        out = self.dropout(out)
        return out


class DilatedDenseNet(nn.Module):
    """
    Backbone consisting of stacked DenseBlocks with dense connections.
    """

    def __init__(self, in_channels, growth_rate, kernel_size, dilations, dropout):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.growth_rate = growth_rate

        current_in_channels = in_channels

        for d in dilations:
            block = DenseBlock(
                in_channels=current_in_channels,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            self.blocks.append(block)
            current_in_channels += growth_rate

        self.out_channels = current_in_channels

    def forward(self, x):
        # Dense connectivity: concat input + all previous outputs
        features = [x]
        for block in self.blocks:
            # Concatenate all previous features
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Return concatenation of all features (including original input)
        return torch.cat(features, dim=1)


class FeedbackModule(nn.Module):
    """
    Global-Context Pure-Feedback Module.
    Processes recycled predictions with strict channel masking.
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.in_channels = Config.FEEDBACK_IN_CHANNELS
        self.hidden_channels = Config.FEEDBACK_HIDDEN_CHANNELS
        self.backbone_growth = Config.FEEDBACK_BACKBONE_GROWTH_RATE
        self.layers = Config.FEEDBACK_LAYERS

        # Scored indices mask (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # Indices: 0, 1, 3. Indices to zero: 2, 4.
        self.register_buffer("mask_indices", torch.tensor([0, 1, 3], dtype=torch.long))

        # Spatial Stem
        self.stem_conv = nn.Conv1d(
            self.in_channels, self.backbone_growth, kernel_size=3, padding=1
        )
        self.stem_ln = LayerNorm1d(self.backbone_growth)
        self.stem_act = nn.SiLU()

        # Lightweight Backbone
        # Using fixed dilation 1 for lightweight feedback processing or small exponential
        dilations = [2**i for i in range(self.layers)]
        self.backbone = DilatedDenseNet(
            in_channels=self.backbone_growth,
            growth_rate=self.backbone_growth,
            kernel_size=3,
            dilations=dilations,
            dropout=Config.DROPOUT,
        )

        # Projection to hidden channels
        self.proj = nn.Conv1d(
            self.backbone.out_channels, self.hidden_channels, kernel_size=1
        )

    def forward(self, feedback):
        # feedback: (B, L, 5) or (B, 5, L) -> expect (B, L, 5) from loop, transpose here
        if feedback.shape[-1] == 5:
            feedback = feedback.transpose(1, 2)  # (B, 5, L)

        # 1. Strict Channel Masking
        # Create a mask of zeros
        masked_feedback = torch.zeros_like(feedback)
        # Copy only scored channels
        masked_feedback[:, self.mask_indices, :] = feedback[:, self.mask_indices, :]

        # 2. Spatial Stem
        x = self.stem_conv(masked_feedback)
        x = self.stem_ln(x)
        x = self.stem_act(x)

        # 3. Backbone
        x = self.backbone(x)

        # 4. Projection
        x = self.proj(x)  # (B, Hidden, L)

        return x


class RHIGFN(nn.Module):
    """
    Robust Hybrid-Input Global-Feedback Network.
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Hybrid Input Stem
        # =====================================================================
        self.in_channels = Config.IN_CHANNELS
        self.hybrid_growth = Config.BACKBONE_GROWTH_RATE

        # Branch B: Context
        self.stem_conv = nn.Conv1d(
            self.in_channels,
            self.hybrid_growth,
            kernel_size=Config.HYBRID_KERNEL_SIZE,
            padding=1,
        )
        self.stem_ln = LayerNorm1d(self.hybrid_growth)
        self.stem_act = nn.SiLU()

        # Input to backbone is Branch A (Raw) + Branch B (Context)
        self.backbone_in_channels = self.in_channels + self.hybrid_growth

        # =====================================================================
        # 2. Main Backbone
        # =====================================================================
        self.backbone = DilatedDenseNet(
            in_channels=self.backbone_in_channels,
            growth_rate=Config.BACKBONE_GROWTH_RATE,
            kernel_size=Config.BACKBONE_KERNEL_SIZE,
            dilations=Config.DILATIONS,
            dropout=Config.DROPOUT,
        )

        # Latent Projection
        self.latent_proj = nn.Conv1d(
            self.backbone.out_channels, Config.LATENT_DIM, kernel_size=1
        )

        # =====================================================================
        # 3. Feedback Module
        # =====================================================================
        self.feedback_module = FeedbackModule()

        # =====================================================================
        # 4. Interaction & Aggregation
        # =====================================================================
        # Feature dim per position = Latent(Z) + Feedback(E_fb)
        self.feature_dim = Config.LATENT_DIM + Config.FEEDBACK_HIDDEN_CHANNELS

        # Interaction input = Self + Partner
        self.rnn_in_dim = self.feature_dim * 2

        self.rnn = nn.GRU(
            input_size=self.rnn_in_dim,
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

    def forward(self, x, partner_idx, feedback=None):
        """
        Args:
            x: (B, L, In_Channels) - Input features
            partner_idx: (B, L) - Indices of paired bases (-1 if unpaired)
            feedback: (B, L, 5) - Previous predictions (optional)
        """
        B, L, C = x.shape

        # Permute for Conv1d: (B, C, L)
        x_in = x.transpose(1, 2)

        # ---------------------------------------------------------------------
        # 1. Hybrid Input Stem
        # ---------------------------------------------------------------------
        # Branch A: Raw Identity (x_in)
        # Branch B: Context
        x_ctx = self.stem_conv(x_in)
        x_ctx = self.stem_ln(x_ctx)
        x_ctx = self.stem_act(x_ctx)

        # Concatenate
        backbone_input = torch.cat([x_in, x_ctx], dim=1)

        # ---------------------------------------------------------------------
        # 2. Main Backbone (Static Z)
        # ---------------------------------------------------------------------
        z_dense = self.backbone(backbone_input)
        z = self.latent_proj(z_dense)  # (B, Latent, L)

        # ---------------------------------------------------------------------
        # 3. Feedback Processing
        # ---------------------------------------------------------------------
        if feedback is None:
            # Initialize with zeros
            feedback = torch.zeros(
                (B, L, Config.NUM_TARGETS), device=x.device, dtype=x.dtype
            )

        e_fb = self.feedback_module(feedback)  # (B, Fb_Hidden, L)

        # ---------------------------------------------------------------------
        # 4. Interaction & Aggregation
        # ---------------------------------------------------------------------
        # Combine Static Z and Dynamic Feedback
        # Shape: (B, Latent+Fb_Hidden, L)
        features = torch.cat([z, e_fb], dim=1)

        # Permute back to (B, L, Feat) for gathering
        features_t = features.transpose(1, 2)

        # Prepare Partner Gathering
        # Replace -1 with 0 for valid gather indices
        gather_idx = partner_idx.clone()
        mask_unpaired = gather_idx == -1
        gather_idx[mask_unpaired] = 0

        # Expand indices for gather: (B, L, Feat)
        gather_idx_exp = gather_idx.unsqueeze(-1).expand(-1, -1, self.feature_dim)

        # Gather partner vectors
        partner_vecs = torch.gather(features_t, 1, gather_idx_exp)

        # Mask unpaired positions (set partner vec to 0)
        partner_vecs[mask_unpaired] = 0.0

        # Concatenate Self + Partner
        # (B, L, Feat*2)
        rnn_input = torch.cat([features_t, partner_vecs], dim=2)

        # RNN Aggregation
        rnn_out, _ = self.rnn(rnn_input)

        # Final Projection
        logits = self.head(rnn_out)  # (B, L, 5)

        return logits
