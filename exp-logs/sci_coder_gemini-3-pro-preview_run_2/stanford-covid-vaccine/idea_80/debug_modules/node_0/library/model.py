import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import mask_unscored_channels


class LayerNorm1d(nn.Module):
    """
    Applies Layer Normalization to channels in (N, C, L) format.
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


class SpatialStem(nn.Module):
    """
    Initial spatial mixing layer: Conv1d(k=3) -> LN -> SiLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.op = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            LayerNorm1d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.op(x)


class DenseDilatedBlock(nn.Module):
    """
    Post-Activation Dense Block:
    Conv(k=3, d=d) -> LN -> SiLU -> Conv(k=1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, growth_rate, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm1 = LayerNorm1d(growth_rate)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=1)
        self.norm2 = LayerNorm1d(growth_rate)
        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.dropout(out)
        return out


class DenseTCN(nn.Module):
    """
    Temporal Convolutional Network with Dense Connections.
    Input to block i is concatenation of input and outputs of blocks 0..i-1.
    """

    def __init__(self, in_channels, growth_rate, dilations, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_dim = in_channels

        for d in dilations:
            block = DenseDilatedBlock(current_dim, growth_rate, d, dropout)
            self.blocks.append(block)
            current_dim += growth_rate

    def forward(self, x):
        features = [x]
        for block in self.blocks:
            # Dense connection: concatenate all previous features along channel dim
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Return concatenation of all features (DenseNet style)
        return torch.cat(features, dim=1)


class FeedbackModule(nn.Module):
    """
    Processes recycled predictions.
    Spatial Stem -> Lightweight Dense TCN -> Projection.
    """

    def __init__(self, input_dim, growth_rate, out_dim, dilations, dropout=0.1):
        super().__init__()
        # Spatial Feedback Stem
        self.stem = SpatialStem(input_dim, growth_rate, kernel_size=3)

        # Backbone: Lightweight Dense TCN
        self.backbone = DenseTCN(growth_rate, growth_rate, dilations, dropout)

        # Calculate output dim of DenseTCN: Stem + N_blocks * Growth
        tcn_out_dim = growth_rate + len(dilations) * growth_rate

        self.proj = nn.Conv1d(tcn_out_dim, out_dim, kernel_size=1)

    def forward(self, x):
        # x: (N, 5, L)
        x = self.stem(x)
        x = self.backbone(x)
        x = self.proj(x)
        return x


class AS_DRN(nn.Module):
    """
    Anchored Spatial-Dense Recurrent Network.
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------------------
        # 1. Input Dimensions
        # ----------------------------------------------------------------------
        # Seq(4) + Struct(3) + Loop(7) + PartnerID(4) = 18
        self.input_dim = 18

        # ----------------------------------------------------------------------
        # 2. Main Backbone (Static)
        # ----------------------------------------------------------------------
        self.main_growth = Config.MAIN_GROWTH_RATE
        self.dilations = Config.DILATIONS
        self.dropout = Config.DROPOUT

        # Spatial Stem: Map inputs to growth rate
        self.input_stem = SpatialStem(
            self.input_dim, self.main_growth, kernel_size=Config.KERNEL_SIZE
        )

        # Dense TCN
        self.main_backbone = DenseTCN(
            self.main_growth, self.main_growth, self.dilations, self.dropout
        )

        # Latent Projection
        # Output of DenseTCN is Stem + N * Growth
        main_out_dim = self.main_growth + len(self.dilations) * self.main_growth
        self.latent_proj = nn.Conv1d(main_out_dim, Config.LATENT_DIM, kernel_size=1)

        # ----------------------------------------------------------------------
        # 3. Feedback Module
        # ----------------------------------------------------------------------
        self.fb_growth = Config.FB_GROWTH_RATE
        self.fb_dim = Config.FEEDBACK_DIM

        self.feedback_module = FeedbackModule(
            input_dim=Config.NUM_TARGETS,  # 5
            growth_rate=self.fb_growth,
            out_dim=self.fb_dim,
            dilations=self.dilations,
            dropout=self.dropout,
        )

        # ----------------------------------------------------------------------
        # 4. Aggregation
        # ----------------------------------------------------------------------
        # Input to RNN: (Latent + Feedback) * 2 (Self + Partner)
        rnn_input_dim = (Config.LATENT_DIM + self.fb_dim) * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Linear(Config.HIDDEN_DIM * 2, Config.NUM_TARGETS)

    def forward_pass(self, z, fb_in, partner_indices):
        """
        Executes one pass of the interaction + aggregation.
        z: (N, Latent, L) - Static backbone features
        fb_in: (N, 5, L) - Masked predictions from previous step (or zeros)
        partner_indices: (N, L) - Indices of paired bases
        """
        # 1. Compute Feedback Embeddings
        e_fb = self.feedback_module(fb_in)  # (N, FB_Dim, L)

        # 2. Augmented Gather
        # Concatenate Self: [Z, E_fb] -> (N, Latent+FB, L)
        self_feat = torch.cat([z, e_fb], dim=1)

        # Prepare for gathering (Permute to N, L, C)
        self_feat_t = self_feat.transpose(1, 2)  # (N, L, C)
        batch_size, seq_len, _ = self_feat_t.shape

        # Gather Partner Features
        # Create batch indices for advanced indexing
        batch_idx = (
            torch.arange(batch_size, device=z.device).unsqueeze(1).expand(-1, seq_len)
        )
        partner_feat_t = self_feat_t[batch_idx, partner_indices]  # (N, L, C)

        # 3. Fusion
        # Concatenate Self + Partner
        combined = torch.cat([self_feat_t, partner_feat_t], dim=2)  # (N, L, C*2)

        # 4. Global Aggregation (RNN)
        rnn_out, _ = self.rnn(combined)  # (N, L, Hidden*2)

        # 5. Output
        logits = self.head(rnn_out)  # (N, L, 5)

        return logits

    def forward(self, inputs, partner_indices):
        """
        Main forward method implementing the Iterative Refinement Loop.
        inputs: (N, L, 18)
        partner_indices: (N, L)
        """
        # Permute inputs to (N, C, L) for Conv1d
        x = inputs.transpose(1, 2)

        # 1. Static Backbone (Compute Z once)
        x = self.input_stem(x)
        x = self.main_backbone(x)
        z = self.latent_proj(x)  # (N, Latent, L)

        batch_size, _, seq_len = z.shape

        # 2. Iterative Refinement Loop

        # --- Pass 1: Zero Feedback ---
        # Initialize feedback with zeros
        y_zero = torch.zeros((batch_size, Config.NUM_TARGETS, seq_len), device=z.device)
        y_1 = self.forward_pass(z, y_zero, partner_indices)  # Returns (N, L, 5)

        # --- Pass 2: Feedback from Pass 1 ---
        # Detach gradients to stop backprop through the feedback loop
        y_1_detached = y_1.detach()  # (N, L, 5)

        # Mask unscored channels to prevent noise injection
        # mask_unscored_channels expects (N, L, 5) based on utils.py logic
        y_1_masked = mask_unscored_channels(y_1_detached)  # (N, L, 5)

        # Transpose to (N, 5, L) for the Conv1d feedback module
        y_1_input = y_1_masked.transpose(1, 2)

        y_2 = self.forward_pass(z, y_1_input, partner_indices)  # Returns (N, L, 5)

        # Return both outputs.
        # Loss = MCRMSE(y_2) + 0.5 * MCRMSE(y_1)
        return y_2, y_1
