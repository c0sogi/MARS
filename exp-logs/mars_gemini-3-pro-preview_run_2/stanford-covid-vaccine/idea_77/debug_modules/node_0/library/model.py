import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class Permute(nn.Module):
    """
    Helper module to permute tensor dimensions for LayerNorm.
    Permutes (B, C, L) -> (B, L, C) and vice versa.
    """

    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class DenseBlock(nn.Module):
    """
    Single-Layer Dilated Block with Post-Activation and Dense Connectivity.
    Structure: Conv(k=3) -> LN -> SiLU -> Conv(k=1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, out_channels, dilation, dropout=0.1):
        super(DenseBlock, self).__init__()

        # 1. Dilated Convolution (Kernel 3)
        # Padding is set to dilation to maintain sequence length (k=3, pad=d)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.ln1 = nn.LayerNorm(out_channels)
        self.act1 = nn.SiLU()

        # 2. Pointwise Convolution (Kernel 1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.ln2 = nn.LayerNorm(out_channels)
        self.act2 = nn.SiLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C_in, L)

        # Block 1
        out = self.conv1(x)  # (B, C_out, L)
        out = out.permute(0, 2, 1)  # (B, L, C_out)
        out = self.ln1(out)
        out = self.act1(out)
        out = out.permute(0, 2, 1)  # (B, C_out, L)

        # Block 2
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)  # (B, C_out, L)

        return out


class DenseTCN(nn.Module):
    """
    Stack of DenseBlocks with dense connections.
    """

    def __init__(self, in_channels, growth_rate, dilations, dropout=0.1):
        super(DenseTCN, self).__init__()
        self.blocks = nn.ModuleList()
        self.growth_rate = growth_rate

        current_in_channels = in_channels

        for d in dilations:
            block = DenseBlock(
                in_channels=current_in_channels,
                out_channels=growth_rate,
                dilation=d,
                dropout=dropout,
            )
            self.blocks.append(block)
            current_in_channels += growth_rate

    def forward(self, x):
        # x: (B, C, L)
        features = [x]

        for block in self.blocks:
            # Concatenate all previous features
            in_feat = torch.cat(features, dim=1)
            out = block(in_feat)
            features.append(out)

        # Return concatenation of all features (or just the processed ones if preferred)
        # The strategy says "concatenating outputs of all prior blocks" for the *next* layer.
        # For the final output, we usually aggregate.
        return torch.cat(features, dim=1)


class ADFRN(nn.Module):
    """
    Anchored Dense-Feedback Recurrent Network (ADF-RN).
    """

    def __init__(self):
        super(ADFRN, self).__init__()

        # ====================
        # 1. Main Backbone
        # ====================
        self.input_stem = nn.Conv1d(
            Config.INPUT_CHANNELS, Config.BACKBONE_GROWTH_RATE, kernel_size=3, padding=1
        )

        # Backbone Dense TCN
        # Input to first block is Stem (Growth Rate)
        # Subsequent blocks take accumlated channels
        self.backbone = DenseTCN(
            in_channels=Config.BACKBONE_GROWTH_RATE,
            growth_rate=Config.BACKBONE_GROWTH_RATE,
            dilations=Config.DILATIONS,
            dropout=Config.DROPOUT,
        )

        # Calculate output channels of backbone
        # Stem (64) + 6 layers * 64 = 448
        backbone_out_channels = Config.BACKBONE_GROWTH_RATE + (
            len(Config.DILATIONS) * Config.BACKBONE_GROWTH_RATE
        )

        self.latent_proj = nn.Conv1d(
            backbone_out_channels, Config.LATENT_DIM, kernel_size=1
        )

        # ====================
        # 2. Feedback Module
        # ====================
        self.feedback_stem = nn.Conv1d(
            Config.NUM_TARGETS, Config.FEEDBACK_GROWTH_RATE, kernel_size=3, padding=1
        )

        self.feedback_net = DenseTCN(
            in_channels=Config.FEEDBACK_GROWTH_RATE,
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            dilations=Config.DILATIONS,  # Use same dilation pattern
            dropout=Config.DROPOUT,
        )

        # Stem (16) + 6 layers * 16 = 112
        feedback_out_channels = Config.FEEDBACK_GROWTH_RATE + (
            len(Config.DILATIONS) * Config.FEEDBACK_GROWTH_RATE
        )

        self.feedback_proj = nn.Conv1d(
            feedback_out_channels, Config.FEEDBACK_OUT_DIM, kernel_size=1
        )

        # ====================
        # 3. Aggregation
        # ====================
        # Interaction Input: Latent (64) + Feedback (32) = 96
        # Paired Input: Self (96) + Partner (96) = 192
        self.interaction_dim = Config.LATENT_DIM + Config.FEEDBACK_OUT_DIM
        rnn_input_dim = self.interaction_dim * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.RNN_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
        )

        rnn_output_dim = Config.RNN_HIDDEN_DIM * (2 if Config.BIDIRECTIONAL else 1)

        self.head = nn.Linear(rnn_output_dim, Config.NUM_TARGETS)

        # Identify indices to mask
        self.register_buffer("mask_indices", self._get_mask_indices())

    def _get_mask_indices(self):
        """Returns a tensor of indices for unscored columns."""
        indices = []
        for i, col in enumerate(Config.TARGET_COLS):
            if col in Config.UNSCORED_COLS:
                indices.append(i)
        return torch.tensor(indices, dtype=torch.long)

    def _gather_partner_features(self, x, partner_indices):
        """
        Gathers features from partner positions.
        x: (B, C, L)
        partner_indices: (B, L) with -1 for unpaired
        """
        B, C, L = x.shape

        # Replace -1 with 0 for valid gathering (we will mask later)
        # Clone to avoid modifying original tensor
        idx = partner_indices.clone()
        unpaired_mask = idx == -1  # (B, L)
        idx[unpaired_mask] = 0

        # Expand indices for gather: (B, C, L)
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)

        # Gather
        gathered = torch.gather(x, 2, idx_expanded)

        # Apply mask: Set features to 0 where unpaired
        # mask shape (B, 1, L)
        mask = (~unpaired_mask).unsqueeze(1).float()
        gathered = gathered * mask

        return gathered

    def decode(self, z, y_prev, partner_indices):
        """
        Runs the decoder part: Feedback -> Interaction -> RNN -> Head
        z: Backbone latent features (B, Latent_Dim, L)
        y_prev: Previous predictions (B, Num_Targets, L)
        """
        B, _, L = z.shape

        # 1. Feedback Processing
        # Apply channel mask
        y_masked = y_prev.clone()
        if self.mask_indices.numel() > 0:
            y_masked[:, self.mask_indices, :] = 0.0

        fb_h = self.feedback_stem(y_masked)
        fb_feat = self.feedback_net(fb_h)
        e_fb = self.feedback_proj(fb_feat)  # (B, 32, L)

        # 2. Interaction
        # Concatenate Self Latent + Feedback
        node_feat = torch.cat([z, e_fb], dim=1)  # (B, 96, L)

        # Gather Partner Features
        partner_feat = self._gather_partner_features(node_feat, partner_indices)

        # Concatenate Self + Partner
        combined = torch.cat([node_feat, partner_feat], dim=1)  # (B, 192, L)

        # 3. RNN Aggregation
        # Permute for RNN: (B, L, C)
        combined_perm = combined.permute(0, 2, 1)

        rnn_out, _ = self.rnn(combined_perm)  # (B, L, Hidden*2)

        # 4. Head
        out = self.head(rnn_out)  # (B, L, 5)

        # Permute back to (B, 5, L) for consistency if needed,
        # but Loss expects (B, L, 5). Let's return (B, L, 5).
        return out

    def forward(self, x, partner_indices):
        """
        Args:
            x: (B, L, 18) Input features
            partner_indices: (B, L) Partner map
        Returns:
            y1: (B, L, 5) Pass 1 predictions
            y2: (B, L, 5) Pass 2 predictions
        """
        # Permute input to (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)

        # ====================
        # 1. Backbone (Static)
        # ====================
        h = self.input_stem(x)
        features = self.backbone(h)
        z = self.latent_proj(features)  # (B, 64, L)

        # ====================
        # 2. Pass 1 (Zero Feedback)
        # ====================
        B, _, L = z.shape
        y_init = torch.zeros((B, Config.NUM_TARGETS, L), device=x.device, dtype=x.dtype)

        y1 = self.decode(z, y_init, partner_indices)  # Returns (B, L, 5)

        # ====================
        # 3. Pass 2 (Recycled Feedback)
        # ====================
        # Detach gradients from Pass 1 output for feedback input
        # Permute y1 to (B, 5, L) for feedback input
        y1_detached = y1.detach().permute(0, 2, 1)

        y2 = self.decode(z, y1_detached, partner_indices)  # Returns (B, L, 5)

        return y1, y2
