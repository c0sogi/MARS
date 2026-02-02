import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DecoupledBlock(nn.Module):
    """
    Decoupled Block Structure:
    Dilated Conv (k=3) -> Norm -> ReLU -> Pointwise Conv (k=1) -> Norm -> ReLU

    This block separates spatial aggregation (via dilated convolution) from
    channel mixing (via pointwise convolution), as identified in Lesson 00111.
    """

    def __init__(self, in_channels, out_channels, dilation, dropout=0.0):
        super().__init__()
        # Spatial Aggregation
        self.spatial_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ReLU()

        # Channel Mixing
        self.channel_conv = nn.Conv1d(
            out_channels, out_channels, kernel_size=1, bias=False
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.spatial_conv(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.channel_conv(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.dropout(out)
        return out


class DenseDilatedBackbone(nn.Module):
    """
    Static backbone utilizing Dense Connections.
    Features from all preceding blocks are concatenated as input to the next block.
    """

    def __init__(self, config):
        super().__init__()
        self.in_channels = config.input_channels
        self.growth_rate = config.backbone_channels
        self.dilations = config.backbone_dilations
        self.dropout = config.backbone_dropout

        self.blocks = nn.ModuleList()

        # Initial stem projection
        self.stem = nn.Conv1d(self.in_channels, self.growth_rate, kernel_size=1)

        curr_channels = self.growth_rate

        for d in self.dilations:
            # Input to block is concatenation of all previous features
            block = DecoupledBlock(
                curr_channels, self.growth_rate, dilation=d, dropout=self.dropout
            )
            self.blocks.append(block)
            # Update channel count for next block (DenseNet growth)
            curr_channels += self.growth_rate

        # Final projection to latent dimension
        self.final_proj = nn.Conv1d(curr_channels, config.latent_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, C, L)
        stem_out = self.stem(x)

        # List to store feature maps for dense concatenation
        features = [stem_out]

        for block in self.blocks:
            # Concatenate all previous features along channel dimension
            dense_input = torch.cat(features, dim=1)
            new_feat = block(dense_input)
            features.append(new_feat)

        # Final concatenation of all features
        total_out = torch.cat(features, dim=1)

        # Project to latent space
        z = self.final_proj(total_out)
        return z


class FeedbackEncoder(nn.Module):
    """
    Lightweight Dilated TCN for processing feedback and topology features.
    Uses Residual connections (Add) instead of Dense to maintain parameter efficiency.
    """

    def __init__(self, config):
        super().__init__()
        self.in_channels = config.feedback_input_channels
        self.out_channels = config.feedback_channels
        self.dilations = config.feedback_dilations

        # Initial projection
        self.stem = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)

        self.blocks = nn.ModuleList()
        for d in self.dilations:
            # Residual block using the same Decoupled structure
            self.blocks.append(
                DecoupledBlock(self.out_channels, self.out_channels, dilation=d)
            )

    def forward(self, x):
        out = self.stem(x)
        for block in self.blocks:
            # Residual connection
            out = out + block(out)
        return out


class TISRNModel(nn.Module):
    """
    Topology-Informed Stabilized Recurrent Network (TI-SRN).

    Features:
    1. Static Dense Dilated Backbone.
    2. Iterative Feedback Loop injecting recycled predictions + raw topology.
    3. Partner-Aware Interaction gathering.
    4. Bidirectional GRU Aggregation.
    """

    def __init__(self, config=Config()):
        super().__init__()
        self.config = config

        # 1. Main Backbone
        self.backbone = DenseDilatedBackbone(config)

        # 2. Feedback Encoder
        self.feedback_encoder = FeedbackEncoder(config)

        # 3. Aggregation Head
        # Input to RNN is concatenation of Self (Latent+Feedback) and Partner (Latent+Feedback)
        rnn_input_dim = (config.latent_dim + config.feedback_channels) * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=config.rnn_hidden_dim,
            num_layers=config.rnn_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

        rnn_out_dim = (
            config.rnn_hidden_dim * 2 if config.bidirectional else config.rnn_hidden_dim
        )
        self.head = nn.Linear(rnn_out_dim, 5)

    def forward(self, x, partner_indices, mask=None):
        """
        Args:
            x (torch.Tensor): Input features (B, L, 18).
            partner_indices (torch.Tensor): Indices of paired bases (B, L).
            mask (torch.Tensor, optional): Valid position mask (B, L).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Predictions_Pass1, Predictions_Pass2)
            Each shape (B, L, 5).
        """
        # Permute input to (Batch, Channels, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)

        B, C, L = x.shape

        # --- 1. Compute Static Backbone Features ---
        # Z: (B, 64, L)
        z = self.backbone(x)

        # --- 2. Extract Raw Topology Features ---
        # Config channels: Seq(0-4), Struct(4-7), Loop(7-14), Partner(14-18)
        # We want Structure + Loop = Indices 4 to 14
        topo_features = x[:, 4:14, :]  # (B, 10, L)

        # --- 3. Pass 1: Zero Initialization ---
        # Initialize predictions with zeros
        y_hat_0 = torch.zeros(B, 5, L, device=x.device, dtype=x.dtype)

        # Prepare Feedback Input: [Predictions, Topology]
        fb_in_1 = torch.cat([y_hat_0, topo_features], dim=1)  # (B, 15, L)

        # Encode Feedback
        e_fb_1 = self.feedback_encoder(fb_in_1)  # (B, 32, L)

        # Predict
        y_hat_1 = self._predict_head(z, e_fb_1, partner_indices)  # (B, 5, L)

        # --- 4. Pass 2: Refinement ---
        # Detach Pass 1 predictions to stop gradients flowing into initialization
        r = y_hat_1.detach()

        # Apply mask to recycled predictions if provided
        if mask is not None:
            # mask: (B, L) -> (B, 5, L)
            mask_expanded = mask.unsqueeze(1).float()
            r = r * mask_expanded

        # Prepare Feedback Input
        fb_in_2 = torch.cat([r, topo_features], dim=1)

        # Encode Feedback
        e_fb_2 = self.feedback_encoder(fb_in_2)

        # Predict
        y_hat_2 = self._predict_head(z, e_fb_2, partner_indices)  # (B, 5, L)

        # Permute to (B, L, 5) for output
        return y_hat_1.permute(0, 2, 1), y_hat_2.permute(0, 2, 1)

    def _predict_head(self, z, e_fb, partner_indices):
        """
        Helper to perform Interaction, RNN, and Linear Projection.

        Args:
            z: Backbone latent (B, 64, L)
            e_fb: Feedback embeddings (B, 32, L)
            partner_indices: (B, L)

        Returns:
            logits: (B, 5, L)
        """
        B, _, L = z.shape

        # 1. Construct Self Vector: [Z, E_fb]
        # (B, 96, L)
        self_vec = torch.cat([z, e_fb], dim=1)

        # 2. Gather Partner Vector
        # Handle unpaired indices (-1) by replacing with 0 and masking later
        p_idx = partner_indices.clone()
        unpaired_mask = p_idx == -1  # (B, L)
        p_idx[unpaired_mask] = 0

        # Expand indices for gathering: (B, L) -> (B, 96, L)
        C_total = self_vec.shape[1]
        p_idx_expanded = p_idx.unsqueeze(1).expand(-1, C_total, -1)

        # Gather features from partner positions
        partner_vec = torch.gather(self_vec, 2, p_idx_expanded)

        # Zero out features for unpaired bases
        mask_expanded = unpaired_mask.unsqueeze(1).expand(-1, C_total, -1)
        partner_vec[mask_expanded] = 0.0

        # 3. Concatenate Self and Partner
        # (B, 192, L)
        combined = torch.cat([self_vec, partner_vec], dim=1)

        # 4. Bidirectional GRU
        # GRU expects (B, L, C)
        rnn_in = combined.permute(0, 2, 1)
        rnn_out, _ = self.rnn(rnn_in)

        # 5. Linear Head
        # rnn_out: (B, L, Hidden*2)
        logits = self.head(rnn_out)  # (B, L, 5)

        # Permute back to (B, 5, L) for channel-first processing in feedback loop
        return logits.permute(0, 2, 1)
