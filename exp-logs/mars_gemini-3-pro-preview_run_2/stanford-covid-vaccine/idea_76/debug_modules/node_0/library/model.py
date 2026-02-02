import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import (
    HybridInputStem,
    DenseDilatedBlock,
    FeedbackStem,
    PermuteLayerNorm,
)


class AHCHIDN(nn.Module):
    """
    Anchored High-Capacity Hybrid-Input Dense Network (AHC-HIDN).

    Architecture:
    1. Hybrid Input Stem: Splits input into Identity (Raw) and Context (Spatial) branches.
    2. Main Backbone: High-Capacity Dense Dilated TCN.
    3. Feedback Module: Processes recycled predictions via a lightweight Dense TCN.
    4. Interaction Head: Augmented Gather (Self+Partner) + Bidirectional GRU.
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Static Input Processing (Backbone)
        # =====================================================================
        # Input features: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (PartnerID) = 18
        self.in_channels = 18

        # Branch B (Context) output size. We use the growth rate for this.
        self.context_channels = Config.GROWTH_RATE

        # Hybrid Stem
        self.input_stem = HybridInputStem(self.in_channels, self.context_channels)

        # Main Backbone: Dense Dilated TCN
        # We maintain a list of blocks. In a DenseNet, input to layer L is
        # concatenation of all previous outputs.
        self.backbone_blocks = nn.ModuleList()
        self.dilations = Config.DILATIONS
        self.growth_rate = Config.GROWTH_RATE

        # Initial input size to the first block: Raw Features + Context Features
        curr_channels = self.in_channels + self.context_channels

        for d in self.dilations:
            block = DenseDilatedBlock(
                in_channels=curr_channels,
                growth_rate=self.growth_rate,
                dilation=d,
                dropout=Config.DROPOUT,
            )
            self.backbone_blocks.append(block)
            # Update channel count for next dense layer
            curr_channels += self.growth_rate

        # Latent Projection: Projects the massive dense concatenation to Latent Dim Z
        self.latent_proj = nn.Conv1d(curr_channels, Config.LATENT_DIM, kernel_size=1)

        # =====================================================================
        # 2. Feedback Processing (Recycling)
        # =====================================================================
        self.feedback_dim = Config.FEEDBACK_DIM

        # Feedback Stem: Handles channel masking and initial projection
        self.feedback_stem = FeedbackStem(self.feedback_dim)

        # Feedback Backbone: Lightweight Dense TCN
        self.fb_blocks = nn.ModuleList()
        self.fb_growth_rate = Config.FEEDBACK_GROWTH_RATE

        curr_fb_channels = self.feedback_dim

        for d in self.dilations:
            block = DenseDilatedBlock(
                in_channels=curr_fb_channels,
                growth_rate=self.fb_growth_rate,
                dilation=d,
                dropout=Config.DROPOUT,
            )
            self.fb_blocks.append(block)
            curr_fb_channels += self.fb_growth_rate

        # Output Projection for Feedback: Projects dense feedback features to E_fb
        self.fb_proj = nn.Conv1d(curr_fb_channels, self.feedback_dim, kernel_size=1)

        # =====================================================================
        # 3. Interaction & Aggregation
        # =====================================================================
        # Input to interaction: Z (Latent) + E_fb (Feedback)
        self.interaction_dim = Config.LATENT_DIM + self.feedback_dim

        # After gathering: Self Vector + Partner Vector
        self.rnn_input_dim = self.interaction_dim * 2

        # Global Aggregation: Bidirectional GRU
        self.gru = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=Config.RNN_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Output Projection: From GRU hidden states to 5 Targets
        # Bidirectional GRU outputs 2 * hidden_dim
        self.head = nn.Linear(Config.RNN_HIDDEN_DIM * 2, 5)

    def forward(self, inputs, partner_indices, prev_preds=None):
        """
        Forward pass of the AHC-HIDN.

        Args:
            inputs (torch.Tensor): (Batch, Length, 18) - Raw one-hot features.
            partner_indices (torch.Tensor): (Batch, Length) - Indices of paired bases (-1 if unpaired).
            prev_preds (torch.Tensor, optional): (Batch, Length, 5) - Predictions from previous recycling step.

        Returns:
            torch.Tensor: (Batch, Length, 5) - Predicted degradation rates.
        """
        B, L, C = inputs.shape

        # ---------------------------------------------------------------------
        # 1. Static Backbone (Compute Z)
        # ---------------------------------------------------------------------
        # Transpose inputs to (B, C, L) for Conv1d operations
        x = inputs.transpose(1, 2)

        # Hybrid Stem: Splits and processes input
        x = self.input_stem(x)  # Output: (B, 18 + Context, L)

        # Dense Backbone: Accumulate features
        features = x
        for block in self.backbone_blocks:
            out = block(features)
            features = torch.cat([features, out], dim=1)

        # Project to Latent Z
        z = self.latent_proj(features)  # (B, 64, L)

        # ---------------------------------------------------------------------
        # 2. Feedback Path (Compute E_fb)
        # ---------------------------------------------------------------------
        if prev_preds is None:
            # First pass: Initialize with zeros
            prev_preds = torch.zeros(
                (B, L, 5), device=inputs.device, dtype=inputs.dtype
            )

        # Feedback Stem: Applies channel masking to unscored targets
        fb_x = self.feedback_stem(prev_preds)  # (B, 32, L)

        # Feedback Dense Backbone
        fb_features = fb_x
        for block in self.fb_blocks:
            out = block(fb_features)
            fb_features = torch.cat([fb_features, out], dim=1)

        # Project to E_fb
        e_fb = self.fb_proj(fb_features)  # (B, 32, L)

        # ---------------------------------------------------------------------
        # 3. Interaction & Aggregation
        # ---------------------------------------------------------------------
        # Concatenate Z and E_fb along channel dimension
        # z: (B, 64, L), e_fb: (B, 32, L) -> combined: (B, 96, L)
        combined = torch.cat([z, e_fb], dim=1)

        # Transpose to (B, L, 96) for gathering and RNN
        combined = combined.transpose(1, 2)

        # Augmented Gather
        # Create mask for paired bases
        mask = partner_indices != -1  # (B, L)

        # Create safe indices (replace -1 with 0 to avoid gather errors)
        safe_indices = partner_indices.clone()
        safe_indices[~mask] = 0

        # Expand indices for gathering: (B, L, 96)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, self.interaction_dim)

        # Gather partner vectors
        partner_vecs = torch.gather(combined, 1, gather_indices)

        # Null-Masking: Zero out vectors where there is no partner
        partner_vecs = partner_vecs * mask.unsqueeze(-1).float()

        # Concatenate Self + Partner
        # (B, L, 96) + (B, L, 96) -> (B, L, 192)
        rnn_input = torch.cat([combined, partner_vecs], dim=2)

        # Global Aggregation (BiGRU)
        # We process the full sequence length (0-107) to maintain boundary anchoring
        rnn_out, _ = self.gru(rnn_input)  # (B, L, 128)

        # Final Projection
        logits = self.head(rnn_out)  # (B, L, 5)

        return logits
