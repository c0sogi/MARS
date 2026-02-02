import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedConvBlock(nn.Module):
    """
    Single-Layer Dilated Residual Block.
    Structure: LayerNorm -> ReLU -> Dilated Conv1d -> Dropout.
    Used in both the main backbone and the feedback module.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(in_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Channels, Seq_Len)
        # LayerNorm expects (Batch, Seq_Len, Channels), so we permute
        residual = x
        out = x.permute(0, 2, 1)
        out = self.norm(out)
        out = self.activation(out)
        out = out.permute(0, 2, 1)

        out = self.conv(out)
        out = self.dropout(out)

        # If channel dimensions change, we cannot simply add residual.
        # However, in the Dense Backbone, this block is used to generate *new* features
        # which are then concatenated. The 'residual' connection mentioned in the
        # description likely refers to the internal structure if in_channels == out_channels,
        # or it is implicitly handled by the dense concatenation strategy.
        # For the Feedback TCN (where in=out), we add residual.
        if x.shape[1] == out.shape[1]:
            out = out + residual

        return out


class GraphSmoothedFeedback(nn.Module):
    """
    Graph-Smoothed Feedback Module.

    1. Masks unscored targets in recycled predictions.
    2. Projects to feedback dimension.
    3. Smooths along sequence using a lightweight Dilated TCN.
    4. Diffuses information across base pairs using a structural gather.
    """

    def __init__(self, config: Config):
        super(GraphSmoothedFeedback, self).__init__()
        self.config = config

        # Target columns: ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
        # Scored: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
        # Unscored: deg_pH10 (2), deg_50C (4)
        # Mask: [1, 1, 0, 1, 0]
        self.register_buffer(
            "mask", torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        )

        self.input_proj = nn.Linear(5, config.feedback_dim)

        # Lightweight Dilated TCN (2 blocks)
        self.tcn = nn.Sequential(
            DilatedConvBlock(
                config.feedback_dim,
                config.feedback_dim,
                kernel_size=3,
                dilation=1,
                dropout=config.dropout,
            ),
            DilatedConvBlock(
                config.feedback_dim,
                config.feedback_dim,
                kernel_size=3,
                dilation=2,
                dropout=config.dropout,
            ),
        )

    def forward(self, prev_preds, partner_indices):
        """
        Args:
            prev_preds: (Batch, Seq_Len, 5)
            partner_indices: (Batch, Seq_Len)
        Returns:
            E_fb: (Batch, Seq_Len, Feedback_Dim)
        """
        # 1. Masking
        masked_preds = prev_preds * self.mask.view(1, 1, 5)

        # 2. Projection
        # (B, L, 5) -> (B, L, 32)
        x = self.input_proj(masked_preds)

        # 3. Sequence Smoothing (TCN)
        # Permute to (B, C, L) for Conv1d
        x = x.permute(0, 2, 1)
        h_seq = self.tcn(x)  # (B, 32, L)

        # Permute back to (B, L, 32) for gathering
        h_seq = h_seq.permute(0, 2, 1)

        # 4. Structural Smoothing (Gather)
        batch_size, seq_len, _ = h_seq.shape

        # Handle -1 indices in partner_indices by replacing with 0 temporarily
        # We will mask the result later.
        gather_indices = partner_indices.clone()
        gather_indices[gather_indices == -1] = 0

        # Create batch indices for gather
        # batch_indices: (B, L)
        batch_indices = (
            torch.arange(batch_size, device=h_seq.device)
            .unsqueeze(1)
            .expand(-1, seq_len)
        )

        # Gather features from partners
        # h_seq: (B, L, C)
        # gather_indices: (B, L)
        # Result: (B, L, C)
        h_partner = h_seq[batch_indices, gather_indices]

        # 5. Null-Masking
        # Zero out features where there is no partner (index was -1)
        partner_mask = (partner_indices != -1).unsqueeze(-1).float()  # (B, L, 1)
        h_partner = h_partner * partner_mask

        # 6. Fusion
        e_fb = h_seq + h_partner

        return e_fb


class GSRDN(nn.Module):
    """
    Graph-Smoothed Recurrent Dense Network.

    Structure:
    1. Static Dense Dilated TCN Backbone -> Latent Z
    2. Graph-Smoothed Feedback Module -> Feedback E_fb
    3. Partner-Aware Interaction (Z + E_fb)
    4. Bidirectional GRU
    5. Output Head
    """

    def __init__(self):
        super(GSRDN, self).__init__()
        self.config = Config()

        # ---------------------------------------------------------------------
        # 1. Static Dense Backbone
        # ---------------------------------------------------------------------
        self.growth_rate = 64
        self.backbone_input_proj = nn.Conv1d(
            self.config.input_channels, self.growth_rate, kernel_size=1
        )

        self.dense_blocks = nn.ModuleList()
        current_dim = self.growth_rate

        # Dilations: [1, 2, 4, 8, 16, 32]
        for d in self.config.dilations:
            # Dense Connection: Input to block is concatenation of all previous
            # Here we model it by tracking current_dim which grows
            block = DilatedConvBlock(
                current_dim,
                self.growth_rate,
                kernel_size=self.config.kernel_size,
                dilation=d,
                dropout=self.config.dropout,
            )
            self.dense_blocks.append(block)
            current_dim += self.growth_rate

        # Final projection to Latent Dim (Z)
        self.backbone_out_proj = nn.Conv1d(
            current_dim, self.config.hidden_dim, kernel_size=1
        )

        # ---------------------------------------------------------------------
        # 2. Feedback Module
        # ---------------------------------------------------------------------
        self.feedback_module = GraphSmoothedFeedback(self.config)

        # ---------------------------------------------------------------------
        # 3. Interaction & Aggregation
        # ---------------------------------------------------------------------
        # Input to Interaction: Z (64) + E_fb (32) = 96
        self.interaction_dim = self.config.hidden_dim + self.config.feedback_dim

        # Input to RNN: Self Vector (96) + Partner Vector (96) = 192
        self.rnn_input_dim = self.interaction_dim * 2

        self.rnn = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.rnn_layers,
            batch_first=True,
            bidirectional=self.config.bidirectional,
            dropout=self.config.dropout if self.config.rnn_layers > 1 else 0,
        )

        # ---------------------------------------------------------------------
        # 4. Output Head
        # ---------------------------------------------------------------------
        rnn_out_dim = (
            self.config.hidden_dim * 2
            if self.config.bidirectional
            else self.config.hidden_dim
        )
        self.head = nn.Linear(rnn_out_dim, 5)

    def get_static_features(self, x):
        """
        Runs the heavy static backbone once.
        Args:
            x: (Batch, Seq_Len, Input_Channels)
        Returns:
            z: (Batch, Seq_Len, Hidden_Dim)
        """
        # Permute to (B, C, L)
        x = x.permute(0, 2, 1)

        # Initial projection
        out = self.backbone_input_proj(x)

        # Dense Blocks
        features = [out]
        for block in self.dense_blocks:
            # Concatenate all prior features
            dense_in = torch.cat(features, dim=1)
            new_feat = block(dense_in)
            features.append(new_feat)

        # Final Concatenation
        total_dense = torch.cat(features, dim=1)

        # Projection to Z
        z = self.backbone_out_proj(total_dense)

        # Permute back to (B, L, C)
        z = z.permute(0, 2, 1)
        return z

    def forward(self, x, partner_indices, prev_preds=None, z_cached=None):
        """
        Args:
            x: (Batch, Seq_Len, 18) - Static Inputs
            partner_indices: (Batch, Seq_Len) - Partner Map
            prev_preds: (Batch, Seq_Len, 5) - Recycled Predictions (Optional)
            z_cached: (Batch, Seq_Len, 64) - Precomputed backbone features (Optional)

        Returns:
            preds: (Batch, Seq_Len, 5)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Get Static Features (Z)
        if z_cached is not None:
            z = z_cached
        else:
            z = self.get_static_features(x)

        # 2. Get Feedback Features (E_fb)
        if prev_preds is None:
            prev_preds = torch.zeros(
                (batch_size, seq_len, 5), device=x.device, dtype=x.dtype
            )

        e_fb = self.feedback_module(prev_preds, partner_indices)

        # 3. Interaction (Self + Partner)
        # Concatenate Z and E_fb -> (B, L, 96)
        combined_self = torch.cat([z, e_fb], dim=2)

        # Gather Partner Features
        gather_indices = partner_indices.clone()
        gather_indices[gather_indices == -1] = 0
        batch_indices = (
            torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, seq_len)
        )

        combined_partner = combined_self[batch_indices, gather_indices]

        # Mask unpaired
        partner_mask = (partner_indices != -1).unsqueeze(-1).float()
        combined_partner = combined_partner * partner_mask

        # Concatenate Self and Partner -> (B, L, 192)
        rnn_in = torch.cat([combined_self, combined_partner], dim=2)

        # 4. RNN Aggregation
        rnn_out, _ = self.rnn(rnn_in)

        # 5. Output Head
        preds = self.head(rnn_out)

        return preds
