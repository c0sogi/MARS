import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseDilatedBlock(nn.Module):
    """
    Single Dilated Residual Block.
    Structure: LayerNorm -> ReLU -> Conv1d (dilated) -> Dropout
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.ReLU(),
            # Permute for Conv1d: (B, L, C) -> (B, C, L) handled in forward or via layers that support it?
            # Standard PyTorch LayerNorm expects (B, L, C), Conv1d expects (B, C, L).
            # We will handle permutation in forward.
        )
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

        # Projection for residual connection if dimensions change
        self.project = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        # x: (B, C, L)

        # LayerNorm expects (B, L, C), so we permute
        residual = x
        x_ln = x.permute(0, 2, 1)  # (B, L, C)
        x_ln = self.net[0](x_ln)  # LayerNorm
        x_ln = self.net[1](x_ln)  # ReLU
        x_ln = x_ln.permute(0, 2, 1)  # Back to (B, C, L)

        out = self.conv(x_ln)
        out = self.dropout(out)

        return out + self.project(residual)


class DenseDilatedTCN(nn.Module):
    """
    Static Backbone with Dense Connections.
    Input to layer i is concatenation of outputs of layers 0...i-1.
    """

    def __init__(
        self, in_channels, growth_rate, kernel_size, dilations, latent_dim, dropout=0.1
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.dilations = dilations

        # Initial projection
        self.stem = nn.Conv1d(in_channels, growth_rate, 1)

        current_channels = growth_rate

        for d in dilations:
            # Input to this block is the accumulation of all previous features
            # But standard DenseNet concats the input to the output.
            # Here we define the block to take 'current_channels' and output 'growth_rate'.
            # Then we concat the output to the input for the next layer.
            block = DenseDilatedBlock(
                in_channels=current_channels,
                out_channels=growth_rate,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            self.blocks.append(block)
            current_channels += growth_rate

        # Final projection to latent dim
        self.final_proj = nn.Conv1d(current_channels, latent_dim, 1)

    def forward(self, x):
        # x: (B, L, C_in) -> permute to (B, C_in, L)
        x = x.permute(0, 2, 1)

        features = [self.stem(x)]

        for block in self.blocks:
            # Concatenate all previous features
            block_input = torch.cat(features, dim=1)
            block_out = block(block_input)
            features.append(block_out)

        # Concatenate everything for final projection
        total_features = torch.cat(features, dim=1)
        out = self.final_proj(total_features)

        # Return to (B, L, C_out)
        return out.permute(0, 2, 1)


class TopologyAwareFeedbackEncoder(nn.Module):
    """
    Lightweight TCN for processing feedback + topology.
    """

    def __init__(
        self, in_channels, hidden_channels, kernel_size, dilations, dropout=0.1
    ):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, hidden_channels, 1)

        self.blocks = nn.ModuleList()
        for d in dilations:
            # Simple residual stack, no dense connections here to keep it light
            self.blocks.append(
                DenseDilatedBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )

    def forward(self, x):
        # x: (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        # (B, L, C)
        return x.permute(0, 2, 1)


class TAFRDNModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # ---------------------------------------------------------------------
        # 1. Static Backbone
        # ---------------------------------------------------------------------
        # Input Channels: Seq(4) + Struct(3) + Loop(7) + PartnerSeq(4) = 18
        input_channels = 18
        self.backbone = DenseDilatedTCN(
            in_channels=input_channels,
            growth_rate=config.backbone_channels,
            kernel_size=config.backbone_kernel_size,
            dilations=config.backbone_dilations,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
        )

        # ---------------------------------------------------------------------
        # 2. Topology-Aware Feedback Encoder
        # ---------------------------------------------------------------------
        # Inputs: Preds(5) + Struct(3) + Loop(7) = 15
        feedback_in_channels = config.num_targets + 3 + 7
        self.feedback_encoder = TopologyAwareFeedbackEncoder(
            in_channels=feedback_in_channels,
            hidden_channels=config.feedback_channels,
            kernel_size=config.feedback_kernel_size,
            dilations=config.feedback_dilations,
            dropout=config.dropout,
        )

        # ---------------------------------------------------------------------
        # 3. Interaction & Aggregation
        # ---------------------------------------------------------------------
        # Node Feature = Z (64) + E_fb (32) = 96
        node_dim = config.latent_dim + config.feedback_channels

        # Interaction: Self + Partner = 96 + 96 = 192
        rnn_input_dim = node_dim * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=config.rnn_hidden_dim,
            num_layers=config.rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.rnn_layers > 1 else 0,
        )

        # Output Head
        self.head = nn.Linear(config.rnn_hidden_dim * 2, config.num_targets)

        # Indices for masking feedback
        # We only want to propagate gradients/signal for scored columns to avoid noise
        # Scored cols: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        self.register_buffer(
            "scored_mask_indices", torch.tensor([0, 1, 3], dtype=torch.long)
        )

    def forward(self, inputs, partner_indices):
        """
        Args:
            inputs: (B, L, 18)
            partner_indices: (B, L) - Indices of paired bases, -1 if unpaired
        """
        batch_size, seq_len, _ = inputs.shape

        # 1. Static Backbone (Run Once)
        z = self.backbone(inputs)  # (B, L, latent_dim)

        # Extract Topology Features for Feedback Module
        # Indices based on process_data:
        # 0-3: Seq, 4-6: Struct, 7-13: Loop, 14-17: PartnerSeq
        # We need Struct (4-6) and Loop (7-13)
        topology_features = inputs[:, :, 4:14]  # (B, L, 10)

        outputs = []

        # Initialize predictions for first pass
        current_preds = torch.zeros(
            batch_size, seq_len, self.config.num_targets, device=inputs.device
        )

        # 2. Recycling Loop
        # steps: 0 -> Pass 1 (Zero Feedback), 1 -> Pass 2 (Feedback from Pass 1)
        for step in range(self.config.recycling_steps):

            # Prepare Feedback Input
            if step == 0:
                # First pass: Zero feedback (or could be learned init, but zero is standard)
                masked_preds = current_preds
            else:
                # Subsequent passes: Use previous predictions
                # Detach to stop gradient backpropagation through time (optional, but usually done in recycling)
                # However, prompt implies "Detach Gradients: R = Y_1.detach()"
                prev_preds = current_preds.detach()

                # Mask unscored columns to 0 to prevent noise injection
                # Create a mask
                mask = torch.zeros_like(prev_preds)
                mask[:, :, self.scored_mask_indices] = 1.0
                masked_preds = prev_preds * mask

            # Concatenate Preds + Topology
            feedback_in = torch.cat(
                [masked_preds, topology_features], dim=-1
            )  # (B, L, 15)

            # Encode Feedback
            e_fb = self.feedback_encoder(feedback_in)  # (B, L, feedback_channels)

            # Construct Node Features: [Z, E_fb]
            node_features = torch.cat([z, e_fb], dim=-1)  # (B, L, 96)

            # Interaction: Gather Partner Features
            # partner_indices is (B, L). We need to gather from dim 1.
            # Create batch indices
            batch_idx = (
                torch.arange(batch_size, device=inputs.device)
                .unsqueeze(1)
                .expand(-1, seq_len)
            )

            # Handle -1 indices (unpaired) by clamping to 0 and then masking result
            safe_partner_idx = partner_indices.clone()
            unpaired_mask = safe_partner_idx == -1
            safe_partner_idx[unpaired_mask] = 0

            # Gather
            # node_features: (B, L, C)
            partner_features = node_features[
                batch_idx, safe_partner_idx, :
            ]  # (B, L, C)

            # Mask unpaired positions to 0
            partner_features[unpaired_mask] = 0.0

            # Concatenate Self + Partner
            rnn_in = torch.cat([node_features, partner_features], dim=-1)  # (B, L, 192)

            # Global Aggregation (RNN)
            rnn_out, _ = self.rnn(rnn_in)  # (B, L, 256)

            # Prediction
            preds = self.head(rnn_out)  # (B, L, 5)

            outputs.append(preds)
            current_preds = preds

        return outputs
