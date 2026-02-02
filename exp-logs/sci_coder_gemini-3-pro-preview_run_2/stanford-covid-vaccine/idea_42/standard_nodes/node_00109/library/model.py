import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseLayer(nn.Module):
    """
    Single Dense Layer: Norm -> ReLU -> Conv1d -> Dropout.
    Designed for DenseNet connectivity where input channels grow.
    """

    def __init__(
        self, in_channels, growth_rate, kernel_size=3, dilation=1, dropout=0.1
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.ReLU()
        # Padding ensures output length matches input length
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, L)
        # LayerNorm expects (B, L, C)
        out = x.permute(0, 2, 1)
        out = self.norm(out)
        out = self.act(out)
        out = out.permute(0, 2, 1)

        out = self.conv(out)
        out = self.dropout(out)
        return out


class StaticBackbone(nn.Module):
    """
    Static Dense Dilated TCN Backbone.
    Processes the input sequence once to produce a dense feature representation.
    """

    def __init__(
        self, in_channels, growth_rate, layers, kernel_size, dropout, latent_dim
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        # Initial projection to establish the feature space
        self.init_conv = nn.Conv1d(in_channels, growth_rate, kernel_size=1)
        current_channels = growth_rate

        # Exponentially increasing dilation rates
        dilations = [2**i for i in range(layers)]

        for d in dilations:
            self.layers.append(
                DenseLayer(
                    current_channels,
                    growth_rate,
                    kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
            # In DenseNet, the channel count increases by growth_rate at each step
            current_channels += growth_rate

        # Final projection to Latent Dimension (Z)
        self.final_proj = nn.Conv1d(current_channels, latent_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, 18, L)
        x = self.init_conv(x)
        features = [x]

        for layer in self.layers:
            # Dense Connection: Concatenate all prior feature maps
            inp = torch.cat(features, dim=1)
            out = layer(inp)
            features.append(out)

        # Concatenate everything for the final projection
        h_dense = torch.cat(features, dim=1)
        z = self.final_proj(h_dense)
        return z


class FeedbackModule(nn.Module):
    """
    Lightweight Dense Feedback Module.
    Processes recycled predictions to extract error gradients/features.
    """

    def __init__(self, in_channels, hidden_dim, growth_rate, layers, embed_dim):
        super().__init__()
        # Project predictions to hidden dimension
        self.proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)

        self.dense_layers = nn.ModuleList()
        current_channels = hidden_dim

        for _ in range(layers):
            # Standard dense layers with dilation 1 for local refinement
            self.dense_layers.append(
                DenseLayer(
                    current_channels,
                    growth_rate,
                    kernel_size=3,
                    dilation=1,
                    dropout=0.1,
                )
            )
            current_channels += growth_rate

        # Project to Feedback Embedding Dimension (E_fb)
        self.out_proj = nn.Conv1d(current_channels, embed_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, 5, L)
        x = self.proj(x)
        features = [x]

        for layer in self.dense_layers:
            inp = torch.cat(features, dim=1)
            out = layer(inp)
            features.append(out)

        h_dense = torch.cat(features, dim=1)
        e_fb = self.out_proj(h_dense)
        return e_fb


class DR_RHN(nn.Module):
    """
    Dense-Refined Recurrent Hybrid Network.
    Combines a static dense backbone with a lightweight dense feedback loop.
    """

    def __init__(self):
        super().__init__()

        self.num_targets = Config.NUM_TARGETS
        self.scored_indices = Config.SCORED_TARGET_INDICES

        # 1. Static Backbone
        self.backbone = StaticBackbone(
            in_channels=Config.NUM_NODE_FEATURES,
            growth_rate=Config.BACKBONE_GROWTH_RATE,
            layers=Config.BACKBONE_LAYERS,
            kernel_size=Config.BACKBONE_KERNEL_SIZE,
            dropout=Config.BACKBONE_DROPOUT,
            latent_dim=Config.LATENT_DIM,
        )

        # 2. Feedback Module
        self.feedback = FeedbackModule(
            in_channels=Config.FEEDBACK_IN_CHANNELS,
            hidden_dim=Config.FEEDBACK_HIDDEN_DIM,
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            layers=Config.FEEDBACK_LAYERS,
            embed_dim=Config.FEEDBACK_EMBED_DIM,
        )

        # 3. Aggregation & Head
        # Input to RNN is concatenation of:
        # Self: [Z (64), E_fb (32)]
        # Partner: [Z (64), E_fb (32)]
        # Total: 192
        rnn_input_dim = (Config.LATENT_DIM + Config.FEEDBACK_EMBED_DIM) * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
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

    def forward(self, x, partner_idx, pairing_mask):
        """
        Args:
            x: Input features (B, L, 18)
            partner_idx: Indices of paired bases (B, L)
            pairing_mask: 1 if paired, 0 if unpaired (B, L)
        """
        # Permute x for Conv1d: (B, L, 18) -> (B, 18, L)
        x = x.permute(0, 2, 1)
        B, C, L = x.shape

        # Step 1: Compute Static Backbone Features
        z = self.backbone(x)  # (B, 64, L)

        # Step 2: Recycling Loop
        # Initialize predictions with zeros
        y_current = torch.zeros(B, L, self.num_targets, device=x.device)
        outputs = []

        # We run 2 passes:
        # Pass 1: Feedback is 0 -> Generates y_1
        # Pass 2: Feedback is y_1 (masked) -> Generates y_2
        for pass_idx in range(2):

            # Prepare Feedback Input
            if pass_idx == 0:
                fb_in = y_current.permute(0, 2, 1)  # (B, 5, L) - all zeros
            else:
                # Detach gradients from previous pass
                y_detached = y_current.detach()

                # Mask unscored columns to prevent noise injection
                # Create a mask for columns NOT in scored_indices
                # Scored: [0, 1, 3], Unscored: [2, 4]
                mask_indices = [
                    i for i in range(self.num_targets) if i not in self.scored_indices
                ]
                y_detached[:, :, mask_indices] = 0.0

                fb_in = y_detached.permute(0, 2, 1)

            # Compute Feedback Embeddings
            e_fb = self.feedback(fb_in)  # (B, 32, L)

            # Interaction: Concatenate Static (Z) and Dynamic (E_fb)
            # z: (B, 64, L), e_fb: (B, 32, L) -> self_feat: (B, 96, L)
            self_feat = torch.cat([z, e_fb], dim=1)

            # Gather Partner Features
            # partner_idx: (B, L). We need to expand it to gather along channel dim.
            # self_feat: (B, C, L)
            idx_expanded = partner_idx.unsqueeze(1).expand(-1, self_feat.size(1), -1)
            partner_feat = torch.gather(self_feat, 2, idx_expanded)

            # Null-Masking: Zero out partner features if base is unpaired
            # pairing_mask: (B, L) -> (B, 1, L) -> Broadcast to (B, C, L)
            p_mask_expanded = pairing_mask.unsqueeze(1).expand_as(partner_feat)
            partner_feat = partner_feat * p_mask_expanded

            # Late Fusion: Concatenate Self and Partner vectors
            # (B, 96, L) + (B, 96, L) -> (B, 192, L)
            combined = torch.cat([self_feat, partner_feat], dim=1)

            # Global Aggregation (RNN)
            # RNN expects (B, L, C)
            rnn_in = combined.permute(0, 2, 1)
            rnn_out, _ = self.rnn(rnn_in)

            # Prediction Head
            logits = self.head(rnn_out)  # (B, L, 5)

            y_current = logits
            outputs.append(y_current)

        # Return both intermediate and final predictions for loss calculation
        return outputs[0], outputs[1]
