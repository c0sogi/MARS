import torch
import torch.nn as nn
from library.config import Config


class ResBlock(nn.Module):
    """
    Residual Block for the Pyramidal Backbone.
    Maintains dimensionality to allow for skip connections.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> Dropout -> Add(x) -> ReLU
    """

    def __init__(self, dim, dropout_rate=Config.DROPOUT_RATE):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout_rate),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class LRPNet(nn.Module):
    """
    Linear-Residual Pyramidal Invariant Network (LRP-Net).

    A Triple-Stream Network:
    1. Stream 1: Wide Kinematic Path (Linear Highway) - Captures dominant physical signals directly.
    2. Stream 2: Deep Kinematic Path (Pyramidal Backbone) - Captures complex non-linear dynamics.
    3. Stream 3: Visual Path (Shallow Correction) - Adjusts predictions based on visual cues.

    Fusion: Logit = L_linear + L_deep + lambda * L_vis
    """

    def __init__(self, input_dim):
        super(LRPNet, self).__init__()

        # --- Feature Splitting Logic ---
        # Visual features are the last N columns (4 cols per player * 2 players)
        self.vis_dim = len(Config.VISUAL_COLS) * 2
        self.kin_dim = input_dim - self.vis_dim

        if self.kin_dim <= 0:
            raise ValueError(
                f"Input dimension {input_dim} is too small for calculated visual dimension {self.vis_dim}."
            )

        # --- Input Clamping (Functional, defined in forward) ---
        self.clamp_min = Config.CLAMP_MIN
        self.clamp_max = Config.CLAMP_MAX

        # --- Stream 1: Linear Highway ---
        # Direct connection from kinematic features to output
        self.stream_linear = nn.Linear(self.kin_dim, 1)

        # --- Stream 2: Pyramidal Invariant Backbone ---
        # Structure: Input -> Project -> ResBlock -> Project -> ResBlock -> ... -> Linear
        layers = []
        in_d = self.kin_dim

        for out_d in Config.PYRAMID_DIMS:
            # Projection Layer
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.BatchNorm1d(out_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.DROPOUT_RATE))

            # Residual Block
            layers.append(ResBlock(out_d, Config.DROPOUT_RATE))

            in_d = out_d

        self.stream_deep_backbone = nn.Sequential(*layers)
        self.stream_deep_head = nn.Linear(Config.PYRAMID_DIMS[-1], 1)

        # --- Stream 3: Visual Correction ---
        # Shallow MLP
        self.stream_visual = nn.Sequential(
            nn.Linear(self.vis_dim, Config.VISUAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.VISUAL_HIDDEN_DIM, 1),
        )

        # --- Fusion Parameter ---
        self.visual_lambda = Config.VISUAL_LAMBDA

    def forward(self, x):
        # 1. Input Clamping
        # Prevents extreme outliers from destabilizing gradients.
        # Note: Inputs are standardized, so +/- 50 is a very wide safety guardrail.
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

        # 2. Split Features
        # x_kin: [Batch, kin_dim]
        # x_vis: [Batch, vis_dim]
        x_kin = x[:, : self.kin_dim]
        x_vis = x[:, -self.vis_dim :]

        # 3. Stream 1: Linear Highway
        l_linear = self.stream_linear(x_kin)

        # 4. Stream 2: Deep Pyramidal Backbone
        deep_feat = self.stream_deep_backbone(x_kin)
        l_deep = self.stream_deep_head(deep_feat)

        # 5. Stream 3: Visual Correction
        l_vis = self.stream_visual(x_vis)

        # 6. Fusion
        # Logits summation
        logits = l_linear + l_deep + (self.visual_lambda * l_vis)

        return logits
