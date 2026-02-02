import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Standard Residual Block for tabular features.
    Structure: Input -> Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> Add -> ReLU
    """

    def __init__(self, in_features, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)


class PIRVNet(nn.Module):
    """
    Pyramidal Invariant Residual-Visual Network (PIRV-Net).

    A Dual-Stream architecture:
    1. Kinematic Stream: Interleaved Pyramidal Backbone (Projection -> ResBlock)
       to hierarchically abstract physical features.
    2. Visual Stream: Shallow MLP to provide visual correction based on helmet metrics.

    Fusion: Additive Residual Fusion (Logits_final = Logits_kin + Logits_vis).
    """

    def __init__(self, input_dim_kin, input_dim_vis):
        super(PIRVNet, self).__init__()

        # ---------------------------------------------------------------------
        # Kinematic Stream (Pyramidal Invariant Backbone)
        # ---------------------------------------------------------------------
        # Structure: Input -> [Project -> ResBlock] x N -> Output
        kin_layers = []
        curr_dim = input_dim_kin

        for next_dim in Config.PYRAMID_LAYERS:
            # Projection Block: Compress/Expand dimensions
            kin_layers.append(nn.Linear(curr_dim, next_dim))
            kin_layers.append(nn.BatchNorm1d(next_dim))
            kin_layers.append(nn.ReLU())

            # Refinement Block: Residual learning at current resolution
            kin_layers.append(ResidualBlock(next_dim))

            curr_dim = next_dim

        self.kin_backbone = nn.Sequential(*kin_layers)

        # Final Kinematic Head (Scalar Logit)
        self.kin_head = nn.Linear(curr_dim, 1)

        # ---------------------------------------------------------------------
        # Visual Stream (Shallow Correction)
        # ---------------------------------------------------------------------
        # Structure: Input -> Linear -> ReLU -> Linear -> Output
        self.vis_backbone = nn.Sequential(
            nn.Linear(input_dim_vis, Config.VISUAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.VISUAL_HIDDEN_DIM, 1),
        )

    def forward(self, x_kin, x_vis):
        """
        Args:
            x_kin (Tensor): Flattened kinematic window features.
            x_vis (Tensor): Flattened visual features.

        Returns:
            Tensor: Raw logits (N, 1).
        """
        # 1. Kinematic Stream
        k = self.kin_backbone(x_kin)
        logits_kin = self.kin_head(k)

        # 2. Visual Stream
        logits_vis = self.vis_backbone(x_vis)

        # 3. Residual Fusion
        # The visual stream acts as a residual correction to the physics-based prediction
        return logits_kin + logits_vis
