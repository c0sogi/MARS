import torch
import torch.nn as nn
from library.config import Config


class InputClamping(nn.Module):
    """
    Fixed, non-trainable layer that clamps inputs to a physical range.
    Prevents outliers in derivative features (e.g., acceleration) from destabilizing the network.
    """

    def __init__(self, min_val: float, max_val: float):
        super(InputClamping, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


class ResBlock(nn.Module):
    """
    Standard Residual Block for tabular data:
    Input -> Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> Add(Input) -> ReLU
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.activation(out)


class KinematicStream(nn.Module):
    """
    Pyramidal Invariant Backbone.
    Hierarchically abstracts kinematic features using an interleaved Project -> ResBlock structure.
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        super(KinematicStream, self).__init__()

        # Input clamping
        self.clamping = InputClamping(Config.CLAMP_MIN, Config.CLAMP_MAX)

        layers = []
        curr_dim = input_dim

        # Build Pyramidal Structure
        for h_dim in hidden_dims:
            # Projection Layer (Downsampling/Abstraction)
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            # Residual Refinement
            layers.append(ResBlock(h_dim, dropout))

            curr_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Final Logit Output
        self.head = nn.Linear(curr_dim, 1)

    def forward(self, x):
        x = self.clamping(x)
        feat = self.backbone(x)
        return self.head(feat)


class VisualStream(nn.Module):
    """
    Shallow MLP for Visual Correction.
    Outputs a correction signal and a reliability gate based on bounding box metrics.
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.1):
        super(VisualStream, self).__init__()

        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Dual Heads
        self.logit_head = nn.Linear(curr_dim, 1)  # Correction Logit (L_vis)
        self.gate_head = nn.Linear(curr_dim, 1)  # Reliability Gate (G_vis)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.backbone(x)

        logit = self.logit_head(feat)
        gate = self.sigmoid(self.gate_head(feat))

        return logit, gate


class APIRVNet(nn.Module):
    """
    Adaptive Pyramidal Invariant Residual-Visual Network.
    Fuses a robust kinematic stream with an adaptive visual correction stream.
    """

    def __init__(self, kin_input_dim: int, vis_input_dim: int):
        super(APIRVNet, self).__init__()

        # Kinematic Stream
        self.kin_stream = KinematicStream(
            input_dim=kin_input_dim,
            hidden_dims=Config.KINEMATIC_HIDDEN_DIMS,
            dropout=Config.DROPOUT_RATE,
        )

        # Visual Stream
        self.vis_stream = VisualStream(
            input_dim=vis_input_dim,
            hidden_dims=Config.VISUAL_HIDDEN_DIMS,
            dropout=Config.DROPOUT_RATE,
        )

        # Learnable Fusion Parameter (Lambda)
        # Initialized to 1.0 to allow balanced initial contribution
        self.fusion_lambda = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_kin, x_vis):
        # 1. Kinematic Forward
        l_kin = self.kin_stream(x_kin)

        # 2. Visual Forward
        l_vis, g_vis = self.vis_stream(x_vis)

        # 3. Adaptive Residual Fusion
        # Logit_final = L_kin + lambda * (G_vis * L_vis)
        # The gate G_vis ensures visual correction is only applied when reliable.
        correction = g_vis * l_vis
        out = l_kin + self.fusion_lambda * correction

        return out
