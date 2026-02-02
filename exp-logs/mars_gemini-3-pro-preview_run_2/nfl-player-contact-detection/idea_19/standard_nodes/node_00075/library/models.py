import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Residual Block for Deep MLP.
    Structure: Input -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add -> Output
    """

    def __init__(self, dim: int, dropout_rate: float):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        # Residual connection: x + block(x)
        return x + self.block(x)


class KinematicStream(nn.Module):
    """
    Primary backbone processing tracking data using a Deep Residual MLP.
    Architecture:
        For each hidden dimension:
            Linear (Projection) -> BN -> ReLU -> Dropout
            ResidualBlock
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout_rate: float):
        super(KinematicStream, self).__init__()
        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            # Projection / Downsampling layer to change dimension
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

            # Residual Processing at the new dimension
            layers.append(ResidualBlock(h_dim, dropout_rate))

            current_dim = h_dim

        self.features = nn.Sequential(*layers)
        # Output single logit
        self.head = nn.Linear(current_dim, 1)

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


class VisualStream(nn.Module):
    """
    Comparator MLP processing stereoscopic visual features.
    Designed to learn logical disparity rules (e.g., IoU checks) without
    residual constraints, acting as a non-linear function approximator.
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout_rate: float):
        super(VisualStream, self).__init__()
        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        self.features = nn.Sequential(*layers)
        # Output single logit
        self.head = nn.Linear(current_dim, 1)

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


class SRVNet(nn.Module):
    """
    Stereoscopic Residual-Visual Network.
    Fuses Kinematic and Visual streams via a residual connection.
    Formula: Logit_final = Logit_kinematic + lambda * Logit_visual
    """

    def __init__(
        self,
        input_dim_kin: int = Config.INPUT_DIM_KINEMATIC,
        input_dim_vis: int = Config.INPUT_DIM_VISUAL,
        kinematic_hidden_dims: list = Config.KINEMATIC_HIDDEN_DIMS,
        visual_hidden_dims: list = Config.VISUAL_HIDDEN_DIMS,
        dropout_rate: float = Config.DROPOUT_RATE,
        lambda_visual: float = Config.LAMBDA_VISUAL,
    ):
        super(SRVNet, self).__init__()

        self.kinematic_stream = KinematicStream(
            input_dim_kin, kinematic_hidden_dims, dropout_rate
        )

        self.visual_stream = VisualStream(
            input_dim_vis, visual_hidden_dims, dropout_rate
        )

        self.lambda_visual = lambda_visual

    def forward(self, x_kin, x_vis):
        """
        Args:
            x_kin: Kinematic features [Batch, InputDimKin]
            x_vis: Visual features [Batch, InputDimVis]
        Returns:
            logits: Fused logits [Batch, 1]
        """
        # Get raw logits from both streams
        logit_kin = self.kinematic_stream(x_kin)
        logit_vis = self.visual_stream(x_vis)

        # Residual Fusion
        # The visual stream acts as a correction term to the kinematic baseline
        return logit_kin + self.lambda_visual * logit_vis
