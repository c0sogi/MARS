import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Implements the specific residual block structure:
    Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, dim, dropout):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        # Note: The prompt specifies "... -> Add".
        # Standard ResNet applies ReLU after addition, which helps with gradient flow.
        self.final_activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.final_activation(out)


class KinematicBackbone(nn.Module):
    """
    Deep Residual MLP for the Kinematic Stream.
    Structure: Projection -> [Residual Block] -> Projection -> [Residual Block] ...
    """

    def __init__(self):
        super(KinematicBackbone, self).__init__()
        input_dim = Config.get_kinematic_input_dim()
        hidden_dims = Config.KIN_HIDDEN_DIMS
        dropout = Config.KIN_DROPOUT

        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            # Transition/Projection layer to change dimension
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            # Residual Block at the current hidden dimension
            # This adds depth and non-linearity while preserving the feature space
            layers.append(ResidualBlock(h_dim, dropout))

            curr_dim = h_dim

        self.features = nn.Sequential(*layers)
        # Output a single scalar logit
        self.head = nn.Linear(curr_dim, 1)

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


class VisualBackbone(nn.Module):
    """
    Shallow MLP for the Visual Stream (Correction Branch).
    """

    def __init__(self):
        super(VisualBackbone, self).__init__()
        input_dim = Config.get_visual_input_dim()
        hidden_dims = Config.VIS_HIDDEN_DIMS
        dropout = Config.VIS_DROPOUT

        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        self.features = nn.Sequential(*layers)
        # Output a single scalar logit
        self.head = nn.Linear(curr_dim, 1)

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


class RVCNet(nn.Module):
    """
    Residual-Visual Corrected Kinematic Network (RVC-Net).
    Fuses a robust kinematic stream with a visual correction stream via residual connection.
    """

    def __init__(self):
        super(RVCNet, self).__init__()

        # Primary Backbone
        self.kinematic = KinematicBackbone()

        # Correction Branch
        self.visual = VisualBackbone()

        # Learnable residual weight
        # Initialized to the value specified in Config
        self.res_lambda = nn.Parameter(torch.tensor(Config.RESIDUAL_LAMBDA_INIT))

    def forward(self, k_input, v_input):
        """
        Args:
            k_input (torch.Tensor): Flattened kinematic features.
            v_input (torch.Tensor): Flattened visual features.

        Returns:
            torch.Tensor: Final logits combining both streams.
        """
        # Get logits from both streams
        k_logit = self.kinematic(k_input)
        v_logit = self.visual(v_input)

        # Residual Fusion: Base + lambda * Correction
        # This structure allows the visual stream to nudge the robust kinematic prediction
        final_logit = k_logit + self.res_lambda * v_logit

        return final_logit
