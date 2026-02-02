import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Applies an affine transformation to the input features conditioned on an external vector.

    Formula: h_out = gamma(condition) * h_in + beta(condition)
    """

    def __init__(self, num_features, condition_dim=Config.FILM_DIM):
        super(FiLMLayer, self).__init__()

        # Generators for scale (gamma) and shift (beta)
        self.scale_gen = nn.Linear(condition_dim, num_features)
        self.shift_gen = nn.Linear(condition_dim, num_features)

        self._init_weights()

    def _init_weights(self):
        # Initialize scale to Identity (1.0) and shift to Zero (0.0)
        # This ensures the layer starts as an identity mapping, aiding convergence.
        nn.init.constant_(self.scale_gen.weight, 0)
        nn.init.constant_(self.scale_gen.bias, 1)

        nn.init.constant_(self.shift_gen.weight, 0)
        nn.init.constant_(self.shift_gen.bias, 0)

    def forward(self, x, condition):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Num_Features)
            condition (torch.Tensor): Conditioning vector (Batch, Condition_Dim)
        """
        gamma = self.scale_gen(condition)
        beta = self.shift_gen(condition)

        # Apply affine transformation
        return x * gamma + beta


class ResidualBlock(nn.Module):
    """
    Residual Block with FiLM conditioning.
    Structure: Input -> [Linear->BN->ReLU->Dropout] -> [Linear->BN] -> FiLM -> Add -> ReLU
    """

    def __init__(
        self,
        hidden_dim,
        condition_dim=Config.FILM_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super(ResidualBlock, self).__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.film = FiLMLayer(hidden_dim, condition_dim)

    def forward(self, x, condition):
        residual = x

        # First dense part
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)

        # Second dense part
        out = self.fc2(out)
        out = self.bn2(out)

        # Feature-wise Linear Modulation
        out = self.film(out, condition)

        # Residual Connection
        out = out + residual

        # Final Activation
        out = self.act(out)

        return out


class CKResNet(nn.Module):
    """
    Conditioned Kinematic Residual Network (CK-ResNet).

    A wide-input MLP backbone enhanced with residual connections and FiLM layers
    to handle distinct physics for Player-Player vs Player-Ground interactions.
    """

    def __init__(
        self,
        input_dim,
        center_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_RES_BLOCKS,
        condition_dim=Config.FILM_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        """
        Args:
            input_dim (int): Dimension of the flattened wide feature vector.
            center_dim (int): Dimension of the features at t=0 (for skip connection).
            hidden_dim (int): Internal hidden dimension.
            num_blocks (int): Number of residual blocks.
            condition_dim (int): Dimension of the conditioning vector (is_ground).
            dropout_rate (float): Dropout probability.
        """
        super(CKResNet, self).__init__()

        # Initial projection of wide input
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

        # Stack of Residual Blocks
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, condition_dim, dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        # Output Head
        # Concatenates the residual output with the raw center frame features
        self.output_head = nn.Linear(hidden_dim + center_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_wide, x_center, condition):
        """
        Args:
            x_wide (torch.Tensor): Flattened window features (Batch, Input_Dim).
            x_center (torch.Tensor): Features at t=0 (Batch, Center_Dim).
            condition (torch.Tensor): Conditioning vector (Batch, Condition_Dim).

        Returns:
            torch.Tensor: Logits (Batch, 1).
        """
        # Project wide input
        out = self.input_proj(x_wide)

        # Pass through residual blocks with conditioning
        for block in self.blocks:
            out = block(out, condition)

        # Center-Frame Skip Connection
        # Concatenate the processed deep features with the raw instantaneous state
        combined = torch.cat([out, x_center], dim=1)

        # Final prediction
        logits = self.output_head(combined)

        return logits
