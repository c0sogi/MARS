import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    A Dense Residual Block designed for the EC-PIRN architecture.
    Structure: Input -> Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> Add(Input) -> ReLU
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class ECPIRN(nn.Module):
    """
    Entity-Centric Physics-Informed Residual Network (EC-PIRN).

    A Deep Residual Multi-Layer Perceptron that processes flattened wide-format
    temporal features. It uses residual connections to enable deep non-linear
    feature extraction while preserving the gradient flow, correcting the
    capacity limitations of shallow baselines.
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_RESIDUAL_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super(ECPIRN, self).__init__()

        # Initial projection from input space to hidden space
        self.initial_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Stack of Residual Blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        # Final Output Layer (Logits)
        # No activation function here to ensure numerical stability with Focal Loss (BCEWithLogits)
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Project input
        x = self.initial_projection(x)

        # Pass through residual backbone
        for block in self.blocks:
            x = block(x)

        # Generate logits
        x = self.output_head(x)
        return x
