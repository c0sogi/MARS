import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Standard Residual Block (Linear -> BN -> ReLU -> Dropout).
    Cite Lesson 00018: Standard Residual Blocks outperform GLUs for dense tabular data.
    """

    def __init__(self, input_dim, dropout=0.2, use_batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.bn = nn.BatchNorm1d(input_dim) if use_batch_norm else nn.Identity()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class ResidualMLP(nn.Module):
    """
    Residual MLP (ResNet) for Tabular Data.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,
        use_batch_norm=True,
    ):
        super(ResidualMLP, self).__init__()

        # Input Projection (Stem)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.input_act = nn.ReLU()
        self.input_drop = nn.Dropout(dropout)

        # Residual Blocks
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout, use_batch_norm)
                for _ in range(num_layers)
            ]
        )

        # Output Head
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_act(x)
        x = self.input_drop(x)

        for block in self.blocks:
            x = block(x)

        return self.head(x)


def build_model(input_dim, config=Config):
    """
    Factory function to build the ResidualMLP model using configuration parameters.
    """
    model = ResidualMLP(
        input_dim=input_dim,
        num_classes=config.NUM_CLASSES,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        use_batch_norm=config.USE_BATCH_NORM,
    )
    return model
