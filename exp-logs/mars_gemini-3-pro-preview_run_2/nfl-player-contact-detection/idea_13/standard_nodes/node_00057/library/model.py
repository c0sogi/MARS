import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    A dense residual block for tabular data.
    Structure: Input -> [Linear->BN->ReLU->Dropout->Linear] + Input -> ReLU
    """

    def __init__(self, dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        # f(x)
        out = self.block(x)
        # Add skip connection: f(x) + x
        out = out + x
        # Final activation
        return self.activation(out)


class WideResNetMLP(nn.Module):
    """
    Entity-Centric Wide-Residual-MLP.
    Accepts a flattened wide feature vector, projects to hidden dim,
    passes through residual blocks, and outputs raw logits.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_blocks=Config.NUM_RES_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super(WideResNetMLP, self).__init__()

        # 1. Input Projection
        # Project high-dimensional wide input to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

        # 2. Residual Backbone
        # Stack of residual blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(hidden_dim, dropout_rate))
        self.backbone = nn.Sequential(*blocks)

        # 3. Output Head
        # Single linear layer to logits (no activation)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.backbone(x)
        logits = self.head(x)
        return logits
