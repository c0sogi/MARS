import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock(nn.Module):
    """
    A standard Residual Block for tabular data.
    Structure: Input -> [BN -> ReLU -> Dropout -> Linear] x 2 -> Add Input
    """

    def __init__(self, hidden_dim, dropout=Config.DROPOUT):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetClassifier(nn.Module):
    """
    ResNet-MLP Classifier.
    Cite solution_lesson_node_00014: Avoid unsupervised pretraining, use direct supervised training.
    """

    def __init__(
        self,
        input_dim,
        num_classes=Config.NUM_CLASSES,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
        num_blocks=2,
    ):
        super(ResNetClassifier, self).__init__()

        # Project input dim to hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(hidden_dim, dropout))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Project and pass through ResNet blocks
        h = self.input_proj(x)
        h = self.blocks(h)

        # Classification head
        logits = self.head(h)
        return logits
