import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Standard Residual Block for Tabular Data.
    Structure: Linear -> BN -> ReLU -> Dropout
    Cite solution_lesson_node_00018: Prefer standard Residual Blocks over GLUs.
    """

    def __init__(self, input_dim, dropout=0.0, use_batch_norm=True):
        super(ResidualBlock, self).__init__()

        self.linear = nn.Linear(input_dim, input_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.use_batch_norm = use_batch_norm

        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        if self.use_batch_norm:
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
        self.input_activation = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout)

        # Residual Blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                ResidualBlock(
                    input_dim=hidden_dim,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                )
            )

        # Output Head
        self.output_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # 1. Input Projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_activation(x)
        x = self.input_dropout(x)

        # 2. Residual Blocks
        for layer in self.layers:
            x = layer(x)

        # 3. Classification Head
        logits = self.output_head(x)
        return logits


def build_model(input_dim, config=Config):
    """
    Factory function to build the ResNet model using configuration parameters.
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
