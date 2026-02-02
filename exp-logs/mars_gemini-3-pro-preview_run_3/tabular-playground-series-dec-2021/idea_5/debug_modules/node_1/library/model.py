import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) block.
    Computes the element-wise product of a linear transformation and a sigmoid-activated gate.
    Formula: x = (x @ W_val + b_val) * Sigmoid(x @ W_gate + b_gate)
    """

    def __init__(self, input_dim, output_dim, dropout=0.0, use_batch_norm=True):
        super(GatedLinearUnit, self).__init__()

        self.fc_val = nn.Linear(input_dim, output_dim)
        self.fc_gate = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_batch_norm = use_batch_norm

        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        # Value branch
        val = self.fc_val(x)

        # Gate branch
        gate = torch.sigmoid(self.fc_gate(x))

        # Gated output
        out = val * gate

        # Normalization and Regularization
        if self.use_batch_norm:
            out = self.bn(out)

        out = self.dropout(out)
        return out


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) for Tabular Data.
    Stacks GatedLinearUnit blocks with residual connections.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=512,
        num_layers=3,
        dropout=0.2,
        use_batch_norm=True,
    ):
        super(GatedResidualNetwork, self).__init__()

        # Input Projection (Stem)
        # Projects input features to the hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.input_activation = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout)

        # Residual Blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GatedLinearUnit(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
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
            # Residual connection: x = x + block(x)
            residual = x
            out = layer(x)
            x = residual + out

        # 3. Classification Head
        logits = self.output_head(x)
        return logits


def build_model(input_dim, config=Config):
    """
    Factory function to build the GRN model using configuration parameters.

    Args:
        input_dim (int): Dimensionality of the input features (after engineering).
        config (Config): Configuration class containing hyperparameters.

    Returns:
        GatedResidualNetwork: The instantiated model.
    """
    model = GatedResidualNetwork(
        input_dim=input_dim,
        num_classes=config.NUM_CLASSES,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        use_batch_norm=config.USE_BATCH_NORM,
    )
    return model
