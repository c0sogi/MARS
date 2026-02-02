import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Dense Residual Block: Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, size, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.bn1 = nn.BatchNorm1d(size)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(size, size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = out + residual
        return out


class CA_WRN(nn.Module):
    """
    Center-Augmented Wide-Residual Network (CA-WRN).

    Features:
    - Wide Input: Accepts flattened temporal window features.
    - Deep Residual Backbone: Stacks of ResidualBlocks for non-linear processing.
    - Center-Feature Skip Connection: Concatenates raw t=0 features directly to the
      classifier input to prevent signal dilution.
    - Logits Output: Returns raw logits for use with BCEWithLogitsLoss (Focal Loss).
    """

    def __init__(
        self,
        input_dim,
        center_indices,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    ):
        """
        Args:
            input_dim (int): Dimension of the flattened input vector.
            center_indices (list[int]): List of indices in the input vector corresponding
                                        to the center frame (t=0) features to be skipped.
            hidden_size (int): Dimension of the hidden layers.
            num_layers (int): Number of residual blocks.
            dropout (float): Dropout probability.
        """
        super(CA_WRN, self).__init__()

        # Register center indices as a buffer so it moves to device with the model
        # but is not treated as a trainable parameter.
        self.register_buffer(
            "center_indices", torch.tensor(center_indices, dtype=torch.long)
        )

        # Initial projection from input space to hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Stack of Residual Blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, dropout) for _ in range(num_layers)]
        )

        # Final Classification Head
        # Input: Output of Residual Stack (hidden_size) + Raw Center Features (len(center_indices))
        num_center_features = len(center_indices)
        self.head = nn.Linear(hidden_size + num_center_features, 1)

    def forward(self, x):
        # x shape: (batch_size, input_dim)

        # 1. Extract Center Features for Skip Connection
        # These are the raw kinematic values at t=0 (e.g., distance, closing_speed)
        center_features = x[:, self.center_indices]

        # 2. Deep Temporal Processing
        # Project to hidden dim
        out = self.input_proj(x)

        # Pass through residual blocks
        for block in self.blocks:
            out = block(out)

        # 3. Concatenate (Skip Connection)
        # Combine the deep learned representation with the explicit instantaneous physics
        combined = torch.cat([out, center_features], dim=1)

        # 4. Output Logits
        # No activation function here; loss function handles sigmoid
        logits = self.head(combined)

        return logits
