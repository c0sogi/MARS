import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l

    Initialization:
    - Weights (w) are initialized with a near-zero standard deviation (1e-4)
      to ensure the layer starts as an approximate identity mapping (Warm-Start).
    """

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # Weight vector w: [input_dim]
        self.w = nn.Parameter(torch.empty(input_dim))
        # Bias vector b: [input_dim]
        self.b = nn.Parameter(torch.zeros(input_dim))

        # Initialization: Near-Zero to start as identity mapping
        nn.init.normal_(self.w, mean=0, std=1e-4)

    def forward(self, x0, xl):
        """
        Args:
            x0: Initial input features [batch_size, input_dim]
            xl: Output from previous layer [batch_size, input_dim]
        Returns:
            x_{l+1}: [batch_size, input_dim]
        """
        # Dot product (batch-wise): (xl * w).sum(dim=1) -> [batch]
        # We broadcast this scalar to [batch, 1] for multiplication
        dot_prod = (xl * self.w).sum(dim=1, keepdim=True)  # [batch, 1]

        # Interaction: x0 * scalar
        interaction = x0 * dot_prod  # [batch, input_dim]

        # Add bias and residual connection
        return interaction + self.b + xl


class PreActResBlock(nn.Module):
    """
    Full Pre-Activation ResNet Block.
    Topology: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lin1 = nn.Linear(dim, dim)

        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        # First sub-block
        out = self.bn1(x)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.lin1(out)

        # Second sub-block
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.lin2(out)

        # Residual connection
        return x + out


class DeepParallelVectorDCNResNet(nn.Module):
    """
    Deep Parallel Vector-DCN-ResNet Architecture.

    Combines two parallel branches:
    1. Vector-based Deep & Cross Network (DCN) for explicit feature interactions.
    2. Deep Pre-Activation ResNet for high-order non-linear representation learning.

    The outputs of both branches are concatenated and passed to a final classification head.
    """

    def __init__(
        self,
        input_dim,
        num_classes=7,
        hidden_dim=512,
        num_cross_layers=3,
        num_res_blocks=4,
        dropout_rate=0.2,
    ):
        super().__init__()

        # Branch 1: Vector DCN
        # Stack of VectorCrossLayers
        self.cross_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(num_cross_layers)]
        )

        # Branch 2: Deep ResNet
        # Projection from input_dim to hidden_dim
        self.res_proj = nn.Linear(input_dim, hidden_dim)
        # Stack of PreActResBlocks
        self.res_blocks = nn.ModuleList(
            [PreActResBlock(hidden_dim, dropout_rate) for _ in range(num_res_blocks)]
        )

        # Combination Head
        # Concatenate: input_dim (from DCN) + hidden_dim (from ResNet)
        concat_dim = input_dim + hidden_dim
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # x: [batch, input_dim]

        # Branch 1: DCN Forward Pass
        # x0 is the original input x
        x_dcn = x
        for layer in self.cross_layers:
            x_dcn = layer(x, x_dcn)  # Pass x0 (x) and xl (x_dcn)

        # Branch 2: ResNet Forward Pass
        x_res = self.res_proj(x)
        for block in self.res_blocks:
            x_res = block(x_res)

        # Combine
        x_concat = torch.cat([x_dcn, x_res], dim=1)

        # Classification
        logits = self.head(x_concat)

        return logits
