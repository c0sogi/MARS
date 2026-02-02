import torch
import torch.nn as nn


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l^T w) + b + x_l
    """

    def __init__(self, input_dim):
        super(VectorCrossLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        # Initialize weight with Xavier Normal
        nn.init.xavier_normal_(self.weight.unsqueeze(0))

    def forward(self, x0, xl):
        """
        Args:
            x0: Original input features (Batch, Input_Dim)
            xl: Output from previous layer (Batch, Input_Dim)
        """
        # Dot-Product Mixing: (xl * w).sum(dim=1) -> Scalar per sample (Batch, 1)
        # We use element-wise multiplication followed by sum over features
        dot_prod = (xl * self.weight).sum(dim=1, keepdim=True)

        # Apply formula: x_{l+1} = x_0 * dot_prod + b + x_l
        out = x0 * dot_prod + self.bias + xl
        return out


class PreActResNetBlock(nn.Module):
    """
    Pre-Activation Residual Block.
    Structure: BN -> ReLU -> Dropout -> Linear
    """

    def __init__(self, dim, dropout=0.2):
        super(PreActResNetBlock, self).__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear(out)
        # Residual connection
        return x + out


class DeepParallelDCNResNet(nn.Module):
    """
    Deep Parallel Vector-DCN-ResNet (Pre-Activation Variant).
    Combines a Vector DCN branch and a Deep Pre-Act ResNet branch.
    """

    def __init__(self, input_dim, hidden_dim, num_blocks, dropout, num_classes):
        super(DeepParallelDCNResNet, self).__init__()

        # Branch 1: Vector DCN (3 layers)
        # The first layer takes x0 and x0, subsequent layers take x0 and xl
        self.dcn_layers = nn.ModuleList([VectorCrossLayer(input_dim) for _ in range(3)])

        # Branch 2: Pre-Act ResNet
        # Projection from input_dim to hidden_dim
        self.resnet_proj = nn.Linear(input_dim, hidden_dim)
        # Stack of Pre-Activation Blocks
        self.resnet_blocks = nn.ModuleList(
            [PreActResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # Combination Head
        # Concatenates DCN output (input_dim) and ResNet output (hidden_dim)
        concat_dim = input_dim + hidden_dim
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # Branch 1: Vector DCN
        # x0 is the original input x
        x_dcn = x
        for layer in self.dcn_layers:
            x_dcn = layer(x, x_dcn)

        # Branch 2: Pre-Act ResNet
        x_res = self.resnet_proj(x)
        for block in self.resnet_blocks:
            x_res = block(x_res)

        # Combine outputs from both branches
        combined = torch.cat([x_dcn, x_res], dim=1)

        # Final Classification
        logits = self.head(combined)
        return logits
