import torch
import torch.nn as nn


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l
    where (.) is dot product, resulting in a scalar gating of x_0.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.w = nn.Parameter(torch.empty(input_dim))
        self.b = nn.Parameter(torch.empty(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization: Near-Zero Standard Deviation to ensure identity-like start
        nn.init.normal_(self.w, mean=0, std=1e-4)
        nn.init.zeros_(self.b)

    def forward(self, x0, xl):
        # x0: [batch, dim], xl: [batch, dim], w: [dim]
        # interaction = (xl * w).sum(dim=1) -> [batch] (Scalar per sample)
        interaction = (xl * self.w).sum(dim=1, keepdim=True)
        return x0 * interaction + self.b + xl


class PreActResNetBlock(nn.Module):
    """
    Full Pre-Activation ResNet Block.
    BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, dim, dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.lin1 = nn.Linear(dim, dim)

        self.bn2 = nn.BatchNorm1d(dim)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim, dim)

        self.reset_parameters()

    def reset_parameters(self):
        # Standard init for Linear layers
        nn.init.kaiming_normal_(self.lin1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.lin2.weight, nonlinearity="relu")
        nn.init.zeros_(self.lin1.bias)
        nn.init.zeros_(self.lin2.bias)

        # Zero-Gamma Initialization for the second BN to force identity mapping at start
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.constant_(self.bn2.weight, 0.0)  # Zero Init
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.lin1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.drop2(out)
        out = self.lin2(out)

        return out + residual


class ZeroInitDeepAsymmetricNet(nn.Module):
    """
    Hybrid architecture with two parallel branches:
    1. Asymmetric Vector-Based DCN (Warm-Start)
    2. Deep Full Pre-Activation ResNet (Zero-Initialized)
    """

    def __init__(
        self, input_dim, hidden_dim, num_blocks, dcn_layers, num_classes, dropout=0.2
    ):
        super().__init__()

        # Input Projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Branch 1: Vector DCN (Warm-Start)
        self.dcn_layers = nn.ModuleList(
            [VectorCrossLayer(hidden_dim) for _ in range(dcn_layers)]
        )

        # Branch 2: Deep Pre-Act ResNet (Zero-Init)
        self.resnet_blocks = nn.ModuleList(
            [PreActResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # Head
        # Concatenation of both branches (hidden_dim * 2) -> Output
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # Project input to hidden dim
        x_proj = self.input_proj(x)

        # Branch 1: DCN
        x_dcn = x_proj
        for layer in self.dcn_layers:
            x_dcn = layer(x_proj, x_dcn)

        # Branch 2: ResNet
        x_res = x_proj
        for block in self.resnet_blocks:
            x_res = block(x_res)

        # Combine
        combined = torch.cat([x_dcn, x_res], dim=1)
        logits = self.head(combined)
        return logits
