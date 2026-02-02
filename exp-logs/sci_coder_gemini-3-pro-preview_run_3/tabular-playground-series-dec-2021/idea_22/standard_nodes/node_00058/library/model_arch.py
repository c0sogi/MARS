import torch
import torch.nn as nn


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l
    """

    def __init__(self, input_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim))
        self.bias = nn.Parameter(torch.Tensor(input_dim))
        # Initialization
        nn.init.normal_(self.weight, std=0.01)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        # Dot product (xl . w) -> scalar per sample (broadcasted)
        # We sum over the feature dimension (dim=1) to get the dot product
        dot_prod = torch.sum(xl * self.weight, dim=1, keepdim=True)

        # Mix: x0 * scalar + bias + xl
        out = x0 * dot_prod + self.bias + xl
        return out


class ResNetBlock(nn.Module):
    """
    Deep ResNet Block with Dropout.
    Structure: x + (Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout)
    """

    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class HybridModel(nn.Module):
    """
    Deep Parallel Vector-DCN-ResNet Architecture.
    Combines a Vector-based DCN branch for explicit feature interactions
    and a Deep ResNet branch for high-order non-linearities.
    """

    def __init__(
        self, input_dim, num_classes, resnet_depth=4, resnet_width=512, dropout=0.2
    ):
        super().__init__()

        # ==========================
        # Branch 1: Vector DCN
        # ==========================
        # Stack of VectorCrossLayers.
        # We use 4 layers as per the design.
        self.num_cross_layers = 4
        self.cross_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(self.num_cross_layers)]
        )

        # ==========================
        # Branch 2: Deep ResNet
        # ==========================
        # Input projection to ResNet width
        self.resnet_input = nn.Sequential(
            nn.Linear(input_dim, resnet_width), nn.ReLU(), nn.Dropout(dropout)
        )

        # Stack of Residual Blocks
        self.resnet_blocks = nn.ModuleList(
            [ResNetBlock(resnet_width, dropout) for _ in range(resnet_depth)]
        )

        # ==========================
        # Combination Head
        # ==========================
        # Concatenate DCN output (input_dim) and ResNet output (resnet_width)
        concat_dim = input_dim + resnet_width
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # --- Branch 1: DCN ---
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        dcn_out = xl

        # --- Branch 2: ResNet ---
        res = self.resnet_input(x)
        for block in self.resnet_blocks:
            res = block(res)
        resnet_out = res

        # --- Combine ---
        combined = torch.cat([dcn_out, resnet_out], dim=1)
        logits = self.head(combined)

        return logits
