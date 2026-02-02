import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    DCN_LAYERS,
    DCN_INIT_STD,
    RESNET_BLOCKS,
    HIDDEN_DIM,
    DROPOUT_RATE,
    NUM_CLASSES,
)


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l
    """

    def __init__(self, input_dim):
        super().__init__()
        # Weight vector w: (input_dim, 1) to produce a scalar score per sample
        self.w = nn.Parameter(torch.empty(input_dim, 1))
        # Bias b: (input_dim, )
        self.b = nn.Parameter(torch.empty(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize w with near-zero std to start as identity mapping (Lesson 00066)
        nn.init.normal_(self.w, mean=0, std=DCN_INIT_STD)
        nn.init.zeros_(self.b)

    def forward(self, x0, xl):
        # x0: (batch, dim) - Original input
        # xl: (batch, dim) - Output from previous layer

        # Compute scalar score: (batch, dim) x (dim, 1) -> (batch, 1)
        score = torch.matmul(xl, self.w)

        # Mix: x0 scaled by score, add bias, add residual
        # Broadcasting score across dimension D
        out = x0 * score + self.b + xl
        return out


class PreActResBlock(nn.Module):
    """
    Full Pre-Activation Residual Block.
    Structure: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lin1 = nn.Linear(dim, dim)

        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        # Full pre-activation path (Lesson 00064)
        out = self.bn1(x)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.lin1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.lin2(out)

        return x + out


class WideAsymmetricDCNResNet(nn.Module):
    """
    Wide Asymmetric Parallel Vector-DCN-ResNet.
    Combines a shallow Vector-DCN branch with a wide Pre-Activation ResNet backbone.
    """

    def __init__(self, input_dim, num_classes=NUM_CLASSES):
        super().__init__()

        # Branch 1: Asymmetric Vector-Based DCN (Warm-Start)
        # Keeps dimension equal to input_dim. Asymmetric depth (Lesson 00071).
        self.dcn_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(DCN_LAYERS)]
        )

        # Branch 2: Wide Full Pre-Activation ResNet Backbone
        # Projects input to scaled HIDDEN_DIM (1024) (Lesson 00029)
        self.resnet_proj = nn.Linear(input_dim, HIDDEN_DIM)

        self.resnet_blocks = nn.ModuleList(
            [PreActResBlock(HIDDEN_DIM, DROPOUT_RATE) for _ in range(RESNET_BLOCKS)]
        )

        # Combination Head
        # Concatenates DCN output (input_dim) and ResNet output (HIDDEN_DIM)
        concat_dim = input_dim + HIDDEN_DIM
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # Branch 1: DCN
        # x0 is fixed as the original input x
        x_dcn = x
        for layer in self.dcn_layers:
            x_dcn = layer(x, x_dcn)

        # Branch 2: ResNet
        x_res = self.resnet_proj(x)
        for block in self.resnet_blocks:
            x_res = block(x_res)

        # Concatenate
        combined = torch.cat([x_dcn, x_res], dim=1)

        # Classification
        logits = self.head(combined)
        return logits
