import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class LowRankCrossLayer(nn.Module):
    """
    Implements the Low-Rank Factorized Cross Layer:
    x_{l+1} = x_0 * (U * V^T * x_l) + b + x_l

    Uses Rank-4 decomposition to balance expressivity and parameter efficiency.
    """

    def __init__(self, input_dim, rank=4):
        super(LowRankCrossLayer, self).__init__()
        self.input_dim = input_dim
        self.rank = rank

        # Weight decomposition W = U * V^T
        # U: (input_dim, rank)
        # V: (input_dim, rank)
        self.U = nn.Parameter(torch.empty(input_dim, rank))
        self.V = nn.Parameter(torch.empty(input_dim, rank))
        self.bias = nn.Parameter(torch.zeros(input_dim))

        self._init_parameters()

    def _init_parameters(self):
        # Warm-Start Initialization: Near-zero standard deviation
        # Ensures the branch starts as an approximate identity mapping (Lesson 00066)
        nn.init.normal_(self.U, mean=0.0, std=1e-4)
        nn.init.normal_(self.V, mean=0.0, std=1e-4)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        """
        x0: Initial input features (Batch, Dim)
        xl: Output from previous layer (Batch, Dim)
        """
        # Compute V^T * xl
        # (Batch, Dim) @ (Dim, Rank) -> (Batch, Rank)
        v_x = torch.matmul(xl, self.V)

        # Compute U * (V^T * xl)
        # (Batch, Rank) @ (Rank, Dim) -> (Batch, Dim)
        # Note: self.U.t() is (Rank, Dim)
        w_x = torch.matmul(v_x, self.U.t())

        # Compute interaction: x0 * (W * xl) + b
        # Element-wise multiplication with x0
        interaction = x0 * w_x + self.bias

        # Add residual connection
        return interaction + xl


class PreActResBlock(nn.Module):
    """
    Full Pre-Activation Residual Block:
    Input -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add(Input)

    Optimized for gradient flow (Lesson 00064).
    """

    def __init__(self, dim, dropout_rate):
        super(PreActResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.drop1 = nn.Dropout(dropout_rate)

        self.bn2 = nn.BatchNorm1d(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # First sub-block
        out = self.bn1(x)
        out = F.relu(out)
        out = self.drop1(out)
        out = self.lin1(out)

        # Second sub-block
        out = self.bn2(out)
        out = F.relu(out)
        out = self.drop2(out)
        out = self.lin2(out)

        # Residual connection
        return x + out


class ParallelDCNResNet(nn.Module):
    """
    Asymmetric Parallel Low-Rank-DCN-ResNet (Rank-4)

    Structure:
    1. Branch 1: Low-Rank Factorized DCN (3 layers, Rank 4) - Captures explicit interactions.
    2. Branch 2: Pre-Activation ResNet Backbone (4 blocks, 512 dim) - Captures deep implicit patterns.
    3. Head: Concatenation -> Linear Classification.
    """

    def __init__(self, input_dim, num_classes):
        super(ParallelDCNResNet, self).__init__()

        # Configuration from Config
        dcn_rank = Config.DCN_RANK
        dcn_layers = Config.DCN_LAYERS

        resnet_blocks = Config.RESNET_BLOCKS
        hidden_dim = Config.HIDDEN_DIM
        dropout_rate = Config.DROPOUT_RATE

        # Branch 1: Low-Rank DCN
        # Operates on the raw input dimension to avoid high-order noise (Lesson 00071)
        self.dcn_layers = nn.ModuleList(
            [LowRankCrossLayer(input_dim, rank=dcn_rank) for _ in range(dcn_layers)]
        )

        # Branch 2: ResNet Backbone
        # Projects input to hidden dimension first
        self.resnet_projection = nn.Linear(input_dim, hidden_dim)
        self.resnet_blocks = nn.ModuleList(
            [PreActResBlock(hidden_dim, dropout_rate) for _ in range(resnet_blocks)]
        )

        # Combination Head
        # Concatenates output of DCN (input_dim) and ResNet (hidden_dim)
        concat_dim = input_dim + hidden_dim
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # Branch 1: DCN Forward
        # x0 is the original input x
        xl = x
        for layer in self.dcn_layers:
            xl = layer(x, xl)
        dcn_out = xl

        # Branch 2: ResNet Forward
        res_out = self.resnet_projection(x)
        for block in self.resnet_blocks:
            res_out = block(res_out)

        # Concatenate branches
        combined = torch.cat([dcn_out, res_out], dim=1)

        # Classification
        logits = self.head(combined)

        return logits
