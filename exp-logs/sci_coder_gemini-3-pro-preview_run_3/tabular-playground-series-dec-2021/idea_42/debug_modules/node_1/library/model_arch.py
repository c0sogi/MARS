import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set fixed random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


class LowRankCrossLayer(nn.Module):
    """
    Implements a Low-Rank Factorized Cross Layer for the DCN branch.
    Formula: x_{l+1} = x_0 * (U V^T x_l) + b + x_l

    Attributes:
        U (nn.Parameter): Left factorization matrix (input_dim, rank).
        V (nn.Parameter): Right factorization matrix (input_dim, rank).
        bias (nn.Parameter): Bias vector.
    """

    def __init__(self, input_dim, rank=4):
        super(LowRankCrossLayer, self).__init__()
        self.input_dim = input_dim
        self.rank = rank

        # Rank-4 Factorization: W ~ U * V^T
        self.U = nn.Parameter(torch.Tensor(input_dim, rank))
        self.V = nn.Parameter(torch.Tensor(input_dim, rank))
        self.bias = nn.Parameter(torch.Tensor(input_dim))

        self._init_parameters()

    def _init_parameters(self):
        # Warm-Start Initialization: Near-Zero Standard Deviation (1e-4)
        # Ensures the branch starts as an approximate identity mapping
        nn.init.normal_(self.U, mean=0.0, std=1e-4)
        nn.init.normal_(self.V, mean=0.0, std=1e-4)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        """
        Args:
            x0: Initial input features (Batch, Input_Dim)
            xl: Output from previous layer (Batch, Input_Dim)
        """
        # Compute V^T * xl -> (Batch, Rank)
        vw_x = torch.matmul(xl, self.V)

        # Compute U * (V^T * xl) -> (Batch, Input_Dim)
        w_x = torch.matmul(vw_x, self.U.t())

        # Apply formula: x0 * (W xl) + b + xl
        output = x0 * w_x + self.bias + xl
        return output


class PreActResNetBlock(nn.Module):
    """
    Implements a Full Pre-Activation Residual Block.
    Topology: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add(Input)
    """

    def __init__(self, dim, dropout_rate=0.2):
        super(PreActResNetBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lin1 = nn.Linear(dim, dim)

        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x

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
        out += residual
        return out


class DualViewDCNResNet(nn.Module):
    """
    Dual-View Asymmetric Parallel Low-Rank-DCN-ResNet.

    Processes inputs through two parallel branches:
    1. Asymmetric Low-Rank Factorized DCN (Rank-4, 3 Layers)
    2. Deep Full Pre-Activation ResNet Backbone (512 width, 4 Blocks)

    The outputs are concatenated and passed to a final linear classifier.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        dcn_rank=4,
        dcn_layers=3,
        resnet_blocks=4,
        resnet_dim=512,
        dropout_rate=0.2,
    ):
        super(DualViewDCNResNet, self).__init__()

        # --- Branch 1: Asymmetric Low-Rank Factorized DCN ---
        self.dcn_layers = nn.ModuleList(
            [LowRankCrossLayer(input_dim, rank=dcn_rank) for _ in range(dcn_layers)]
        )

        # --- Branch 2: Deep Full Pre-Activation ResNet Backbone ---
        # Linear projection to align input dimension with ResNet width
        self.resnet_projection = nn.Linear(input_dim, resnet_dim)

        self.resnet_blocks = nn.ModuleList(
            [PreActResNetBlock(resnet_dim, dropout_rate) for _ in range(resnet_blocks)]
        )

        # --- Combination Head ---
        # Concatenates output of DCN (input_dim) and ResNet (resnet_dim)
        combined_dim = input_dim + resnet_dim
        self.final_linear = nn.Linear(combined_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch, Input_Dim)

        # Branch 1: DCN Forward
        # x0 is the original input x, passed to every cross layer
        x_dcn = x
        for layer in self.dcn_layers:
            x_dcn = layer(x, x_dcn)

        # Branch 2: ResNet Forward
        x_res = self.resnet_projection(x)
        for block in self.resnet_blocks:
            x_res = block(x_res)

        # Fusion
        x_combined = torch.cat([x_dcn, x_res], dim=1)

        # Classification
        logits = self.final_linear(x_combined)

        return logits
