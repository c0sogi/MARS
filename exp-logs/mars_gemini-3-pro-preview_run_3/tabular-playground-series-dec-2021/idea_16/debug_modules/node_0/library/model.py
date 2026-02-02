import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import ModelConfig


class LowRankCrossLayer(nn.Module):
    """
    Low-Rank Factorized Cross Layer.
    Implements: x_{l+1} = x_0 * (U (V^T x_l) + b) + x_l
    where W = U V^T is a rank-r approximation of the full weight matrix.
    """

    def __init__(self, in_features, rank):
        super(LowRankCrossLayer, self).__init__()
        self.in_features = in_features
        self.rank = rank

        # U and V are matrices of shape (d, r)
        # W = U V^T
        self.U = nn.Parameter(torch.Tensor(in_features, rank))
        self.V = nn.Parameter(torch.Tensor(in_features, rank))
        self.bias = nn.Parameter(torch.Tensor(in_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization for U and V to ensure stable variance
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        """
        Args:
            x0: Initial input features (Batch, d)
            xl: Output from previous layer (Batch, d)
        """
        # 1. Project input to low-rank space: V^T x_l
        # Operation: (Batch, d) @ (d, r) -> (Batch, r)
        low_rank_proj = torch.mm(xl, self.V)

        # 2. Project back to high-dimensional space: U (...)
        # Operation: (Batch, r) @ (r, d) -> (Batch, d)
        # Note: self.U is (d, r), so we use transpose (r, d)
        interaction = torch.mm(low_rank_proj, self.U.t()) + self.bias

        # 3. Element-wise interaction with x0 and residual connection
        out = x0 * interaction + xl
        return out


class ResNetBlock(nn.Module):
    """
    Wide Residual Block for Tabular Data.
    Structure: Linear -> BN -> ReLU -> Linear -> BN -> Residual -> ReLU
    """

    def __init__(self, hidden_dim, dropout=0.0):
        super(ResNetBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = self.bn2(out)

        # Residual connection before final ReLU
        out += identity
        out = self.relu(out)
        out = self.dropout2(out)

        return out


class ParallelLowRankDCNResNet(nn.Module):
    """
    Hybrid architecture combining:
    1. Parallel Low-Rank DCN Branch (for explicit feature interactions)
    2. Wide ResNet Backbone (for deep representation learning)
    """

    def __init__(self):
        super(ParallelLowRankDCNResNet, self).__init__()

        # Load Hyperparameters
        input_dim = ModelConfig.INPUT_DIM
        hidden_dim = ModelConfig.HIDDEN_DIM
        dcn_rank = ModelConfig.DCN_RANK
        num_classes = ModelConfig.NUM_CLASSES
        dropout = ModelConfig.DROPOUT

        # --- Branch 1: Low-Rank DCN ---
        # Stack of Cross Layers. 3 layers is a standard depth for DCN.
        self.num_cross_layers = 3
        self.cross_layers = nn.ModuleList(
            [
                LowRankCrossLayer(input_dim, dcn_rank)
                for _ in range(self.num_cross_layers)
            ]
        )

        # --- Branch 2: Wide ResNet Backbone ---
        # Input Projection to Hidden Dimension
        self.resnet_input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Deep Residual Blocks
        # Using 2 blocks as the backbone
        self.resnet_blocks = nn.Sequential(
            ResNetBlock(hidden_dim, dropout), ResNetBlock(hidden_dim, dropout)
        )

        # --- Combination Head ---
        # Concatenate outputs: DCN (input_dim) + ResNet (hidden_dim)
        concat_dim = input_dim + hidden_dim
        self.head = nn.Linear(concat_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input tensor (Batch, Input_Dim)
        Returns:
            logits: (Batch, Num_Classes)
        """
        # --- DCN Branch ---
        # x0 is the original input, xl evolves
        x_dcn = x
        for layer in self.cross_layers:
            x_dcn = layer(x, x_dcn)

        # --- ResNet Branch ---
        x_res = self.resnet_input_proj(x)
        x_res = self.resnet_blocks(x_res)

        # --- Combination ---
        # Concatenate features from both branches
        x_concat = torch.cat([x_dcn, x_res], dim=1)

        # Final Classification
        logits = self.head(x_concat)

        return logits
