import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FactorizedCrossLayer(nn.Module):
    """
    Rank-4 Factorized Cross Layer (DCNv2 Low-Rank).
    Formula: x_{l+1} = x_0 * (U * V^T * x_l + b) + x_l
    """

    def __init__(self, input_dim, rank, init_std):
        super().__init__()
        self.rank = rank

        # Factorized Weight Matrices: W = U * V^T
        # U: (input_dim, rank)
        # V: (input_dim, rank)
        self.U = nn.Parameter(torch.Tensor(input_dim, rank))
        self.V = nn.Parameter(torch.Tensor(input_dim, rank))
        self.bias = nn.Parameter(torch.Tensor(input_dim))

        # Warm-Start Initialization (Near-Zero Std)
        # Ensures the branch starts as an approximate identity mapping
        nn.init.normal_(self.U, std=init_std)
        nn.init.normal_(self.V, std=init_std)
        nn.init.zeros_(self.bias)

    def forward(self, x, x0):
        """
        Args:
            x: Output from the previous layer (batch_size, input_dim)
            x0: Original input features (batch_size, input_dim)
        """
        # Compute V^T * x -> (batch, rank)
        # x: (batch, dim), V: (dim, rank)
        v_x = torch.matmul(x, self.V)

        # Compute U * (V^T * x) -> (batch, dim)
        # v_x: (batch, rank), U: (dim, rank)
        w_x = torch.matmul(v_x, self.U.t())

        # Add bias
        w_x = w_x + self.bias

        # Apply Cross Interaction: x0 * (Wx + b) + x
        return x0 * w_x + x


class PreActResNetBlock(nn.Module):
    """
    Full Pre-Activation Residual Block.
    Structure: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(dim, dim)

        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        # First sub-block
        out = self.bn1(x)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.linear1(out)

        # Second sub-block
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.linear2(out)

        # Residual connection
        return out + x


class AsymmetricParallelNet(nn.Module):
    """
    Asymmetric Parallel Factorized-DCN-ResNet (Rank-4).
    Combines a shallow, high-expressivity interaction branch with a deep, robust backbone.
    """

    def __init__(self, input_dim, num_classes=Config.NUM_CLASSES):
        super().__init__()

        # =================================================
        # Branch 1: Asymmetric Factorized Deep & Cross Network
        # =================================================
        # Rank-4 Factorization, Asymmetric Depth (3 layers)
        self.dcn_layers = nn.ModuleList(
            [
                FactorizedCrossLayer(
                    input_dim=input_dim,
                    rank=Config.DCN_RANK,
                    init_std=Config.DCN_INIT_STD,
                )
                for _ in range(Config.DCN_LAYERS)
            ]
        )

        # =================================================
        # Branch 2: Deep Full Pre-Activation ResNet Backbone
        # =================================================
        # Projection to hidden dimension
        self.resnet_projection = nn.Linear(input_dim, Config.HIDDEN_DIM)

        # 4 Residual Blocks
        self.resnet_blocks = nn.Sequential(
            *[
                PreActResNetBlock(
                    dim=Config.HIDDEN_DIM, dropout_rate=Config.RESNET_DROPOUT
                )
                for _ in range(Config.RESNET_BLOCKS)
            ]
        )

        # =================================================
        # Combination Head
        # =================================================
        # Concatenate outputs: DCN (input_dim) + ResNet (hidden_dim)
        final_dim = input_dim + Config.HIDDEN_DIM
        self.classifier = nn.Linear(final_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, input_dim)

        # --- Branch 1: DCN Forward ---
        x_dcn = x
        for layer in self.dcn_layers:
            # Pass current state (x_dcn) and original input (x)
            x_dcn = layer(x_dcn, x)

        # --- Branch 2: ResNet Forward ---
        x_res = self.resnet_projection(x)
        x_res = self.resnet_blocks(x_res)

        # --- Combination & Classification ---
        x_concat = torch.cat([x_dcn, x_res], dim=1)
        logits = self.classifier(x_concat)

        return logits
