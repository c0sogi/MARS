import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import config


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer for DCN branch.
    Formula: x_{l+1} = x_0 * (x_l^T w) + b + x_l
    Uses Dot-Product Mixing and Near-Zero Initialization.
    """

    def __init__(self, input_dim: int, init_std: float = 1e-4):
        super().__init__()
        # Weight vector w: (input_dim, 1) for dot product
        self.w = nn.Parameter(torch.empty(input_dim, 1))
        # Bias vector b: (input_dim,)
        self.b = nn.Parameter(torch.empty(input_dim))

        # Initialization
        # Near-zero initialization for w to start as identity mapping
        # This stabilizes the deep backbone's early gradients (Lesson 00066)
        nn.init.normal_(self.w, mean=0, std=init_std)
        nn.init.zeros_(self.b)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: Original input features (Batch, Input_Dim)
            xl: Output from previous layer (Batch, Input_Dim)
        """
        # Term 1: Dot product (xl . w) -> Scalar per sample
        # (B, D) @ (D, 1) -> (B, 1)
        dot_prod = torch.matmul(xl, self.w)

        # Term 2: Element-wise multiplication with x0 (Broadcasting)
        # (B, D) * (B, 1) -> (B, D)
        mixed = x0 * dot_prod

        # Sum: mixed + bias + residual
        out = mixed + self.b + xl
        return out


class PreActResBlock(nn.Module):
    """
    Full Pre-Activation ResNet Block.
    Structure: Input -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add(Input)
    Corrects capacity flaw of single-layer blocks (Lesson 00064)
    """

    def __init__(self, hidden_dim: int, dropout_rate: float):
        super().__init__()

        # First sub-block
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)

        # Second sub-block
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input for residual connection
        shortcut = x

        # Block 1
        out = self.bn1(x)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.linear1(out)

        # Block 2
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.linear2(out)

        # Residual Add
        return out + shortcut


class ParallelDCNResNet(nn.Module):
    """
    Deep Parallel Vector-DCN-ResNet Architecture.
    Branch 1: Deep & Cross Network (Vector-based)
    Branch 2: Deep Full Pre-Activation ResNet
    """

    def __init__(self):
        super().__init__()

        # Load hyperparameters from config
        input_dim = config.model.input_dim
        hidden_dim = config.model.hidden_dim
        num_blocks = config.model.num_resnet_blocks
        num_cross_layers = config.model.num_cross_layers
        dropout_rate = config.model.dropout_rate
        num_classes = config.model.num_classes
        init_std = config.model.cross_layer_init_std

        # --- Branch 1: Vector DCN ---
        # Stack of Cross Layers
        self.cross_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim, init_std) for _ in range(num_cross_layers)]
        )

        # --- Branch 2: ResNet ---
        # Projection from Input Dim to Hidden Dim
        self.resnet_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of Pre-Activation Residual Blocks
        self.resnet_blocks = nn.Sequential(
            *[PreActResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        # --- Combination Head ---
        # Concatenation of DCN output (input_dim) and ResNet output (hidden_dim)
        concat_dim = input_dim + hidden_dim
        self.final_linear = nn.Linear(concat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Input_Dim)

        # --- Branch 1: DCN Forward ---
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        dcn_out = xl

        # --- Branch 2: ResNet Forward ---
        # Project
        res_x = self.resnet_proj(x)
        # Backbone
        res_out = self.resnet_blocks(res_x)

        # --- Combination ---
        # Concatenate outputs from both branches
        combined = torch.cat([dcn_out, res_out], dim=1)

        # Final Classification
        logits = self.final_linear(combined)

        return logits
