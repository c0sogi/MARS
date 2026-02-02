import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WideBranch(nn.Module):
    """
    Wide Linear Branch: A single Linear layer projecting raw input to class logits.
    Explicitly captures low-level linear dependencies (e.g., identity mappings).
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l

    This layer explicitly models feature interactions using a learnable vector 'w'.
    """

    def __init__(self, input_dim):
        super().__init__()
        # Parameter w: vector of size input_dim (D, 1)
        self.w = nn.Parameter(torch.empty(input_dim, 1))
        # Parameter b: vector of size input_dim (D)
        self.b = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self):
        # Warm-Start Initialization: Near-Zero Standard Deviation N(0, 1e-4)
        # Ensures the branch starts as an approximate identity mapping to prevent
        # multiplicative noise from destabilizing the backbone early in training.
        nn.init.normal_(self.w, mean=0.0, std=1e-4)
        nn.init.zeros_(self.b)

    def forward(self, x_l, x_0):
        """
        Args:
            x_l: Input from previous layer (Batch, Dim)
            x_0: Original input (Batch, Dim)
        """
        # Calculate scalar mixing score for each sample: x_l dot w
        # (Batch, Dim) @ (Dim, 1) -> (Batch, 1)
        mixing_score = torch.matmul(x_l, self.w)

        # Apply mixing to x_0, add bias and residual connection
        # x_0 * mixing_score broadcasts (Batch, Dim) * (Batch, 1) -> (Batch, Dim)
        out = x_0 * mixing_score + self.b + x_l
        return out


class PreActResBlock(nn.Module):
    """
    Full Pre-Activation ResNet Block.
    Topology: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add(Input)

    This topology improves gradient flow compared to Post-Activation blocks.
    """

    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

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
        return out + x


class TriBranchWDCNet(nn.Module):
    """
    Tri-Branch Wide-Deep-Cross Network.

    Architecture:
    1. Wide Branch: Linear bypass.
    2. Cross Branch: Asymmetric Vector-DCN (3 layers).
    3. Deep Branch: Deep Pre-Act ResNet Backbone (4 blocks).

    The outputs of Deep and Cross branches are concatenated and projected,
    then summed with the Wide branch logits.
    """

    def __init__(self, input_dim, num_classes=7):
        super().__init__()

        # 1. Wide Linear Branch
        self.wide_branch = WideBranch(input_dim, num_classes)

        # 2. Asymmetric Vector-Based Deep & Cross Branch
        # Decoupled from backbone, limited depth (Config.NUM_CROSS_LAYERS)
        self.cross_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(Config.NUM_CROSS_LAYERS)]
        )

        # 3. Deep Full Pre-Activation ResNet Backbone
        # Projection to hidden dimension
        self.deep_projection = nn.Linear(input_dim, Config.HIDDEN_DIM)
        # ResNet Blocks
        self.deep_backbone = nn.ModuleList(
            [
                PreActResBlock(Config.HIDDEN_DIM, Config.DROPOUT_RATE)
                for _ in range(Config.NUM_RESNET_BLOCKS)
            ]
        )

        # Combination Head
        # Concatenates Cross output (input_dim) and Deep output (hidden_dim)
        concat_dim = input_dim + Config.HIDDEN_DIM
        self.hybrid_head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # x: (Batch, Input_Dim)

        # --- Wide Branch ---
        wide_logits = self.wide_branch(x)

        # --- Cross Branch ---
        x_cross = x
        x_0 = x
        for layer in self.cross_layers:
            x_cross = layer(x_cross, x_0)

        # --- Deep Branch ---
        x_deep = self.deep_projection(x)
        for block in self.deep_backbone:
            x_deep = block(x_deep)

        # --- Combination ---
        # Concatenate outputs
        combined = torch.cat([x_cross, x_deep], dim=1)

        # Generate Hybrid Logits via hidden linear head
        hybrid_logits = self.hybrid_head(combined)

        # Final Summation of Logits
        final_logits = hybrid_logits + wide_logits

        return final_logits
