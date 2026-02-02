import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l^T w) + b + x_l
    """

    def __init__(self, input_dim):
        super(VectorCrossLayer, self).__init__()
        self.input_dim = input_dim
        # Weight w: (D, 1) to produce a scalar score per sample via dot product with x_l
        self.weight = nn.Parameter(torch.Tensor(input_dim, 1))
        # Bias b: (D, )
        self.bias = nn.Parameter(torch.Tensor(input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Warm-Start Initialization: Near-Zero Standard Deviation
        # This ensures the layer starts as an approximate identity mapping.
        nn.init.normal_(self.weight, mean=0, std=Config.DCN_INIT_STD)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        """
        Args:
            x0: Initial input features (B, D)
            xl: Output from previous layer (B, D)
        Returns:
            x_{l+1}: (B, D)
        """
        # Calculate scalar score: x_l^T w -> (B, 1)
        score = torch.matmul(xl, self.weight)

        # Apply formula: x0 * score + b + xl
        # Broadcasting score (B, 1) over x0 (B, D)
        out = x0 * score + self.bias + xl
        return out


class PreActResBlock(nn.Module):
    """
    Full Pre-Activation Residual Block.
    Structure: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add(Input)
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(PreActResBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)

        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
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

        # Residual Connection
        return x + out


class DualViewDCNResNet(nn.Module):
    """
    Deeply-Supervised (Annealed) Dual-View Asymmetric Parallel Vector-DCN-ResNet.

    Architecture:
    - Input: Concatenated Dual-View Features (Physical + Statistical)
    - Branch 1: Asymmetric Vector-Based DCN (3 Layers)
    - Branch 2: Deeply-Supervised ResNet Backbone (5 Blocks)
    - Aux Head: Attached to Block 3 of ResNet
    - Output: Concatenation of Branch 1 & 2 -> Linear -> Logits
    """

    def __init__(self, input_dim, num_classes):
        super(DualViewDCNResNet, self).__init__()

        # --- Branch 1: Asymmetric Vector-Based DCN ---
        # Keeps dimension as input_dim
        self.dcn_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(Config.DCN_LAYERS)]
        )

        # --- Branch 2: ResNet Backbone ---
        # Stem: Project input_dim -> hidden_dim
        self.backbone_stem = nn.Linear(input_dim, Config.HIDDEN_DIM)

        # Backbone Blocks
        self.blocks = nn.ModuleList()
        for _ in range(Config.BACKBONE_BLOCKS):
            self.blocks.append(PreActResBlock(Config.HIDDEN_DIM, Config.DROPOUT))

        # Auxiliary Head (Linear)
        # Maps hidden_dim -> num_classes
        self.aux_head = nn.Linear(Config.HIDDEN_DIM, num_classes)

        # --- Final Combination Head ---
        # Concatenates DCN output (input_dim) and ResNet output (hidden_dim)
        self.final_linear = nn.Linear(input_dim + Config.HIDDEN_DIM, num_classes)

        self.aux_block_index = Config.AUX_BLOCK_INDEX

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, input_dim)
        Returns:
            logits: Primary classification logits (B, num_classes)
            aux_logits: Auxiliary classification logits (B, num_classes) or None
        """
        # --- Branch 1: DCN Forward ---
        xl = x
        for layer in self.dcn_layers:
            xl = layer(x, xl)
        dcn_out = xl

        # --- Branch 2: ResNet Forward ---
        h = self.backbone_stem(x)

        aux_logits = None

        for i, block in enumerate(self.blocks):
            h = block(h)
            # Capture Auxiliary Output
            if i == self.aux_block_index:
                aux_logits = self.aux_head(h)

        resnet_out = h

        # --- Combination ---
        # Concatenate outputs from both branches
        combined = torch.cat([dcn_out, resnet_out], dim=1)

        # Final Classification
        logits = self.final_linear(combined)

        return logits, aux_logits
