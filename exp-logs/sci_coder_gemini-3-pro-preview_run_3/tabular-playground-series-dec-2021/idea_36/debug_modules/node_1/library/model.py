import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Implements the formula: x_{l+1} = x_0 \odot (x_l^T w) + b + x_l

    This layer captures explicit feature interactions efficiently by modulating
    the original embedding x_0 based on a projection of the current state x_l.
    """

    def __init__(self, dim):
        super().__init__()
        # Learnable parameters: weight vector w and bias b
        self.w = nn.Parameter(torch.empty(dim))
        self.b = nn.Parameter(torch.empty(dim))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes weights to ensure Warm-Start behavior.
        w ~ N(0, 1e-4): Starts near zero to approximate identity mapping.
        b = 0
        """
        nn.init.normal_(self.w, mean=0, std=1e-4)
        nn.init.zeros_(self.b)

    def forward(self, x, x0):
        """
        Args:
            x: Current layer input (Batch, Dim)
            x0: Original projection/embedding (Batch, Dim)
        """
        # Compute scalar score per sample: x_l^T w
        # Element-wise multiplication followed by sum over feature dimension
        # (Batch, Dim) * (Dim) -> (Batch, Dim) -> sum -> (Batch, 1)
        score = torch.sum(x * self.w, dim=1, keepdim=True)

        # Apply update: x_{l+1} = x_0 * score + b + x
        # Broadcasting score (Batch, 1) across x0 (Batch, Dim)
        return x0 * score + self.b + x


class PreActResBlock(nn.Module):
    """
    Full Pre-Activation Residual Block.
    Topology: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add(Input)

    This topology improves gradient flow in deeper networks compared to Post-Activation.
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
        return x + out


class DeepSupervisedNet(nn.Module):
    """
    Deeply-Supervised Asymmetric Parallel Vector-DCN-ResNet.

    Features:
    1. Parallel processing of inputs via DCN (feature interactions) and ResNet (deep representation).
    2. Asymmetric depth: Shallow DCN (3 layers) vs Deep ResNet (5 blocks).
    3. Deep Supervision: Auxiliary head attached to ResNet Block 3 to aid optimization.
    """

    def __init__(self, input_dim, num_classes, config: Config):
        super().__init__()

        # Configuration extraction
        self.dcn_dim = config.dcn_projection_dim
        self.res_dim = config.resnet_width
        # Auxiliary head attached after Block 3 (0-indexed index 2)
        self.aux_attach_idx = 2

        # --- Branch 1: Vector-Based DCN ---
        # Projection to define x_0 for the DCN stack
        self.dcn_project = nn.Linear(input_dim, self.dcn_dim)
        # Stack of VectorCrossLayers
        self.dcn_layers = nn.ModuleList(
            [VectorCrossLayer(self.dcn_dim) for _ in range(config.dcn_layers)]
        )

        # --- Branch 2: Deep ResNet ---
        # Projection to ResNet width
        self.res_project = nn.Linear(input_dim, self.res_dim)
        # Stack of PreActResBlocks
        self.res_blocks = nn.ModuleList(
            [
                PreActResBlock(self.res_dim, config.dropout_rate)
                for _ in range(config.resnet_blocks)
            ]
        )

        # --- Heads ---
        # Auxiliary Classification Head (Linear)
        self.aux_head = nn.Linear(self.res_dim, num_classes)

        # Primary Classification Head (Linear)
        # Concatenates output of DCN and ResNet branches
        self.final_head = nn.Linear(self.dcn_dim + self.res_dim, num_classes)

    def forward(self, x):
        # --- DCN Branch Forward ---
        # Create x_0
        x_dcn_0 = self.dcn_project(x)
        x_dcn = x_dcn_0
        # Apply Cross Layers
        for layer in self.dcn_layers:
            x_dcn = layer(x_dcn, x_dcn_0)

        # --- ResNet Branch Forward ---
        x_res = self.res_project(x)
        aux_logits = None

        for i, block in enumerate(self.res_blocks):
            x_res = block(x_res)
            # Capture auxiliary output after specified block
            if i == self.aux_attach_idx:
                aux_logits = self.aux_head(x_res)

        # --- Combination & Output ---
        # Concatenate features from both branches
        combined = torch.cat([x_dcn, x_res], dim=1)

        # Compute primary logits
        primary_logits = self.final_head(combined)

        # Return tuple of (primary, auxiliary)
        # Auxiliary logits are used for loss calculation during training
        return primary_logits, aux_logits
