import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class VectorCrossLayer(nn.Module):
    """
    Asymmetric Vector-Based (Rank-1) Cross Layer.

    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l
    Where (.) denotes the dot product resulting in a scalar, and (*) denotes
    element-wise multiplication (broadcasting the scalar).

    This layer decouples interaction order from feature dimensionality and
    uses a warm-start initialization to behave as an identity mapping initially.
    """

    def __init__(self, in_features):
        super(VectorCrossLayer, self).__init__()
        self.in_features = in_features

        # Weight vector w corresponding to the interaction term
        self.weight = nn.Parameter(torch.empty(in_features))
        # Bias vector b
        self.bias = nn.Parameter(torch.empty(in_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Warm-Start Initialization: N(0, 1e-4)
        # Ensures the interaction term is initially negligible, preserving signal flow.
        nn.init.normal_(self.weight, mean=0, std=1e-4)
        nn.init.zeros_(self.bias)

    def forward(self, x, x0):
        """
        Args:
            x: Input tensor from the previous layer (B, D)
            x0: Original input tensor to the DCN stack (B, D)
        Returns:
            Output tensor (B, D)
        """
        # Compute scalar interaction per sample: (B, D) dot (D,) -> (B,)
        # We use matmul for (B, D) x (D, 1) -> (B, 1) for easier broadcasting
        interaction_scalar = torch.matmul(x, self.weight.unsqueeze(1))

        # Apply formula: x0 * scalar + b + x
        out = x0 * interaction_scalar + self.bias + x
        return out


class PreActResNetBlock(nn.Module):
    """
    Deep Full Pre-Activation ResNet Block.

    Topology: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add(Input)

    This topology optimizes gradient flow (Pre-Act) and capacity (2 layers per block).
    """

    def __init__(self, features, dropout_rate):
        super(PreActResNetBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(features)
        self.linear1 = nn.Linear(features, features)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.bn2 = nn.BatchNorm1d(features)
        self.linear2 = nn.Linear(features, features)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Pre-activation path
        out = self.bn1(x)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.linear1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.linear2(out)

        # Residual connection
        return out + x


class AsymmetricDCNResNet(nn.Module):
    """
    Asymmetric Deep Parallel Vector-DCN-ResNet Architecture.

    Combines a shallow, high-fidelity feature interaction branch (DCN) with a
    deep, high-capacity representation branch (ResNet).
    """

    def __init__(
        self,
        input_dim,
        num_classes=Config.NUM_CLASSES,
        dcn_layers=Config.DCN_LAYERS,
        resnet_blocks=Config.RESNET_BLOCKS,
        hidden_dim=Config.HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super(AsymmetricDCNResNet, self).__init__()

        # ==========================
        # Branch 1: Asymmetric DCN
        # ==========================
        # Explicitly limited depth to capture low-order interactions without noise.
        # Uses VectorCrossLayer for Rank-1 mixing.
        self.dcn_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(dcn_layers)]
        )

        # ==========================
        # Branch 2: Deep ResNet
        # ==========================
        # Projection to hidden dimension
        self.resnet_projection = nn.Linear(input_dim, hidden_dim)

        # Deep backbone with Pre-Activation blocks
        self.resnet_backbone = nn.Sequential(
            *[PreActResNetBlock(hidden_dim, dropout_rate) for _ in range(resnet_blocks)]
        )

        # ==========================
        # Combination Head
        # ==========================
        # Concatenates output of DCN (input_dim) and ResNet (hidden_dim)
        concat_dim = input_dim + hidden_dim
        self.head = nn.Linear(concat_dim, num_classes)

    def forward(self, x):
        # Branch 1: DCN
        x_dcn = x
        x0 = x
        for layer in self.dcn_layers:
            x_dcn = layer(x_dcn, x0)

        # Branch 2: ResNet
        x_res = self.resnet_projection(x)
        x_res = self.resnet_backbone(x_res)

        # Concatenate
        x_combined = torch.cat([x_dcn, x_res], dim=1)

        # Classification
        logits = self.head(x_combined)

        return logits
