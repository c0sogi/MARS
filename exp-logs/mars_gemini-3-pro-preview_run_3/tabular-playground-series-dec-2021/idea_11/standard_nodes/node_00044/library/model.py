import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossNetV2(nn.Module):
    """
    Vector-based Deep & Cross Network (DCN-V2) with strict Dot-Product Mixing.
    Models explicit bounded-degree feature interactions.

    Formula: x_{l+1} = x_0 * (x_l^T . w_l) + b_l + x_l
    """

    def __init__(self, input_dim, num_layers=3):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_layers
        # Parameters W and b for each layer
        # W is a list of vectors (D,), b is a list of vectors (D,)
        self.W = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim)) for _ in range(num_layers)]
        )
        self.b = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim)) for _ in range(num_layers)]
        )
        self._init_parameters()

    def _init_parameters(self):
        for w, b in zip(self.W, self.b):
            # Initialize W with Xavier Uniform (treating it as a 1xD matrix for fan_in/out)
            nn.init.xavier_uniform_(w.unsqueeze(0))
            nn.init.zeros_(b)

    def forward(self, x0):
        # x0: (Batch, InputDim)
        xl = x0
        for i in range(self.num_layers):
            # 1. Calculate dot product scalar: (x_l . w_l)
            # x_l: (B, D), w_l: (D,) -> elementwise mul -> sum dim 1 -> (B, 1)
            dot_prod = torch.sum(xl * self.W[i], dim=1, keepdim=True)

            # 2. Scale x0 and add bias and residual
            # This implements strict Dot-Product Mixing
            xl = x0 * dot_prod + self.b[i] + xl
        return xl


class ResNeXtBlock(nn.Module):
    """
    ResNeXt Block for Tabular Data.
    Uses Grouped Linear Transformations (via Conv1d) to enforce block-diagonal sparsity
    and structural regularization.
    """

    def __init__(self, dim, cardinality=32):
        super(ResNeXtBlock, self).__init__()
        # Ensure dimension is divisible by cardinality
        assert (
            dim % cardinality == 0
        ), f"Dimension {dim} must be divisible by cardinality {cardinality}"

        # Layer 1: Grouped Linear (Conv1d 1x1)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=1, groups=cardinality, bias=False)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()

        # Layer 2: Grouped Linear (Conv1d 1x1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm1d(dim)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x):
        # x: (Batch, Dim, 1)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class ParallelDCNResNeXt(nn.Module):
    """
    Parallel DCN-ResNeXt Architecture.
    Hybrid model with a Cross Network branch for explicit interactions
    and a ResNeXt Backbone branch for deep implicit representation.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        dcn_layers=3,
        resnext_layers=3,
        d_model=1024,
        cardinality=32,
    ):
        super(ParallelDCNResNeXt, self).__init__()

        # Branch 1: Vector-DCN (Explicit Interactions)
        self.dcn = CrossNetV2(input_dim, num_layers=dcn_layers)

        # Branch 2: ResNeXt Backbone (Deep Implicit Features)
        # Initial projection to mix features into latent space
        self.project = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.BatchNorm1d(d_model), nn.ReLU()
        )

        # Stack of ResNeXt Blocks
        self.resnext_blocks = nn.Sequential(
            *[ResNeXtBlock(d_model, cardinality) for _ in range(resnext_layers)]
        )

        # Combination Head
        # Concatenates DCN output (input_dim) and ResNeXt output (d_model)
        self.head = nn.Linear(input_dim + d_model, num_classes)

        # Init projection and head
        self._init_weights()

    def _init_weights(self):
        # Initialize Linear layers in projection
        for m in self.project.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialize Head
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (Batch, InputDim)

        # Branch 1: DCN
        x_dcn = self.dcn(x)

        # Branch 2: ResNeXt
        # Project to latent space
        x_deep = self.project(x)

        # Reshape for Conv1d: (B, D) -> (B, D, 1)
        x_deep = x_deep.unsqueeze(2)

        # Pass through backbone
        x_deep = self.resnext_blocks(x_deep)

        # Flatten: (B, D, 1) -> (B, D)
        x_deep = x_deep.squeeze(2)

        # Concatenate outputs from both branches
        x_concat = torch.cat([x_dcn, x_deep], dim=1)

        # Final Classification
        logits = self.head(x_concat)

        return logits
