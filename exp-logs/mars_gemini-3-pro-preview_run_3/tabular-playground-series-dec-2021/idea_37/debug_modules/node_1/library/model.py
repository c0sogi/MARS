import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import RESNET_BLOCKS, DCN_LAYERS, HIDDEN_DIM, DROPOUT, NOISE_STD


class GaussianNoise(nn.Module):
    """
    Applies Gaussian noise to the input tensor during training.
    Used for continuous feature regularization to prevent overfitting in deep models.
    """

    def __init__(self, std=0.01):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer.
    Implements the formula: x_{l+1} = x_0 * (x_l^T w) + b + x_l
    Uses warm-start initialization to begin as an approximate identity mapping.
    """

    def __init__(self, input_dim):
        super(VectorCrossLayer, self).__init__()
        self.input_dim = input_dim

        # Weight vector w and bias b
        self.weight = nn.Parameter(torch.Tensor(input_dim))
        self.bias = nn.Parameter(torch.Tensor(input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize with near-zero standard deviation for warm-start
        nn.init.normal_(self.weight, mean=0, std=1e-4)
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        # x0: Initial input features (Batch, Dim)
        # xl: Output from previous layer (Batch, Dim)

        # Calculate the scalar score for each sample: (xl . w)
        # Result shape: (Batch, 1)
        score = torch.sum(xl * self.weight, dim=1, keepdim=True)

        # Apply mixing: x0 * score + b + xl
        out = x0 * score + self.bias + xl
        return out


class PreActResBlock(nn.Module):
    """
    Full Pre-Activation Residual Block.
    Topology: Input -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add(Input)
    """

    def __init__(self, dim, dropout_rate=0.2):
        super(PreActResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.bn2 = nn.BatchNorm1d(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(dropout_rate)

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


class ParallelDCNResNet(nn.Module):
    """
    Deep Parallel Vector-DCN-ResNet (5-Block Scaled) with Input Gaussian Noise.
    Combines a shallow asymmetric DCN branch with a deep ResNet backbone.
    """

    def __init__(self, input_info):
        super(ParallelDCNResNet, self).__init__()

        cont_dim = input_info["cont_dim"]
        bin_dim = input_info["bin_dim"]
        num_classes = input_info["num_classes"]

        self.input_dim = cont_dim + bin_dim

        # 1. Input Gaussian Noise (Applied only to continuous features)
        self.noise = GaussianNoise(std=NOISE_STD)

        # 2. Branch 1: Asymmetric Vector-Based Deep & Cross Network
        # Decoupled from backbone, limited depth
        self.dcn_layers = nn.ModuleList(
            [VectorCrossLayer(self.input_dim) for _ in range(DCN_LAYERS)]
        )

        # 3. Branch 2: Deep Full Pre-Activation ResNet Backbone
        # Projects input to hidden dimension first
        self.resnet_projection = nn.Linear(self.input_dim, HIDDEN_DIM)

        # Stack of Pre-Activation Residual Blocks
        self.resnet_blocks = nn.Sequential(
            *[
                PreActResBlock(HIDDEN_DIM, dropout_rate=DROPOUT)
                for _ in range(RESNET_BLOCKS)
            ]
        )

        # 4. Combination Head
        # Concatenates outputs from both branches
        # DCN output dim: input_dim
        # ResNet output dim: HIDDEN_DIM
        concat_dim = self.input_dim + HIDDEN_DIM
        self.head = nn.Linear(concat_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Kaiming initialization for Linear layers with ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Note: VectorCrossLayer initializes itself in its __init__

    def forward(self, x_cont, x_bin):
        # Apply Gaussian noise to continuous features during training
        x_cont = self.noise(x_cont)

        # Concatenate continuous and binary features
        x0 = torch.cat([x_cont, x_bin], dim=1)

        # --- Branch 1: DCN ---
        x_dcn = x0
        for layer in self.dcn_layers:
            x_dcn = layer(x0, x_dcn)

        # --- Branch 2: ResNet ---
        x_res = self.resnet_projection(x0)
        x_res = self.resnet_blocks(x_res)

        # --- Combination ---
        # Concatenate the outputs of both branches
        combined = torch.cat([x_dcn, x_res], dim=1)

        # Final classification
        logits = self.head(combined)

        return logits
