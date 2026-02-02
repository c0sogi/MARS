import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer with Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l^T w) + b + x_l

    Implements 'Warm-Start' initialization where weights start near zero
    to prevent early multiplicative noise from destabilizing the backbone.
    """

    def __init__(self, input_dim):
        super(VectorCrossLayer, self).__init__()
        self.input_dim = input_dim

        # Weight vector w (Parameter)
        self.w = nn.Parameter(torch.empty(input_dim))

        # Bias b (Parameter)
        self.b = nn.Parameter(torch.zeros(input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Warm-Start Initialization: N(0, 1e-4)
        # This ensures the branch starts as an approximate identity mapping.
        nn.init.normal_(self.w, mean=0, std=1e-4)
        nn.init.zeros_(self.b)

    def forward(self, x0, xl):
        """
        Args:
            x0: Initial input features (Batch, Dim)
            xl: Output from previous layer (Batch, Dim)
        """
        # Calculate dot product (xl^T w) for each sample in batch.
        # xl: (Batch, Dim), w: (Dim) -> xl @ w: (Batch)
        # We unsqueeze to (Batch, 1) for broadcasting.
        dot_prod = torch.matmul(xl, self.w).unsqueeze(1)

        # Apply formula: x0 * scalar + b + xl
        out = x0 * dot_prod + self.b + xl
        return out


class PreActResBlock(nn.Module):
    """
    Full Pre-Activation Residual Block.
    Structure: Input -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add(Input)
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(PreActResBlock, self).__init__()

        # First sub-block
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)

        # Second sub-block
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x

        # Path 1
        out = self.bn1(x)
        out = F.relu(out)
        out = self.dropout1(out)
        out = self.linear1(out)

        # Path 2
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.linear2(out)

        # Residual Connection
        out = out + residual
        return out


class AsymmetricParallelNet(nn.Module):
    """
    Asymmetric Parallel Vector-DCN-ResNet with Normalized Fusion.

    Architecture:
    1. Input Layer
    2. Branch 1: Asymmetric Vector-Based DCN (Shallow, 3 layers)
    3. Branch 2: Deep Full Pre-Activation ResNet Backbone (Deep, 4 blocks, 512 width)
    4. Pre-Fusion Normalization (LayerNorm on both branches)
    5. Concatenation & Linear Classification
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        dcn_layers=Config.DCN_LAYERS,
        res_blocks=Config.RES_BLOCKS,
        hidden_size=Config.HIDDEN_SIZE,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super(AsymmetricParallelNet, self).__init__()

        # ------------------------------------------------
        # Branch 1: Asymmetric Vector-Based DCN
        # ------------------------------------------------
        # Decoupled from backbone, limited depth
        self.dcn_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(dcn_layers)]
        )

        # ------------------------------------------------
        # Branch 2: Deep Full Pre-Activation ResNet Backbone
        # ------------------------------------------------
        # Initial projection to hidden size
        self.res_projection = nn.Linear(input_dim, hidden_size)

        # Stack of Pre-Activation Residual Blocks
        self.res_backbone = nn.Sequential(
            *[PreActResBlock(hidden_size, dropout_rate) for _ in range(res_blocks)]
        )

        # ------------------------------------------------
        # Combination Head (Structural Innovation)
        # ------------------------------------------------
        # Pre-Fusion Normalization to stabilize combination
        self.ln_dcn = nn.LayerNorm(input_dim)
        self.ln_res = nn.LayerNorm(hidden_size)

        # Final Classification Layer
        # Input is concatenation of DCN output (input_dim) and ResNet output (hidden_size)
        self.classifier = nn.Linear(input_dim + hidden_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, Input_Dim)

        # --- DCN Branch Forward ---
        x_dcn = x
        for layer in self.dcn_layers:
            # Pass original input (x0) and current state (xl)
            x_dcn = layer(x, x_dcn)

        # --- ResNet Branch Forward ---
        x_res = self.res_projection(x)
        x_res = self.res_backbone(x_res)

        # --- Fusion ---
        # Apply Pre-Fusion Normalization
        x_dcn_norm = self.ln_dcn(x_dcn)
        x_res_norm = self.ln_res(x_res)

        # Concatenate normalized vectors
        concat_features = torch.cat([x_dcn_norm, x_res_norm], dim=1)

        # Final Prediction
        logits = self.classifier(concat_features)

        return logits
