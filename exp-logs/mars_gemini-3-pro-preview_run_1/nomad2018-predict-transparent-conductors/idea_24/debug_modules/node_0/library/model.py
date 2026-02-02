import torch
import torch.nn as nn
from library.config import Config


class DenseLayer(nn.Module):
    """
    A helper class for a dense layer block: Linear -> [BN] -> [ReLU] -> [Dropout].
    Handles reshaping for Batch Normalization if input is > 2D.
    """

    def __init__(self, in_dim, out_dim, dropout_rate=0.0, use_bn=True, activation=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.use_bn = use_bn
        self.activation = activation

        if use_bn:
            self.bn = nn.BatchNorm1d(out_dim)

        if activation:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)

        if self.use_bn:
            # BatchNorm1d expects (N, C). If input is (B, N, C), flatten to (B*N, C).
            original_shape = x.shape
            if x.dim() > 2:
                x_flat = x.view(-1, original_shape[-1])
                x_flat = self.bn(x_flat)
                x = x_flat.view(original_shape)
            else:
                x = self.bn(x)

        if self.activation:
            x = self.relu(x)
            x = self.dropout(x)

        return x


class AtomicStream(nn.Module):
    """
    Anisotropy-Aware Point Processor.
    Processes atomic features using a Wide MLP and aggregates them via Dual Pooling.
    """

    def __init__(self):
        super().__init__()
        in_dim = Config.ATOMIC_FEATURE_DIM
        hidden_dim = Config.ATOMIC_HIDDEN_DIM
        dropout = Config.DROPOUT_RATE

        # Wide MLP with Immediate Expansion
        # Layer 1
        self.layer1 = DenseLayer(
            in_dim, hidden_dim, dropout, use_bn=True, activation=True
        )
        # Layer 2
        self.layer2 = DenseLayer(
            hidden_dim, hidden_dim, dropout, use_bn=True, activation=True
        )
        # Final projection: Linear only (no activation, no BN)
        self.layer3 = DenseLayer(
            hidden_dim, hidden_dim, dropout=0.0, use_bn=False, activation=False
        )

    def forward(self, x, mask):
        # x: (Batch, Atoms, Features)
        # mask: (Batch, Atoms)

        # Point-wise MLP
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # (Batch, Atoms, Hidden)

        # Dual Pooling (Mean + Max)
        mask_expanded = mask.unsqueeze(-1)  # (Batch, Atoms, 1)

        # Apply mask (zero out padded atoms)
        x_masked = x * mask_expanded

        # 1. Global Mean Pooling
        sum_pooled = torch.sum(x_masked, dim=1)
        counts = torch.sum(mask_expanded, dim=1)
        counts = torch.clamp(counts, min=1e-9)
        mean_pooled = sum_pooled / counts

        # 2. Global Max Pooling
        # Set padded values to a very small number so they aren't picked as max
        x_for_max = x_masked.clone()
        x_for_max[mask_expanded.expand_as(x) == 0] = -1e9
        max_pooled = torch.max(x_for_max, dim=1)[0]

        # Concatenate
        out = torch.cat([mean_pooled, max_pooled], dim=1)  # (Batch, 2*Hidden)
        return out


class GlobalStream(nn.Module):
    """
    Thermodynamic Context Processor.
    Processes macroscopic features using a High-Capacity MLP.
    """

    def __init__(self):
        super().__init__()
        in_dim = Config.GLOBAL_FEATURE_DIM
        hidden_dim = Config.GLOBAL_HIDDEN_DIM
        dropout = Config.DROPOUT_RATE

        self.layer1 = DenseLayer(
            in_dim, hidden_dim, dropout, use_bn=True, activation=True
        )
        self.layer2 = DenseLayer(
            hidden_dim, hidden_dim, dropout, use_bn=True, activation=True
        )

    def forward(self, x):
        # x: (Batch, Features)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ACC_WDS(nn.Module):
    """
    Anisotropic Chemically-Contextualized Wide Deep Sets (ACC-WDS).
    Merges Atomic and Global streams via Late Fusion.
    """

    def __init__(self):
        super().__init__()

        self.atomic_stream = AtomicStream()
        self.global_stream = GlobalStream()

        # Fusion Head
        # Atomic output is 2 * ATOMIC_HIDDEN_DIM (due to Dual Pooling)
        # Global output is GLOBAL_HIDDEN_DIM
        fusion_in_dim = 2 * Config.ATOMIC_HIDDEN_DIM + Config.GLOBAL_HIDDEN_DIM
        hidden_dim = Config.FUSION_HIDDEN_DIM
        output_dim = Config.OUTPUT_DIM
        dropout = Config.DROPOUT_RATE

        self.fusion_layer1 = DenseLayer(
            fusion_in_dim, hidden_dim, dropout, use_bn=True, activation=True
        )
        self.fusion_layer2 = DenseLayer(
            hidden_dim, hidden_dim, dropout, use_bn=True, activation=True
        )
        self.regressor = nn.Linear(hidden_dim, output_dim)

    def forward(self, atomic_features, global_features, mask):
        # Process Atomic Stream
        atomic_emb = self.atomic_stream(atomic_features, mask)

        # Process Global Stream
        global_emb = self.global_stream(global_features)

        # Late Fusion
        combined = torch.cat([atomic_emb, global_emb], dim=1)

        # Regression
        x = self.fusion_layer1(combined)
        x = self.fusion_layer2(x)
        out = self.regressor(x)

        return out
