import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library.config import (
    ATOM_FEATURES_DIM,
    GLOBAL_FEATURES_DIM,
    HIDDEN_DIM,
    ATOMIC_LAYERS,
    GLOBAL_LAYERS,
    FUSION_LAYERS,
    DROPOUT,
    USE_BATCH_NORM,
    SEED,
)

# Set seeds for reproducibility
torch.manual_seed(SEED)


class AtomicEncoder(nn.Module):
    """
    Ratio-Enhanced Point Processor.
    Processes dense atomic feature vectors into high-dimensional embeddings.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout, use_bn):
        super(AtomicEncoder, self).__init__()
        self.layers = nn.ModuleList()

        # Immediate expansion to hidden dimension
        current_dim = input_dim

        # Build hidden layers (all but last)
        for _ in range(num_layers - 1):
            layer = []
            layer.append(nn.Linear(current_dim, hidden_dim))
            if use_bn:
                layer.append(nn.BatchNorm1d(hidden_dim))
            layer.append(nn.ReLU())
            if dropout > 0:
                layer.append(nn.Dropout(dropout))
            self.layers.append(nn.Sequential(*layer))
            current_dim = hidden_dim

        # Final projection layer (Linear, no activation)
        self.final_layer = nn.Linear(current_dim, hidden_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)


class GlobalEncoder(nn.Module):
    """
    Anisotropy-Aware Context Encoder.
    Processes global crystal features.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout, use_bn):
        super(GlobalEncoder, self).__init__()
        self.layers = nn.ModuleList()

        # Internal width for global encoder (using 256 as suggested, or half hidden)
        # We'll use a fixed internal width relative to HIDDEN_DIM for consistency with "High-Capacity"
        internal_dim = 256

        current_dim = input_dim

        # Build hidden layers (all but last)
        for _ in range(num_layers - 1):
            layer = []
            layer.append(nn.Linear(current_dim, internal_dim))
            if use_bn:
                layer.append(nn.BatchNorm1d(internal_dim))
            layer.append(nn.ReLU())
            if dropout > 0:
                layer.append(nn.Dropout(dropout))
            self.layers.append(nn.Sequential(*layer))
            current_dim = internal_dim

        # Final projection to match HIDDEN_DIM for fusion
        self.final_layer = nn.Linear(current_dim, hidden_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)


class REMSWDSModel(nn.Module):
    """
    Ratio-Enhanced Multi-Scale Wide Deep Sets Model.

    Architecture:
    1. Atomic Stream: Processes local features -> Dual Pooling (Mean + Max).
    2. Global Stream: Processes global features.
    3. Fusion: Concatenates pooled atomic + global -> Regressor.
    """

    def __init__(
        self,
        atom_features_dim=ATOM_FEATURES_DIM,
        global_features_dim=GLOBAL_FEATURES_DIM,
        hidden_dim=HIDDEN_DIM,
        atomic_layers=ATOMIC_LAYERS,
        global_layers=GLOBAL_LAYERS,
        fusion_layers=FUSION_LAYERS,
        dropout=DROPOUT,
        use_bn=USE_BATCH_NORM,
    ):
        super(REMSWDSModel, self).__init__()

        # 1. Atomic Stream
        self.atomic_encoder = AtomicEncoder(
            input_dim=atom_features_dim,
            hidden_dim=hidden_dim,
            num_layers=atomic_layers,
            dropout=dropout,
            use_bn=use_bn,
        )

        # 2. Global Stream
        self.global_encoder = GlobalEncoder(
            input_dim=global_features_dim,
            hidden_dim=hidden_dim,
            num_layers=global_layers,
            dropout=dropout,
            use_bn=use_bn,
        )

        # 3. Fusion Head
        # Input dim = Atomic_Mean (hidden) + Atomic_Max (hidden) + Global (hidden)
        fusion_input_dim = hidden_dim * 3

        self.fusion_layers = nn.ModuleList()
        current_dim = fusion_input_dim

        # Build fusion layers
        # Gradually reduce dimension
        dims = [hidden_dim, hidden_dim // 2]
        # Ensure we have enough definitions for the requested number of layers
        # If fusion_layers > 2, repeat the last dim or adjust strategy.
        # Here we construct a list of sizes.
        layer_sizes = np.linspace(current_dim, 32, num=fusion_layers + 1).astype(int)[
            1:-1
        ]

        for size in layer_sizes:
            layer = []
            layer.append(nn.Linear(current_dim, size))
            if use_bn:
                layer.append(nn.BatchNorm1d(size))
            layer.append(nn.ReLU())
            if dropout > 0:
                layer.append(nn.Dropout(dropout))
            self.fusion_layers.append(nn.Sequential(*layer))
            current_dim = size

        # Final Output Layer (2 targets)
        self.output_layer = nn.Linear(current_dim, 2)

    def forward(self, atomic_features, batch_index, global_features):
        """
        Args:
            atomic_features: (Total_Atoms, Atom_Dim)
            batch_index: (Total_Atoms,)
            global_features: (Batch_Size, Global_Dim)
        """
        # 1. Atomic Stream
        atom_emb = self.atomic_encoder(atomic_features)

        # 2. Aggregation (Dual Pooling)
        # Scatter Mean
        pooled_mean = scatter_mean(atom_emb, batch_index, dim=0)
        # Scatter Max (returns tuple (values, indices))
        pooled_max, _ = scatter_max(atom_emb, batch_index, dim=0)

        # Handle case where scatter_max might return min_value for empty indices if any
        # Though in this dataset crystals always have atoms.

        # 3. Global Stream
        global_emb = self.global_encoder(global_features)

        # 4. Late Fusion
        # Concatenate: [Mean, Max, Global]
        fused = torch.cat([pooled_mean, pooled_max, global_emb], dim=1)

        # 5. Regression
        x = fused
        for layer in self.fusion_layers:
            x = layer(x)

        output = self.output_layer(x)
        return output
