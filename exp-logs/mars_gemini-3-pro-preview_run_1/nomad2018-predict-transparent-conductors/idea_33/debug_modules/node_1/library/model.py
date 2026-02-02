import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library.config import Config


class AtomicStream(nn.Module):
    """
    Atomic Stream (Chemical Field Processor).
    Processes dense atomic features including chemical density fields.
    """

    def __init__(self):
        super(AtomicStream, self).__init__()

        input_dim = Config.ATOMIC_INPUT_DIM
        hidden_dim = Config.ATOMIC_HIDDEN_DIM
        num_layers = Config.ATOMIC_LAYERS
        dropout_rate = Config.DROPOUT_RATE

        self.layers = nn.ModuleList()

        # Build layers
        # Description: "Encoder: A Wide MLP ... Regularization: Batch Normalization and Dropout are applied after every ReLU activation."
        # "Output: A Linear projection (no activation) to the final embedding space."

        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        if Config.USE_BATCH_NORM:
            self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.Dropout(dropout_rate))

        # Intermediate layers (if any)
        # We need total num_layers. The last one is the projection.
        # So we add num_layers - 2 intermediate blocks.
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            if Config.USE_BATCH_NORM:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(dropout_rate))

        # Final Projection Layer (No activation, no regularization as per description "Output: A Linear projection")
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.projection(x)
        return x


class GlobalStream(nn.Module):
    """
    Global Stream (Thermodynamic Context).
    Processes macroscopic features.
    """

    def __init__(self):
        super(GlobalStream, self).__init__()

        input_dim = Config.GLOBAL_INPUT_DIM
        hidden_dim = Config.GLOBAL_HIDDEN_DIM
        num_layers = Config.GLOBAL_LAYERS
        dropout_rate = Config.DROPOUT_RATE

        self.layers = nn.ModuleList()

        # Build layers
        # Description: "Encoder: A High-Capacity MLP ... with Batch Normalization and Dropout"

        current_dim = input_dim
        for _ in range(num_layers):
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            if Config.USE_BATCH_NORM:
                self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GDCC_WDS(nn.Module):
    """
    Gaussian-Density Chemically-Contextualized Wide Deep Sets.

    Architecture:
    1. Atomic Stream: Processes atom-level features (including chemical densities).
    2. Global Stream: Processes unit-cell level features.
    3. Aggregation: Dual Pooling (Mean + Max) of atomic embeddings.
    4. Fusion: Concatenation of aggregated atomic and global embeddings.
    5. Head: Regressor MLP.
    """

    def __init__(self):
        super(GDCC_WDS, self).__init__()

        self.atomic_stream = AtomicStream()
        self.global_stream = GlobalStream()

        # Fusion Head
        # Atomic output is pooled (Mean + Max), so dimension is 2 * ATOMIC_HIDDEN_DIM
        # Global output dimension is GLOBAL_HIDDEN_DIM
        fusion_input_dim = (2 * Config.ATOMIC_HIDDEN_DIM) + Config.GLOBAL_HIDDEN_DIM
        hidden_dims = Config.FUSION_HIDDEN_DIMS
        output_dim = Config.OUTPUT_DIM
        dropout_rate = Config.DROPOUT_RATE

        self.fusion_layers = nn.ModuleList()

        current_dim = fusion_input_dim
        for h_dim in hidden_dims:
            self.fusion_layers.append(nn.Linear(current_dim, h_dim))
            self.fusion_layers.append(nn.ReLU())
            if Config.USE_BATCH_NORM:
                self.fusion_layers.append(nn.BatchNorm1d(h_dim))
            self.fusion_layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, atomic_features, global_features, batch_indices, num_graphs):
        """
        Args:
            atomic_features: (Total_Atoms, ATOMIC_INPUT_DIM)
            global_features: (Batch_Size, GLOBAL_INPUT_DIM)
            batch_indices: (Total_Atoms,) mapping atoms to their graph index
            num_graphs: int, batch size
        """
        # 1. Atomic Stream
        h_atomic = self.atomic_stream(atomic_features)

        # 2. Aggregation (Dual Pooling)
        # Global Mean Pooling
        h_mean = scatter_mean(h_atomic, batch_indices, dim=0, dim_size=num_graphs)
        # Global Max Pooling (scatter_max returns values, indices)
        h_max, _ = scatter_max(h_atomic, batch_indices, dim=0, dim_size=num_graphs)

        # Concatenate pooled features
        h_atomic_agg = torch.cat([h_mean, h_max], dim=1)

        # 3. Global Stream
        h_global = self.global_stream(global_features)

        # 4. Late Fusion
        h_fusion = torch.cat([h_atomic_agg, h_global], dim=1)

        # 5. Regression Head
        x = h_fusion
        for layer in self.fusion_layers:
            x = layer(x)

        out = self.output_layer(x)

        return out
