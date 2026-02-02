import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_mean, scatter_max
except ImportError:
    # Fallback or error handling if package is missing in specific env
    pass

from library.config import Config


class AtomicStream(nn.Module):
    """
    Distortion-Aware Point Processor.
    Processes local atomic features using a Wide MLP.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        layers = []

        # Initial projection
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Deep layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GlobalStream(nn.Module):
    """
    Bonding-Enhanced Context Processor.
    Processes global features including bond statistics.
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        # High-Capacity MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DualPooling(nn.Module):
    """
    Aggregates atomic embeddings using both Mean and Max pooling.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, batch_indices, num_graphs):
        """
        Args:
            x: Atomic embeddings (Total_Atoms, Hidden_Dim)
            batch_indices: Graph index for each atom (Total_Atoms,)
            num_graphs: Batch size
        Returns:
            Pooled embeddings (Batch_Size, 2 * Hidden_Dim)
        """
        # Mean Pooling
        mean_pool = scatter_mean(x, batch_indices, dim=0, dim_size=num_graphs)

        # Max Pooling (scatter_max returns values, indices)
        max_pool, _ = scatter_max(x, batch_indices, dim=0, dim_size=num_graphs)

        # Concatenate
        return torch.cat([mean_pool, max_pool], dim=1)


class GBAMSDSModel(nn.Module):
    """
    Global-Bonding Augmented Multi-Scale Deep Sets (GBA-MS-DS).
    Integrates local atomic context and global bonding statistics.
    """

    def __init__(self):
        super().__init__()

        # 1. Atomic Stream
        self.atomic_stream = AtomicStream(
            input_dim=Config.ATOMIC_FEATURE_DIM,
            hidden_dim=Config.ATOMIC_HIDDEN_DIM,
            num_layers=Config.ATOMIC_LAYERS,
            dropout=Config.ATOMIC_DROPOUT,
        )

        # 2. Global Stream
        self.global_stream = GlobalStream(
            input_dim=Config.GLOBAL_FEATURE_DIM,
            hidden_dim=Config.GLOBAL_HIDDEN_DIM,
            dropout=Config.GLOBAL_DROPOUT,
        )

        # 3. Aggregation
        self.pooling = DualPooling()

        # 4. Fusion Head
        # Input: (Mean_Atomic + Max_Atomic) + Global
        fusion_input_dim = (2 * Config.ATOMIC_HIDDEN_DIM) + Config.GLOBAL_HIDDEN_DIM

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, Config.FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(Config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(Config.FUSION_HIDDEN_DIM, Config.OUTPUT_DIM),
        )

    def forward(self, batch_data):
        """
        Forward pass for the model.
        Args:
            batch_data (dict): Dictionary containing:
                - 'atomic_features': Tensor (Total_Atoms, 17)
                - 'global_features': Tensor (Batch_Size, 29)
                - 'batch_indices': Tensor (Total_Atoms,)
        Returns:
            output: Tensor (Batch_Size, 2)
        """
        atomic_x = batch_data["atomic_features"]
        global_x = batch_data["global_features"]
        batch_indices = batch_data["batch_indices"]

        # Determine batch size dynamically
        batch_size = global_x.size(0)

        # 1. Encode Local Features
        atomic_emb = self.atomic_stream(atomic_x)

        # 2. Aggregate Local Features (Dual Pooling)
        pooled_atomic = self.pooling(atomic_emb, batch_indices, batch_size)

        # 3. Encode Global Features
        global_emb = self.global_stream(global_x)

        # 4. Late Fusion
        combined = torch.cat([pooled_atomic, global_emb], dim=1)

        # 5. Regression
        output = self.fusion_head(combined)

        return output
