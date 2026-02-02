import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from library.config import Config


class AtomicEncoder(nn.Module):
    """
    Wide MLP to encode local atomic features (multi-scale context).
    Applies Batch Normalization and Dropout after activations to handle
    heterogeneous scales and prevent overfitting.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(AtomicEncoder, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))

        # Output projection (Linear, no activation as per strategy)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class GlobalEncoder(nn.Module):
    """
    High-capacity MLP to encode macroscopic global features.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(GlobalEncoder, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        # Hidden layers (if any)
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MSCWDSModel(nn.Module):
    """
    Multi-Scale Context Wide Deep Sets Model.

    Integrates:
    1. Atomic Stream: Processes local features, aggregates via Dual Pooling (Mean + Max).
    2. Global Stream: Processes global features.
    3. Fusion Head: Concatenates embeddings and predicts targets.
    """

    def __init__(self):
        super(MSCWDSModel, self).__init__()

        # Hyperparameters from Config
        atomic_in = Config.ATOMIC_INPUT_DIM
        atomic_hidden = Config.ATOMIC_HIDDEN_DIM
        atomic_layers = Config.ATOMIC_LAYERS

        global_in = Config.GLOBAL_INPUT_DIM
        global_hidden = Config.GLOBAL_HIDDEN_DIM
        global_layers = Config.GLOBAL_LAYERS

        dropout = Config.DROPOUT_RATE
        num_targets = Config.NUM_TARGETS

        # 1. Atomic Stream
        self.atomic_encoder = AtomicEncoder(
            atomic_in, atomic_hidden, atomic_layers, dropout
        )

        # 2. Global Stream
        self.global_encoder = GlobalEncoder(
            global_in, global_hidden, global_layers, dropout
        )

        # 3. Fusion Head
        # Dual pooling (Mean + Max) doubles the atomic embedding dimension
        fusion_input_dim = (atomic_hidden * 2) + global_hidden

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_targets),
        )

    def forward(self, batch_data):
        """
        Args:
            batch_data (dict): Dictionary containing:
                - 'atomic_features': [Total_Atoms, 16]
                - 'batch_index': [Total_Atoms] (Indices mapping atoms to crystals)
                - 'global_features': [Batch_Size, 12]

        Returns:
            predictions: [Batch_Size, 2]
        """
        atomic_feats = batch_data["atomic_features"]
        batch_idx = batch_data["batch_index"]
        global_feats = batch_data["global_features"]

        batch_size = global_feats.size(0)

        # --- Atomic Stream ---
        # Encode atomic features
        # Shape: [Total_Atoms, atomic_hidden]
        atomic_embeddings = self.atomic_encoder(atomic_feats)

        # Dual Pooling: Aggregate atoms back to crystal level
        # Mean Pooling
        # Shape: [Batch_Size, atomic_hidden]
        # Explicitly pass dim_size to handle empty graphs at the end of batch. Cite debug_lesson_2
        mean_pool = scatter_mean(
            atomic_embeddings, batch_idx, dim=0, dim_size=batch_size
        )

        # Max Pooling (scatter_max returns values and indices, we take values)
        # Shape: [Batch_Size, atomic_hidden]
        max_pool, _ = scatter_max(
            atomic_embeddings, batch_idx, dim=0, dim_size=batch_size
        )

        # Handle empty graphs in max_pool (which are initialized to small sentinel values)
        # We can detect them by checking if the graph had any atoms. Cite debug_lesson_9
        counts = torch.bincount(batch_idx, minlength=batch_size)
        empty_mask = counts == 0
        if empty_mask.any():
            max_pool[empty_mask] = 0.0

        # Concatenate pooled embeddings
        # Shape: [Batch_Size, atomic_hidden * 2]
        crystal_atomic_embedding = torch.cat([mean_pool, max_pool], dim=1)

        # --- Global Stream ---
        # Encode global features
        # Shape: [Batch_Size, global_hidden]
        global_embedding = self.global_encoder(global_feats)

        # --- Late Fusion ---
        # Concatenate atomic and global representations
        # Shape: [Batch_Size, atomic_hidden*2 + global_hidden]
        combined_embedding = torch.cat(
            [crystal_atomic_embedding, global_embedding], dim=1
        )

        # Predict targets
        # Shape: [Batch_Size, 2]
        predictions = self.fusion_head(combined_embedding)

        return predictions
