import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library.config import Config


class AtomicStream(nn.Module):
    """
    Chemically-Aware Point Processor.
    Processes individual atomic features (Identity + Spatial + Chemical Neighbors).
    Uses a Wide MLP with Batch Normalization and Dropout.
    """

    def __init__(self):
        super(AtomicStream, self).__init__()

        input_dim = Config.ATOMIC_FEATURE_DIM
        hidden_dim = Config.ATOMIC_HIDDEN_DIM
        latent_dim = Config.ATOMIC_LATENT_DIM
        dropout = Config.DROPOUT_RATE

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Projection to latent space (no activation/BN at the end of encoder usually)
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (total_atoms_in_batch, ATOMIC_FEATURE_DIM)
        Returns:
            Tensor of shape (total_atoms_in_batch, ATOMIC_LATENT_DIM)
        """
        return self.net(x)


class GlobalStream(nn.Module):
    """
    Thermodynamic Context Encoder.
    Processes macroscopic crystal features.
    """

    def __init__(self):
        super(GlobalStream, self).__init__()

        input_dim = Config.GLOBAL_FEATURE_DIM
        hidden_dim = Config.GLOBAL_HIDDEN_DIM
        latent_dim = Config.GLOBAL_LATENT_DIM
        dropout = Config.DROPOUT_RATE

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, GLOBAL_FEATURE_DIM)
        Returns:
            Tensor of shape (batch_size, GLOBAL_LATENT_DIM)
        """
        return self.net(x)


class CRNDSModel(nn.Module):
    """
    Chemically-Resolved Neighborhood Deep Sets (CRN-DS) Model.

    Architecture:
    1. Atomic Stream: Encodes atom-level features.
    2. Dual Pooling: Aggregates atomic embeddings via Mean and Max pooling.
    3. Global Stream: Encodes crystal-level features.
    4. Fusion Head: Concatenates aggregated atomic and global features to predict targets.
    """

    def __init__(self):
        super(CRNDSModel, self).__init__()

        self.atomic_stream = AtomicStream()
        self.global_stream = GlobalStream()

        # Fusion Input Dimension:
        # (Atomic Mean Pool) + (Atomic Max Pool) + (Global Embedding)
        fusion_input_dim = (Config.ATOMIC_LATENT_DIM * 2) + Config.GLOBAL_LATENT_DIM
        hidden_dim = Config.FUSION_HIDDEN_DIM
        output_dim = Config.NUM_TARGETS
        dropout = Config.DROPOUT_RATE

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, atomic_features, global_features, batch_index):
        """
        Args:
            atomic_features: (total_atoms, ATOMIC_FEATURE_DIM)
            global_features: (batch_size, GLOBAL_FEATURE_DIM)
            batch_index: (total_atoms,) LongTensor mapping atoms to their crystal index in batch

        Returns:
            Tensor of shape (batch_size, NUM_TARGETS)
        """
        # 1. Process Atomic Stream
        # Shape: (total_atoms, ATOMIC_LATENT_DIM)
        atom_embeddings = self.atomic_stream(atomic_features)

        # Cite debug_lesson_2: Handle Empty Graphs by Anchoring Batch Size to Graph-Level Features
        batch_size = global_features.size(0)

        # 2. Dual Pooling (Deep Sets Aggregation)
        # Scatter Mean: (batch_size, ATOMIC_LATENT_DIM)
        mean_pool = scatter_mean(
            atom_embeddings, batch_index, dim=0, dim_size=batch_size
        )

        # Scatter Max: (batch_size, ATOMIC_LATENT_DIM)
        # scatter_max returns (values, indices), we only need values
        max_pool, _ = scatter_max(
            atom_embeddings, batch_index, dim=0, dim_size=batch_size
        )

        # Handle potential numerical instability with max pooling on empty sets
        # scatter_max defaults to lowest representable value for empty indices
        # We replace values for empty graphs (which have 0 atoms) with 0.
        atom_counts = torch.bincount(batch_index, minlength=batch_size)
        empty_mask = (atom_counts == 0).unsqueeze(1)
        if empty_mask.any():
            max_pool = max_pool.masked_fill(empty_mask, 0.0)

        # 3. Process Global Stream
        # Shape: (batch_size, GLOBAL_LATENT_DIM)
        global_embeddings = self.global_stream(global_features)

        # 4. Late Fusion
        # Concatenate: [MeanPool, MaxPool, Global]
        combined = torch.cat([mean_pool, max_pool, global_embeddings], dim=1)

        # 5. Prediction
        output = self.fusion_head(combined)

        return output
