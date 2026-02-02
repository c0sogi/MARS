import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library.config import Config


class AtomicStream(nn.Module):
    """
    Wide Geometric Point Processor for atomic features.
    Processes individual atoms using a wide MLP with Batch Normalization and Dropout.
    Projects 9-dimensional local features into a high-dimensional latent space
    and aggregates them using Dual Pooling (Mean + Max).
    """

    def __init__(self):
        super().__init__()
        input_dim = Config.ATOMIC_INPUT_DIM
        hidden_dim = Config.ATOMIC_HIDDEN_DIM
        # Output dimension before pooling. We choose half of hidden_dim to manage
        # the size after concatenation of mean and max pooling.
        output_dim = 256
        dropout = Config.DROPOUT_RATE

        # Wide MLP layers with BN-ReLU-Dropout block structure
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Final projection without activation to preserve vector magnitude for pooling
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, atomic_features, batch_indices, num_graphs=None):
        """
        Args:
            atomic_features: (N_total_atoms, 9) Tensor containing one-hot species,
                             centered coords, d_min, and d_mean.
            batch_indices: (N_total_atoms,) LongTensor indicating which crystal
                           each atom belongs to.
            num_graphs: (int, optional) The number of graphs in the batch.
                        Used to ensure output size consistency even if some graphs are empty.

        Returns:
            aggregated_features: (Batch_Size, output_dim * 2) Tensor.
        """
        # 1. Point-wise processing
        # Shape: (N_atoms, output_dim)
        atom_embeddings = self.net(atomic_features)

        # 2. Aggregation (Dual Pooling)
        # Global Mean Pooling: Captures average properties (e.g., average packing)
        # dim_size ensures we get an output for every graph index up to num_graphs-1,
        # filling with 0 if no atoms map to that index.
        mean_pool = scatter_mean(
            atom_embeddings, batch_indices, dim=0, dim_size=num_graphs
        )

        # Global Max Pooling: Captures salient structural features (e.g., specific bond constraints)
        # scatter_max returns (values, indices), we only need values
        max_pool, _ = scatter_max(
            atom_embeddings, batch_indices, dim=0, dim_size=num_graphs
        )

        # Concatenate to form the final atomic representation
        # Shape: (Batch_Size, 512)
        aggregated = torch.cat([mean_pool, max_pool], dim=1)

        return aggregated


class GlobalStream(nn.Module):
    """
    Thermodynamic Context Processor for global crystal features.
    Processes macroscopic descriptors (lattice, volume, density, stoichiometry)
    using a high-capacity MLP.
    """

    def __init__(self):
        super().__init__()
        input_dim = Config.GLOBAL_INPUT_DIM
        hidden_dim = Config.GLOBAL_HIDDEN_DIM
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
            # Final projection
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, global_features):
        """
        Args:
            global_features: (Batch_Size, 12) Tensor.

        Returns:
            global_embeddings: (Batch_Size, hidden_dim) Tensor.
        """
        return self.net(global_features)


class PGWDS(nn.Module):
    """
    Parsimonious Geometric Wide Deep Sets (PG-WDS) Model.
    Combines the Atomic Stream and Global Stream via Late Fusion to predict
    formation energy and bandgap energy.
    """

    def __init__(self):
        super().__init__()

        self.atomic_stream = AtomicStream()
        self.global_stream = GlobalStream()

        # Calculate fusion input dimension
        # Atomic Stream output: 256 (mean) + 256 (max) = 512
        atomic_out_dim = 256 * 2
        # Global Stream output: 256
        global_out_dim = Config.GLOBAL_HIDDEN_DIM

        fusion_input_dim = atomic_out_dim + global_out_dim
        hidden_dim = Config.FUSION_HIDDEN_DIM
        dropout = Config.DROPOUT_RATE

        # Fusion Head / Regressor
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Final regression to 2 targets: [formation_energy, bandgap_energy]
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, batch_data):
        """
        Args:
            batch_data: Dictionary containing:
                - atomic_features: (N_atoms, 9)
                - batch_indices: (N_atoms,)
                - global_features: (Batch_Size, 12)

        Returns:
            predictions: (Batch_Size, 2) Tensor containing log-transformed targets.
        """
        atomic_feats = batch_data["atomic_features"]
        batch_idx = batch_data["batch_indices"]
        global_feats = batch_data["global_features"]

        # Determine batch size from global features (which are always present per graph)
        batch_size = global_feats.size(0)

        # 1. Process Atomic Stream
        # Pass batch_size (num_graphs) to handle potential empty crystals
        # Shape: (Batch_Size, 512)
        atomic_embedding = self.atomic_stream(
            atomic_feats, batch_idx, num_graphs=batch_size
        )

        # 2. Process Global Stream
        # Shape: (Batch_Size, 256)
        global_embedding = self.global_stream(global_feats)

        # 3. Late Fusion
        # Concatenate along feature dimension
        # Shape: (Batch_Size, 768)
        combined = torch.cat([atomic_embedding, global_embedding], dim=1)

        # 4. Regression
        # Shape: (Batch_Size, 2)
        output = self.fusion_head(combined)

        return output
