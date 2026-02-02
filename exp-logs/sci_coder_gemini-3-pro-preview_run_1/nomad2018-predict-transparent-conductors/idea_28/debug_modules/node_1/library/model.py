import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library.config import Config


class WideBlock(nn.Module):
    """
    A wide MLP block consisting of Linear -> Batch Normalization -> ReLU -> Dropout.
    This structure is designed to maintain high capacity while providing necessary regularization
    to prevent overfitting on the dense feature set.
    """

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(WideBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class AtomicStream(nn.Module):
    """
    Processes atomic (node-level) features using a wide MLP backbone.
    Aggregates node embeddings into a graph-level representation using Dual Pooling (Mean + Max).
    """

    def __init__(self):
        super(AtomicStream, self).__init__()
        input_dim = Config.ATOMIC_INPUT_DIM
        hidden_dim = Config.ATOMIC_HIDDEN_DIM
        num_layers = Config.ATOMIC_LAYERS
        dropout = Config.DROPOUT_RATE

        layers = []
        # First layer projects input to hidden dimension
        layers.append(WideBlock(input_dim, hidden_dim, dropout))

        # Subsequent layers maintain the wide hidden dimension
        for _ in range(num_layers - 1):
            layers.append(WideBlock(hidden_dim, hidden_dim, dropout))

        self.encoder = nn.Sequential(*layers)

        # Final linear projection before pooling (no activation/BN here to allow full range)
        self.final_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, batch_indices, num_graphs=None):
        """
        Args:
            x: Atomic features tensor of shape (N_atoms, ATOMIC_INPUT_DIM).
            batch_indices: Tensor of shape (N_atoms,) indicating the graph index for each atom.
            num_graphs: Integer indicating the batch size (number of graphs), used for scatter dimension.

        Returns:
            Tensor of shape (Batch_Size, 2 * ATOMIC_HIDDEN_DIM).
        """
        # Encode features
        x = self.encoder(x)
        x = self.final_projection(x)

        # Dual Pooling: Concatenate Mean and Max pooling results
        # scatter_mean and scatter_max aggregate features based on batch_indices
        mean_pool = scatter_mean(x, batch_indices, dim=0, dim_size=num_graphs)
        max_pool, _ = scatter_max(x, batch_indices, dim=0, dim_size=num_graphs)

        # Concatenate along the feature dimension
        out = torch.cat([mean_pool, max_pool], dim=1)
        return out


class GlobalStream(nn.Module):
    """
    Processes global (macroscopic) features using a wide MLP backbone.
    """

    def __init__(self):
        super(GlobalStream, self).__init__()
        input_dim = Config.GLOBAL_INPUT_DIM
        hidden_dim = Config.GLOBAL_HIDDEN_DIM
        num_layers = Config.GLOBAL_LAYERS
        dropout = Config.DROPOUT_RATE

        layers = []
        # First layer
        layers.append(WideBlock(input_dim, hidden_dim, dropout))

        # Subsequent layers
        for _ in range(num_layers - 1):
            layers.append(WideBlock(hidden_dim, hidden_dim, dropout))

        self.encoder = nn.Sequential(*layers)

        # Final linear projection
        self.final_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Global features tensor of shape (Batch_Size, GLOBAL_INPUT_DIM).

        Returns:
            Tensor of shape (Batch_Size, GLOBAL_HIDDEN_DIM).
        """
        x = self.encoder(x)
        x = self.final_projection(x)
        return x


class HCCRDSModel(nn.Module):
    """
    High-Capacity Chemically-Resolved Deep Sets (HC-CRDS) Model.

    This model fuses the outputs of the AtomicStream (local chemical environment) and
    the GlobalStream (thermodynamic context) via Late Fusion to predict material properties.
    """

    def __init__(self):
        super(HCCRDSModel, self).__init__()

        self.atomic_stream = AtomicStream()
        self.global_stream = GlobalStream()

        # Calculate fusion input dimension
        # Atomic stream outputs: 2 * ATOMIC_HIDDEN_DIM (due to Mean + Max pooling)
        # Global stream outputs: GLOBAL_HIDDEN_DIM
        fusion_input_dim = (2 * Config.ATOMIC_HIDDEN_DIM) + Config.GLOBAL_HIDDEN_DIM
        fusion_hidden_dim = Config.FUSION_HIDDEN_DIM
        output_dim = Config.OUTPUT_DIM
        dropout = Config.DROPOUT_RATE

        # Fusion Head: MLP to regress targets from the fused representation
        self.fusion_head = nn.Sequential(
            WideBlock(fusion_input_dim, fusion_hidden_dim, dropout),
            WideBlock(fusion_hidden_dim, fusion_hidden_dim, dropout),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

    def forward(self, atomic_features, batch_indices, global_features):
        """
        Forward pass of the HC-CRDS model.

        Args:
            atomic_features: Tensor (N_atoms, ATOMIC_INPUT_DIM)
            batch_indices: Tensor (N_atoms,)
            global_features: Tensor (Batch_Size, GLOBAL_INPUT_DIM)

        Returns:
            output: Tensor (Batch_Size, 2) containing predicted [formation_energy, bandgap_energy] (log-transformed).
        """
        # Determine batch size from global features
        batch_size = global_features.shape[0]

        # Process Atomic Stream
        # Pass batch_size to ensure scatter operations produce correct output size
        atomic_emb = self.atomic_stream(
            atomic_features, batch_indices, num_graphs=batch_size
        )

        # Process Global Stream
        global_emb = self.global_stream(global_features)

        # Late Fusion: Concatenate embeddings
        fused = torch.cat([atomic_emb, global_emb], dim=1)

        # Final Regression
        output = self.fusion_head(fused)

        return output
