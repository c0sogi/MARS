import torch
import torch.nn as nn

try:
    from torch_scatter import scatter_mean, scatter_max
except ImportError:
    pass
from library.config import Config


class WideBlock(nn.Module):
    """
    A robust building block for wide MLPs with regularization.
    Structure: Linear -> BatchNorm -> ReLU -> Dropout
    """

    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super(WideBlock, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
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
    Processes per-atom features (Chemical Contextualized Point Processor).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super(AtomicStream, self).__init__()
        # Wide MLP Encoder
        self.net = nn.Sequential(
            WideBlock(input_dim, hidden_dim, dropout_rate),
            WideBlock(hidden_dim, hidden_dim, dropout_rate),
            WideBlock(hidden_dim, hidden_dim, dropout_rate),
        )
        # Projection to embedding space (Linear, no activation)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, batch_indices, num_graphs):
        """
        Args:
            x: Atomic features (N_atoms, input_dim)
            batch_indices: Batch index for each atom (N_atoms,)
            num_graphs: Number of graphs in the batch (for pooling)
        Returns:
            Aggregated atomic embedding (Batch_Size, 2 * output_dim)
        """
        # Encode
        x = self.net(x)
        x = self.projection(x)

        # Dual Pooling: Mean + Max
        # scatter functions handle variable graph sizes efficiently
        mean_pool = scatter_mean(x, batch_indices, dim=0, dim_size=num_graphs)
        max_pool, _ = scatter_max(x, batch_indices, dim=0, dim_size=num_graphs)

        # Concatenate aggregated features
        out = torch.cat([mean_pool, max_pool], dim=1)
        return out


class GlobalStream(nn.Module):
    """
    Processes macroscopic global features (Thermodynamic Context).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super(GlobalStream, self).__init__()
        # High-Capacity MLP
        self.net = nn.Sequential(
            WideBlock(input_dim, hidden_dim, dropout_rate),
            WideBlock(hidden_dim, hidden_dim, dropout_rate),
        )
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: Global features (Batch_Size, input_dim)
        Returns:
            Global embedding (Batch_Size, output_dim)
        """
        x = self.net(x)
        x = self.projection(x)
        return x


class LSA_WDS(nn.Module):
    """
    Local-Stoichiometry Augmented Wide Deep Sets.
    Combines atomic and global streams via Late Fusion.
    """

    def __init__(self):
        super(LSA_WDS, self).__init__()

        # Dimensions from Config
        atom_in = Config.ATOMIC_FEATURE_DIM
        atom_hidden = Config.ATOM_HIDDEN_DIM

        global_in = Config.GLOBAL_FEATURE_DIM
        global_hidden = Config.GLOBAL_HIDDEN_DIM

        fusion_dim = Config.FUSION_HIDDEN_DIM
        dropout = Config.DROPOUT_RATE

        # Streams
        self.atomic_stream = AtomicStream(atom_in, atom_hidden, fusion_dim, dropout)
        self.global_stream = GlobalStream(global_in, global_hidden, fusion_dim, dropout)

        # Fusion Head
        # Atomic stream outputs 2 * fusion_dim (Mean + Max)
        # Global stream outputs 1 * fusion_dim
        fusion_input_dim = (2 * fusion_dim) + fusion_dim

        self.fusion_head = nn.Sequential(
            WideBlock(fusion_input_dim, fusion_dim, dropout),
            WideBlock(fusion_dim, fusion_dim // 2, dropout),
            nn.Linear(fusion_dim // 2, 2),  # Output: formation_energy, bandgap_energy
        )

    def forward(self, atomic_features, batch_indices, global_features):
        """
        Args:
            atomic_features: Tensor (N_atoms, D_atom)
            batch_indices: Tensor (N_atoms,)
            global_features: Tensor (Batch_Size, D_global)
        """
        batch_size = global_features.size(0)

        # Process Atomic Stream
        atomic_emb = self.atomic_stream(atomic_features, batch_indices, batch_size)

        # Process Global Stream
        global_emb = self.global_stream(global_features)

        # Late Fusion
        combined = torch.cat([atomic_emb, global_emb], dim=1)

        # Regression
        output = self.fusion_head(combined)

        return output
