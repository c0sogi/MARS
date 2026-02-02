import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from library.config import Config


class ResidualBlock(nn.Module):
    """
    A residual block with the structure:
    Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> BatchNorm -> Add Input -> ReLU
    """

    def __init__(self, dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class AtomicStream(nn.Module):
    """
    Deep Residual Point Processor for atomic features.
    Projects atomic features, processes them through residual blocks,
    and aggregates them using Tri-Pooling (Mean, Max, Std).
    """

    def __init__(self, input_dim, latent_dim, num_blocks, dropout_rate):
        super(AtomicStream, self).__init__()

        # Initial projection
        self.input_proj = nn.Linear(input_dim, latent_dim)

        # Residual Backbone
        self.blocks = nn.ModuleList(
            [ResidualBlock(latent_dim, dropout_rate) for _ in range(num_blocks)]
        )

        # Final projection before pooling (no activation as per description)
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, atom_x, batch_indices, num_graphs=None):
        # atom_x: (Total_Atoms, Atomic_Feature_Dim)
        # batch_indices: (Total_Atoms,)

        # Handle empty atom_x case
        if atom_x.size(0) == 0:
            if num_graphs is None:
                num_graphs = 0
            return torch.zeros(
                (num_graphs, 3 * self.output_proj.out_features), device=atom_x.device
            )

        # Project and Process
        x = self.input_proj(atom_x)
        for block in self.blocks:
            x = block(x)

        x = self.output_proj(x)

        # Tri-Pooling Aggregation
        # Pass dim_size to handle empty graphs in the batch (Cite debug_lesson_10)

        # 1. Mean Pooling
        mean_pool = scatter(x, batch_indices, dim=0, dim_size=num_graphs, reduce="mean")

        # 2. Max Pooling
        max_pool = scatter(x, batch_indices, dim=0, dim_size=num_graphs, reduce="max")
        # Fix init values for empty graphs
        max_pool[max_pool < -1e9] = 0.0

        # 3. Standard Deviation Pooling
        # std = sqrt(E[x^2] - (E[x])^2)
        # We clamp variance to avoid negative values due to numerical errors
        mean_sq_pool = scatter(
            x**2, batch_indices, dim=0, dim_size=num_graphs, reduce="mean"
        )
        var = mean_sq_pool - mean_pool**2
        std_pool = torch.sqrt(torch.clamp(var, min=1e-6))

        # Concatenate: (Batch_Size, 3 * Latent_Dim)
        out = torch.cat([mean_pool, max_pool, std_pool], dim=1)
        return out


class GlobalStream(nn.Module):
    """
    High-Capacity MLP for global macroscopic features.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(GlobalStream, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, global_x):
        return self.net(global_x)


class RTDSModel(nn.Module):
    """
    Residual Tri-Pool Deep Sets (RT-DS) Model.
    Fuses atomic and global streams to predict material properties.
    """

    def __init__(self):
        super(RTDSModel, self).__init__()

        # Atomic Stream
        self.atomic_stream = AtomicStream(
            input_dim=Config.ATOMIC_FEATURE_DIM,
            latent_dim=Config.LATENT_DIM,
            num_blocks=Config.NUM_RES_BLOCKS,
            dropout_rate=Config.DROPOUT_RATE,
        )

        # Global Stream
        self.global_stream = GlobalStream(
            input_dim=Config.GLOBAL_FEATURE_DIM,
            hidden_dim=Config.GLOBAL_HIDDEN_DIM,
            dropout_rate=Config.DROPOUT_RATE,
        )

        # Fusion Head
        # Atomic output is 3 * LATENT_DIM (Mean + Max + Std)
        # Global output is GLOBAL_HIDDEN_DIM
        fusion_dim = (3 * Config.LATENT_DIM) + Config.GLOBAL_HIDDEN_DIM

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, Config.NUM_TARGETS),
        )

    def forward(self, atom_x, glob_x, batch_indices):
        """
        Forward pass.

        Args:
            atom_x: (Total_Atoms, Atomic_Feature_Dim)
            glob_x: (Batch_Size, Global_Feature_Dim)
            batch_indices: (Total_Atoms,) mapping atoms to batch index

        Returns:
            (Batch_Size, Num_Targets) predicted values
        """
        # Determine batch size from global features
        num_graphs = glob_x.size(0)

        # Process streams
        atomic_emb = self.atomic_stream(atom_x, batch_indices, num_graphs=num_graphs)
        global_emb = self.global_stream(glob_x)

        # Fusion
        fused = torch.cat([atomic_emb, global_emb], dim=1)

        # Prediction
        out = self.regressor(fused)
        return out
