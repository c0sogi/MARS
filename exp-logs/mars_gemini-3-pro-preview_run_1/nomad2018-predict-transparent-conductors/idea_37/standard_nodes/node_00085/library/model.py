import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_mean, scatter_max
except ImportError:
    # Fallback implementation if torch_scatter is not available
    def scatter_mean(src, index, dim=0, dim_size=None):
        if dim_size is None:
            dim_size = index.max().item() + 1
        out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
        ones = torch.ones(dim_size, src.size(1), device=src.device, dtype=src.dtype)
        count = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)

        out.index_add_(0, index, src)
        count.index_add_(0, index, torch.ones_like(src))

        # Avoid division by zero
        count[count == 0] = 1
        return out / count

    def scatter_max(src, index, dim=0, dim_size=None):
        # This is a naive implementation; torch_scatter is preferred for speed
        if dim_size is None:
            dim_size = index.max().item() + 1
        # Initialize with very small number
        out = torch.full(
            (dim_size, src.size(1)), -1e38, device=src.device, dtype=src.dtype
        )
        # There isn't a direct index_max_ in standard torch < 1.12 without scatter_reduce
        # We assume torch_scatter is installed as per requirements.
        # If not, this fallback might fail or be slow.
        # Using scatter_reduce if available (Torch 1.11+)
        if hasattr(torch, "scatter_reduce_"):
            out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
            out.scatter_reduce_(
                0,
                index.unsqueeze(1).expand(-1, src.size(1)),
                src,
                reduce="amax",
                include_self=False,
            )
            return out, None
        else:
            # Very slow fallback loop
            for i in range(dim_size):
                mask = index == i
                if mask.any():
                    out[i] = src[mask].max(dim=0)[0]
                else:
                    out[i] = 0.0
            return out, None


from library.config import Config


class WideMLP(nn.Module):
    """
    A Wide Multi-Layer Perceptron with Batch Normalization and Dropout.
    Used for encoding atomic and global features into high-dimensional embeddings.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(WideMLP, self).__init__()
        layers = []

        # Input block
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden blocks
        # We subtract 2 because we have an input block and an output block
        # If num_layers is 3, we have 1 hidden block in between.
        for _ in range(max(0, num_layers - 2)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output projection (Linear only, no activation/BN)
        # This projects to the embedding space for pooling/fusion
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class AtomicStream(nn.Module):
    """
    Dual-Shell Point Processor.
    Processes per-atom features and aggregates them into a crystal embedding.
    """

    def __init__(self):
        super(AtomicStream, self).__init__()
        self.encoder = WideMLP(
            input_dim=Config.ATOMIC_INPUT_DIM,
            hidden_dim=Config.ATOMIC_HIDDEN_DIM,
            output_dim=Config.ATOMIC_HIDDEN_DIM,
            num_layers=Config.ATOMIC_LAYERS,
            dropout=Config.ATOMIC_DROPOUT,
        )

    def forward(self, atom_feats, batch_indices, batch_size):
        """
        Args:
            atom_feats: (Total_Atoms, 9)
            batch_indices: (Total_Atoms,) mapping atoms to batch index
            batch_size: int
        Returns:
            aggregated_feats: (Batch_Size, 2 * ATOMIC_HIDDEN_DIM)
        """
        # 1. Immediate Expansion / Encoding
        atom_embeddings = self.encoder(atom_feats)  # (Total_Atoms, Hidden)

        # 2. Dual Pooling (Mean + Max)
        # Mean Pooling
        mean_pool = scatter_mean(
            atom_embeddings, batch_indices, dim=0, dim_size=batch_size
        )

        # Max Pooling
        max_pool, _ = scatter_max(
            atom_embeddings, batch_indices, dim=0, dim_size=batch_size
        )

        # Fix: Explicitly Handle Empty Sets in Pooling (Cite debug_lesson_9)
        # Calculate counts of atoms per graph
        ones = torch.ones(atom_feats.size(0), device=atom_feats.device)
        counts = torch.zeros(batch_size, device=atom_feats.device)
        counts.index_add_(0, batch_indices, ones)

        # Create a mask for non-empty graphs (shape: [Batch_Size, 1])
        mask = (counts > 0).float().unsqueeze(1)

        # Apply mask to max_pool to reset sentinel values (e.g., -1e38) to 0 for empty graphs
        max_pool = max_pool * mask

        # Concatenate to capture both average properties and salient features
        out = torch.cat([mean_pool, max_pool], dim=1)
        return out


class GlobalStream(nn.Module):
    """
    Thermodynamic Context Encoder.
    Processes macroscopic features of the unit cell.
    """

    def __init__(self):
        super(GlobalStream, self).__init__()
        # Using a High-Capacity MLP for global features
        # We use 2 layers to project global features to the embedding space
        self.encoder = WideMLP(
            input_dim=Config.GLOBAL_INPUT_DIM,
            hidden_dim=Config.GLOBAL_HIDDEN_DIM,
            output_dim=Config.GLOBAL_HIDDEN_DIM,
            num_layers=2,
            dropout=Config.GLOBAL_DROPOUT,
        )

    def forward(self, global_feats):
        """
        Args:
            global_feats: (Batch_Size, 12)
        Returns:
            encoded_globals: (Batch_Size, GLOBAL_HIDDEN_DIM)
        """
        return self.encoder(global_feats)


class FusionHead(nn.Module):
    """
    Late Fusion and Regression Head.
    Combines atomic and global embeddings to predict targets.
    """

    def __init__(self, input_dim):
        super(FusionHead, self).__init__()

        layers = []
        current_dim = input_dim

        # Hidden layers defined in config
        for hidden_dim in Config.FUSION_HIDDEN_DIMS:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.FUSION_DROPOUT))
            current_dim = hidden_dim

        # Final regression layer (Output Dim = 2)
        layers.append(nn.Linear(current_dim, Config.OUTPUT_DIM))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DualShellWideDeepSets(nn.Module):
    """
    Main Model Architecture: DSG-WDS.
    """

    def __init__(self):
        super(DualShellWideDeepSets, self).__init__()

        self.atomic_stream = AtomicStream()
        self.global_stream = GlobalStream()

        # Calculate fusion input dimension
        # Atomic stream outputs [Mean_Pool; Max_Pool] -> 2 * ATOMIC_HIDDEN_DIM
        # Global stream outputs [Global_Embed] -> GLOBAL_HIDDEN_DIM
        fusion_input_dim = (2 * Config.ATOMIC_HIDDEN_DIM) + Config.GLOBAL_HIDDEN_DIM

        self.fusion_head = FusionHead(fusion_input_dim)

    def forward(self, atom_feats, batch_indices, global_feats):
        """
        Forward pass.

        Args:
            atom_feats: (Total_Atoms, 9)
            batch_indices: (Total_Atoms,)
            global_feats: (Batch_Size, 12)

        Returns:
            predictions: (Batch_Size, 2)
        """
        batch_size = global_feats.size(0)

        # 1. Process Atomic Stream
        atomic_repr = self.atomic_stream(
            atom_feats, batch_indices, batch_size
        )  # (B, 1024)

        # 2. Process Global Stream
        global_repr = self.global_stream(global_feats)  # (B, 256)

        # 3. Late Fusion
        fused = torch.cat([atomic_repr, global_repr], dim=1)  # (B, 1280)

        # 4. Regress Targets
        output = self.fusion_head(fused)  # (B, 2)

        return output
