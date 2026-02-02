import torch
import torch.nn as nn
from library.config import Config


class AtomicStream(nn.Module):
    """
    Dual-Coordinate Point Processor.
    Processes atomic features (One-hot, Centered Coords, Fractional Coords, NN Dist, Potential)
    using a wide MLP and aggregates them using Dual Pooling (Mean + Max).
    """

    def __init__(self):
        super().__init__()
        input_dim = Config.ATOMIC_INPUT_DIM
        hidden_dim = Config.ATOMIC_HIDDEN_DIM
        latent_dim = Config.LATENT_DIM
        dropout = Config.DROPOUT

        # Wide MLP to prevent information bottlenecks
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),  # Activation before pooling
        )

    def forward(self, x, mask):
        """
        Args:
            x: (batch_size, max_atoms, input_dim)
            mask: (batch_size, max_atoms) - 1 for atom, 0 for padding
        Returns:
            aggregated: (batch_size, 2 * latent_dim)
        """
        # Pass through MLP
        # h shape: (batch_size, max_atoms, latent_dim)
        h = self.mlp(x)

        # Apply mask to zero out padding
        mask_expanded = mask.unsqueeze(-1)  # (batch, atoms, 1)
        h_masked = h * mask_expanded

        # --- Mean Pooling ---
        sum_h = torch.sum(h_masked, dim=1)
        # Avoid division by zero
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pool = sum_h / counts

        # --- Max Pooling ---
        # Set padded values to a very small number so they aren't selected as max
        # (unless all are padded, which shouldn't happen in valid data)
        h_for_max = h.clone()
        h_for_max[mask == 0] = -1e9
        max_pool, _ = torch.max(h_for_max, dim=1)

        # Concatenate Mean and Max pooling vectors
        return torch.cat([mean_pool, max_pool], dim=1)


class GlobalStream(nn.Module):
    """
    Thermodynamic Context Encoder.
    Processes global features (Lattice, Volume, Density, Stoichiometry, Total Atoms).
    """

    def __init__(self):
        super().__init__()
        input_dim = Config.GLOBAL_INPUT_DIM
        hidden_dim = Config.GLOBAL_HIDDEN_DIM
        latent_dim = Config.LATENT_DIM
        dropout = Config.DROPOUT

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, global_input_dim)
        Returns:
            out: (batch_size, latent_dim)
        """
        return self.mlp(x)


class MCPDSModel(nn.Module):
    """
    Multi-Coordinate Potential Deep Sets (MC-PDS).
    Fuses local atomic information with global thermodynamic context.
    """

    def __init__(self):
        super().__init__()

        self.atomic_stream = AtomicStream()
        self.global_stream = GlobalStream()

        # Fusion Input Dimension:
        # Atomic Stream outputs (Mean + Max) -> 2 * latent_dim
        # Global Stream outputs -> 1 * latent_dim
        fusion_dim = (2 * Config.LATENT_DIM) + Config.LATENT_DIM
        dropout = Config.DROPOUT

        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Targets: formation_energy, bandgap_energy
        )

    def forward(self, atomic_features, global_features, mask):
        """
        Args:
            atomic_features: (batch, max_atoms, atomic_dim)
            global_features: (batch, global_dim)
            mask: (batch, max_atoms)
        Returns:
            predictions: (batch, 2)
        """
        # Process Atomic Stream
        atomic_repr = self.atomic_stream(atomic_features, mask)

        # Process Global Stream
        global_repr = self.global_stream(global_features)

        # Late Fusion
        fused = torch.cat([atomic_repr, global_repr], dim=1)

        # Predict
        output = self.head(fused)

        return output
