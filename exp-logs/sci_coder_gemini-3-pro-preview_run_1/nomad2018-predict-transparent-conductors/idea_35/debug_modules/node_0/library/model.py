import torch
import torch.nn as nn
from library.config import Config


class AtomicStream(nn.Module):
    """
    Processes local atomic features using a wide MLP and aggregates them
    using Dual Pooling (Mean + Max).
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(AtomicStream, self).__init__()

        # Wide MLP Encoder
        # Projects atomic features into a high-dimensional latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Final projection to embedding space (linear, no activation)
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask):
        """
        Args:
            x: (Batch, Max_Atoms, Input_Dim) - Atomic features
            mask: (Batch, Max_Atoms) - Boolean mask (True for real atoms)
        Returns:
            aggregated: (Batch, 2 * Hidden_Dim) - Concatenated Mean and Max pooled embeddings
        """
        B, N, D = x.size()

        # Flatten to (Batch * Max_Atoms, Input_Dim) for batch processing
        x_flat = x.view(B * N, D)

        # Encode features
        h_flat = self.encoder(x_flat)
        h_flat = self.projection(h_flat)

        # Reshape back to (Batch, Max_Atoms, Hidden_Dim)
        h = h_flat.view(B, N, -1)

        # Expand mask for element-wise operations: (Batch, Max_Atoms, Hidden_Dim)
        mask_expanded = mask.unsqueeze(-1).expand_as(h)

        # Zero out padding for Mean Pooling
        h_masked = h * mask_expanded.float()

        # Mean Pooling
        # Calculate lengths (number of real atoms per sample)
        lengths = mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        lengths = torch.clamp(lengths, min=1.0)  # Avoid division by zero
        mean_pool = h_masked.sum(dim=1) / lengths  # (B, Hidden_Dim)

        # Max Pooling
        # Set padding elements to a very small number so they don't affect max
        h_max = h.clone()
        h_max[~mask_expanded] = -1e9
        max_pool = h_max.max(dim=1)[0]  # (B, Hidden_Dim)

        # Concatenate pooling results
        return torch.cat([mean_pool, max_pool], dim=1)


class GlobalStream(nn.Module):
    """
    Processes global macroscopic features (Lattice, Stoichiometry, APF)
    using a high-capacity MLP.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(GlobalStream, self).__init__()

        self.mlp = nn.Sequential(
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
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Input_Dim) - Global features
        Returns:
            h: (Batch, Hidden_Dim) - Global embedding
        """
        return self.mlp(x)


class MNPADSModel(nn.Module):
    """
    Multi-Neighbor Packing-Aware Deep Sets (MNPA-DS) Model.
    Combines AtomicStream and GlobalStream via Late Fusion.
    """

    def __init__(self, config=Config):
        super(MNPADSModel, self).__init__()

        # Atomic Stream for local structure
        self.atomic_stream = AtomicStream(
            input_dim=config.ATOMIC_FEATURE_DIM,
            hidden_dim=config.ATOMIC_HIDDEN_DIM,
            dropout_rate=config.DROPOUT_RATE,
        )

        # Global Stream for macroscopic context and physics priors (APF)
        self.global_stream = GlobalStream(
            input_dim=config.GLOBAL_FEATURE_DIM,
            hidden_dim=config.GLOBAL_HIDDEN_DIM,
            dropout_rate=config.DROPOUT_RATE,
        )

        # Fusion Head
        # Input dimension is sum of Atomic output (2 * Atomic_Hidden) and Global output (Global_Hidden)
        fusion_input_dim = (2 * config.ATOMIC_HIDDEN_DIM) + config.GLOBAL_HIDDEN_DIM

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, config.FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.FUSION_HIDDEN_DIM, config.FUSION_HIDDEN_DIM // 2),
            nn.BatchNorm1d(config.FUSION_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.FUSION_HIDDEN_DIM // 2, 2),  # Output: 2 targets
        )

    def forward(self, atomic_features, global_features, mask):
        """
        Args:
            atomic_features: (Batch, Max_Atoms, 10)
            global_features: (Batch, 13)
            mask: (Batch, Max_Atoms)
        Returns:
            predictions: (Batch, 2)
        """
        # Process Atomic Stream -> (B, 2 * Atomic_Hidden)
        atomic_embedding = self.atomic_stream(atomic_features, mask)

        # Process Global Stream -> (B, Global_Hidden)
        global_embedding = self.global_stream(global_features)

        # Late Fusion -> (B, Fusion_Input_Dim)
        combined = torch.cat([atomic_embedding, global_embedding], dim=1)

        # Prediction -> (B, 2)
        output = self.fusion_head(combined)

        return output
