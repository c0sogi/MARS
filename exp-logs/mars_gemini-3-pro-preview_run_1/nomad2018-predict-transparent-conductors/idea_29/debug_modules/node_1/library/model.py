import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    ATOMIC_HIDDEN_DIM,
    ATOMIC_LAYERS,
    GLOBAL_HIDDEN_DIM,
    GLOBAL_LAYERS,
    FUSION_HIDDEN_DIM,
    DROPOUT,
)


class AtomicStream(nn.Module):
    """
    Robust Point Processor for the Atomic Stream.
    Processes per-atom features (One-hot, Centered Coords, NN Dist) using a Wide MLP.
    Aggregates features using Dual Pooling (Global Mean + Global Max).
    """

    def __init__(
        self,
        input_dim=8,
        hidden_dim=ATOMIC_HIDDEN_DIM,
        num_layers=ATOMIC_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        layers = []
        # Immediate Expansion: Project input to high-dimensional latent space
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Deep processing layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)

        # Final linear projection before pooling
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask):
        """
        Args:
            x: (Batch, Max_Atoms, Input_Dim) - Atomic features
            mask: (Batch, Max_Atoms) - Boolean mask (True for real atoms, False for padding)
        Returns:
            pooled_features: (Batch, 2 * Hidden_Dim) - Concatenated Mean and Max pooled features
        """
        B, N, C = x.shape

        # Flatten batch and atoms dimensions for MLP processing: (B*N, C)
        x_flat = x.view(-1, C)

        # Pass through encoder
        # BatchNorm1d works on (N_samples, C_features), so flattening is appropriate
        h_flat = self.encoder(x_flat)

        # Project to final embedding space
        h_flat = self.projection(h_flat)

        # Reshape back to (B, N, Hidden_Dim)
        h = h_flat.view(B, N, self.hidden_dim)

        # Apply mask to zero out padding (essential for correct Mean Pooling)
        # Expand mask to (B, N, Hidden_Dim)
        mask_expanded = mask.unsqueeze(-1).expand_as(h)
        h_masked = h * mask_expanded.float()

        # --- Dual Pooling ---

        # 1. Global Mean Pooling
        # Sum valid atoms along atom dimension
        sum_pooled = torch.sum(h_masked, dim=1)  # (B, Hidden_Dim)
        # Count valid atoms to compute mean
        counts = mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        counts = torch.clamp(counts, min=1.0)  # Prevent division by zero
        mean_pooled = sum_pooled / counts

        # 2. Global Max Pooling
        # We need to handle padding for Max pooling.
        # Zeros from masking might interfere if actual features are negative.
        # Set padded positions to a very small number.
        h_max_input = h.clone()
        h_max_input[~mask] = -1e9  # Effectively -infinity
        max_pooled = torch.max(h_max_input, dim=1)[0]  # (B, Hidden_Dim)

        # Concatenate pooled representations
        output = torch.cat([mean_pooled, max_pooled], dim=1)  # (B, 2*Hidden_Dim)

        return output


class GlobalStream(nn.Module):
    """
    Physics-Enhanced Context Encoder.
    Processes crystal-level features (Lattice, Stoichiometry, Weighted Physical Props).
    """

    def __init__(
        self,
        input_dim=15,
        hidden_dim=GLOBAL_HIDDEN_DIM,
        num_layers=GLOBAL_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()

        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Subsequent layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (Batch, Input_Dim) - Global features
        Returns:
            embedding: (Batch, Hidden_Dim)
        """
        return self.mlp(x)


class PIGWDS(nn.Module):
    """
    Physics-Informed Global Wide Deep Sets.
    Fuses local atomic embeddings with global physical context for energy prediction.
    """

    def __init__(
        self,
        atomic_input_dim=8,
        global_input_dim=15,
        atomic_hidden_dim=ATOMIC_HIDDEN_DIM,
        global_hidden_dim=GLOBAL_HIDDEN_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        output_dim=2,
        dropout=DROPOUT,
    ):
        super().__init__()

        # Stream Encoders
        self.atomic_stream = AtomicStream(
            input_dim=atomic_input_dim, hidden_dim=atomic_hidden_dim, dropout=dropout
        )
        self.global_stream = GlobalStream(
            input_dim=global_input_dim, hidden_dim=global_hidden_dim, dropout=dropout
        )

        # Fusion Head
        # Input: [Atomic_Mean (H_a), Atomic_Max (H_a), Global (H_g)]
        fusion_input_dim = (2 * atomic_hidden_dim) + global_hidden_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim // 2, output_dim),
        )

    def forward(self, atomic_features, global_features, mask):
        """
        Forward pass.

        Args:
            atomic_features: (B, N, 8)
            global_features: (B, 15)
            mask: (B, N)

        Returns:
            prediction: (B, 2) - [Formation Energy, Bandgap Energy]
        """
        # 1. Process Atomic Stream -> (B, 2*Atomic_Hidden)
        atomic_emb = self.atomic_stream(atomic_features, mask)

        # 2. Process Global Stream -> (B, Global_Hidden)
        global_emb = self.global_stream(global_features)

        # 3. Late Fusion
        combined = torch.cat([atomic_emb, global_emb], dim=1)

        # 4. Regression
        prediction = self.fusion_head(combined)

        return prediction
