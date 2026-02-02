import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    ATOMIC_INPUT_DIM,
    ATOMIC_HIDDEN_DIM,
    ATOMIC_LAYERS,
    ATOMIC_DROPOUT,
    GLOBAL_INPUT_DIM,
    GLOBAL_HIDDEN_DIM,
    GLOBAL_DROPOUT,
    FUSION_INPUT_DIM,
    FUSION_HIDDEN_DIM,
    FUSION_DROPOUT,
    OUTPUT_DIM,
    NUM_ATOM_TYPES,
)


class AtomicEncoder(nn.Module):
    """
    Encodes per-atom features into a latent representation using a Wide MLP.
    Applies Batch Normalization and Dropout for regularization.
    """

    def __init__(self):
        super().__init__()
        layers = []

        # Layer 1: Input -> Hidden
        layers.append(nn.Linear(ATOMIC_INPUT_DIM, ATOMIC_HIDDEN_DIM))
        layers.append(nn.BatchNorm1d(ATOMIC_HIDDEN_DIM))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(ATOMIC_DROPOUT))

        # Layer 2: Hidden -> Hidden (Deepening the point processor)
        # We use ATOMIC_LAYERS from config to determine depth, but here we explicitly
        # implement the structure described in the idea (3 layers total).
        layers.append(nn.Linear(ATOMIC_HIDDEN_DIM, ATOMIC_HIDDEN_DIM))
        layers.append(nn.BatchNorm1d(ATOMIC_HIDDEN_DIM))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(ATOMIC_DROPOUT))

        # Layer 3: Projection to latent space
        # As per design, the final projection does not have activation to allow
        # the full vector space to be utilized during pooling.
        layers.append(nn.Linear(ATOMIC_HIDDEN_DIM, ATOMIC_HIDDEN_DIM))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Input x: (Batch, Max_Atoms, Features)
        b, n, f = x.shape

        # Flatten to (Batch * Max_Atoms, Features) for BatchNorm1d
        x_flat = x.view(-1, f)

        # Pass through MLP
        out_flat = self.net(x_flat)

        # Reshape back to (Batch, Max_Atoms, Hidden)
        return out_flat.view(b, n, -1)


class GlobalEncoder(nn.Module):
    """
    Encodes global crystal features (lattice, stoichiometry, density) into a context vector.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GLOBAL_INPUT_DIM, GLOBAL_HIDDEN_DIM),
            nn.BatchNorm1d(GLOBAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(GLOBAL_DROPOUT),
            # Second layer to increase capacity for thermodynamic context
            nn.Linear(GLOBAL_HIDDEN_DIM, GLOBAL_HIDDEN_DIM),
            nn.BatchNorm1d(GLOBAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(GLOBAL_DROPOUT),
        )

    def forward(self, x):
        # Input x: (Batch, Global_Features)
        return self.net(x)


class EWADeepSets(nn.Module):
    """
    Element-Wise Aggregated Deep Sets.

    Hierarchical Aggregation Strategy:
    1. Process atoms individually.
    2. Global Pooling: Mean and Max over all atoms.
    3. Per-Element Pooling: Mean over specific element types (Al, Ga, In, O).
    4. Fuse with Global Context.
    5. Regress targets.
    """

    def __init__(self):
        super().__init__()
        self.atomic_encoder = AtomicEncoder()
        self.global_encoder = GlobalEncoder()

        # Fusion Head
        # Inputs: Global Mean (512) + Global Max (512) + 4 * Element Mean (2048) + Global Context (256)
        # Total Input Dim: 3328 (Matches FUSION_INPUT_DIM in config)
        self.fusion_head = nn.Sequential(
            nn.Linear(FUSION_INPUT_DIM, FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(FUSION_DROPOUT),
            nn.Linear(FUSION_HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, atomic_features, mask, global_features):
        """
        Args:
            atomic_features: Tensor (B, N, 11) - Per-atom input features.
            mask: Tensor (B, N) - Boolean mask (True for real atoms, False for padding).
            global_features: Tensor (B, 12) - Global crystal features.

        Returns:
            output: Tensor (B, 2) - Predicted Formation Energy and Bandgap.
        """
        # 1. Encode Atomic Features
        # h_atoms: (B, N, 512)
        h_atoms = self.atomic_encoder(atomic_features)

        # Apply mask to zero out padding contributions
        mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
        h_atoms = h_atoms * mask_expanded

        # 2. Global Pooling
        # Count of valid atoms per batch
        atom_counts = mask_expanded.sum(dim=1)  # (B, 1)
        atom_counts = torch.clamp(atom_counts, min=1.0)  # Prevent div by zero

        # Global Mean Pooling
        global_mean = h_atoms.sum(dim=1) / atom_counts  # (B, 512)

        # Global Max Pooling
        # Set padded values to a very small number so they don't affect max
        h_atoms_for_max = h_atoms.clone()
        h_atoms_for_max[~mask] = -1e9
        global_max = h_atoms_for_max.max(dim=1)[0]  # (B, 512)

        # 3. Per-Element Pooling
        # The first 4 columns of atomic_features are one-hot encodings for [Al, Ga, In, O]
        element_means = []
        for i in range(NUM_ATOM_TYPES):
            # Identify atoms of type i
            # atomic_features is float, so we check > 0.5
            is_element = atomic_features[:, :, i] > 0.5  # (B, N)

            # Combine with validity mask
            element_mask = mask & is_element  # (B, N)
            element_mask_expanded = element_mask.unsqueeze(-1).float()  # (B, N, 1)

            # Sum latent vectors for this element
            elem_sum = (h_atoms * element_mask_expanded).sum(dim=1)  # (B, 512)

            # Count atoms of this element
            elem_count = element_mask_expanded.sum(dim=1)  # (B, 1)

            # Compute Mean (add epsilon to avoid div by zero if element is absent)
            elem_mean = elem_sum / (elem_count + 1e-6)  # (B, 512)

            element_means.append(elem_mean)

        # Concatenate all element means -> (B, 4 * 512) = (B, 2048)
        element_pool = torch.cat(element_means, dim=1)

        # 4. Encode Global Features
        h_global = self.global_encoder(global_features)  # (B, 256)

        # 5. Feature Fusion
        # Concatenate: Global Mean, Global Max, Element Means, Global Context
        fused_features = torch.cat(
            [global_mean, global_max, element_pool, h_global], dim=1
        )  # (B, 3328)

        # 6. Final Regression
        output = self.fusion_head(fused_features)  # (B, 2)

        return output
