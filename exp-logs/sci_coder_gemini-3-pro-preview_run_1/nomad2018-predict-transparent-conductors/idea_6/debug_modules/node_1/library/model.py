import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max


class PADSDS(nn.Module):
    """
    Physically-Augmented Dual-Stream Deep Sets (PA-DSDS) model.

    This model processes crystal structures using two parallel streams:
    1. Atomic Stream: Processes local atomic features and aggregates them using
       dual pooling (Mean + Max).
    2. Global Stream: Processes macroscopic crystal properties.

    The streams are fused via concatenation and passed through a regressor to
    predict target properties.
    """

    def __init__(
        self, atomic_input_dim, global_input_dim, hidden_dim=512, latent_dim=512
    ):
        """
        Args:
            atomic_input_dim (int): Dimension of atom-wise input features.
            global_input_dim (int): Dimension of global crystal features.
            hidden_dim (int): Hidden dimension for the atomic encoder MLPs.
            latent_dim (int): Dimension of the latent embeddings before fusion.
        """
        super(PADSDS, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Atomic Stream Encoder
        # Wide MLP to project rich atomic features into latent space.
        # ---------------------------------------------------------------------
        self.atomic_encoder = nn.Sequential(
            nn.Linear(atomic_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # Linear projection to latent space (no activation)
            nn.Linear(hidden_dim, latent_dim),
        )

        # ---------------------------------------------------------------------
        # 2. Global Stream Encoder
        # High-capacity MLP for macroscopic thermodynamic context.
        # ---------------------------------------------------------------------
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------------------
        # 3. Fusion Head
        # Concatenates Mean Pool, Max Pool, and Global Embedding.
        # Input dim = latent_dim (Mean) + latent_dim (Max) + latent_dim (Global)
        # ---------------------------------------------------------------------
        fusion_input_dim = latent_dim * 3

        self.regressor = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            # Output: formation_energy_ev_natom, bandgap_energy_ev
            nn.Linear(256, 2),
        )

    def forward(self, atomic_feats, batch_indices, global_feats):
        """
        Forward pass of the PA-DSDS model.

        Args:
            atomic_feats (Tensor): Atom-wise features [Total_Atoms, atomic_input_dim].
            batch_indices (LongTensor): Batch index for each atom [Total_Atoms].
            global_feats (Tensor): Global crystal features [Batch_Size, global_input_dim].

        Returns:
            Tensor: Predictions [Batch_Size, 2].
        """
        # Determine batch size from global features
        batch_size = global_feats.size(0)

        # --- Atomic Stream ---
        # Project atoms to latent space
        atom_embeddings = self.atomic_encoder(atomic_feats)

        # --- Aggregation (Dual Pooling) ---
        # Mean Pooling: Captures average properties
        pooled_mean = scatter_mean(
            atom_embeddings, batch_indices, dim=0, dim_size=batch_size
        )

        # Max Pooling: Captures salient structural anomalies/features
        # scatter_max returns (values, indices)
        pooled_max, _ = scatter_max(
            atom_embeddings, batch_indices, dim=0, dim_size=batch_size
        )

        # Handle potential -inf from scatter_max on empty graphs (though unlikely in this dataset)
        pooled_max[pooled_max == -float("inf")] = 0.0

        # --- Global Stream ---
        global_embeddings = self.global_encoder(global_feats)

        # --- Fusion ---
        # Concatenate all representations
        combined = torch.cat([pooled_mean, pooled_max, global_embeddings], dim=1)

        # --- Prediction ---
        output = self.regressor(combined)

        return output
