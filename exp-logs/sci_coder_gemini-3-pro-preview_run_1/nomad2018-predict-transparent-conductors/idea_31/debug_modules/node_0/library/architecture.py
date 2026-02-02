import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from library.config import MODEL_PARAMS


class WideBlock(nn.Module):
    """
    A reusable block consisting of Linear -> BatchNorm -> ReLU -> Dropout.
    Implements the 'Wide MLP' layer structure with regularization safeguards.
    """

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.linear(x))))


class LCEWDS(nn.Module):
    """
    Local-Chemical-Environment Wide Deep Sets (LCE-WDS) architecture.

    This model processes atomic features (with local chemical environment context)
    and global crystal features in parallel streams, fuses them via dual pooling,
    and predicts formation and bandgap energies.
    """

    def __init__(self, params=MODEL_PARAMS):
        super().__init__()

        # Extract hyperparameters
        atomic_in = params["atomic_input_dim"]
        global_in = params["global_input_dim"]
        atomic_hidden = params["atomic_hidden_dim"]
        global_hidden = params["global_hidden_dim"]
        fusion_hidden = params["fusion_hidden_dim"]
        output_dim = params["output_dim"]
        dropout = params["dropout"]

        # ---------------------------------------------------------
        # Atomic Stream (Chemically-Contextualized Processor)
        # ---------------------------------------------------------
        # Encoder: Wide MLP to project features into latent space.
        # Ends with a Linear projection (no activation) to the embedding space.
        self.atomic_encoder = nn.Sequential(
            WideBlock(atomic_in, atomic_hidden, dropout),
            WideBlock(atomic_hidden, atomic_hidden, dropout),
            WideBlock(atomic_hidden, atomic_hidden, dropout),
            nn.Linear(atomic_hidden, fusion_hidden),
        )

        # ---------------------------------------------------------
        # Global Stream (Thermodynamic Context)
        # ---------------------------------------------------------
        # Encoder: High-Capacity MLP for macroscopic features.
        self.global_encoder = nn.Sequential(
            WideBlock(global_in, global_hidden, dropout),
            WideBlock(global_hidden, global_hidden, dropout),
            nn.Linear(global_hidden, fusion_hidden),
        )

        # ---------------------------------------------------------
        # Fusion Head
        # ---------------------------------------------------------
        # Concatenate: [Atomic_Mean, Atomic_Max, Global_Emb]
        # Dimension = fusion_hidden + fusion_hidden + fusion_hidden
        fusion_in_dim = fusion_hidden * 3

        self.regressor = nn.Sequential(
            WideBlock(fusion_in_dim, fusion_hidden, dropout),
            WideBlock(fusion_hidden, fusion_hidden, dropout),
            nn.Linear(fusion_hidden, output_dim),
        )

    def forward(self, atomic_features, global_features, batch_indices):
        """
        Forward pass of the LCE-WDS model.

        Args:
            atomic_features (Tensor): (N_atoms, atomic_in) - Dense atomic features
            global_features (Tensor): (batch_size, global_in) - Dense global features
            batch_indices (Tensor): (N_atoms,) - Index mapping each atom to its sample in the batch

        Returns:
            Tensor: (batch_size, output_dim) - Predicted energies
        """
        # 1. Process Atomic Stream
        # Project atoms to latent space
        atom_emb = self.atomic_encoder(atomic_features)  # (N_atoms, fusion_hidden)

        # Aggregation (Dual Pooling)
        # We need the batch size to ensure scatter operations produce the correct output shape
        batch_size = global_features.shape[0]

        # Global Mean Pooling
        pooled_mean = scatter_mean(atom_emb, batch_indices, dim=0, dim_size=batch_size)

        # Global Max Pooling (scatter_max returns values, indices)
        pooled_max, _ = scatter_max(atom_emb, batch_indices, dim=0, dim_size=batch_size)

        # 2. Process Global Stream
        global_emb = self.global_encoder(global_features)  # (batch_size, fusion_hidden)

        # 3. Late Fusion
        # Concatenate aggregated atomic context with global thermodynamic context
        combined = torch.cat([pooled_mean, pooled_max, global_emb], dim=1)

        # 4. Regression
        output = self.regressor(combined)

        return output
