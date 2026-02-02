import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Residual Block: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> Add -> ReLU
    Used for deep feature extraction in the atomic stream.
    """

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        # Projection for skip connection if dimensions change
        self.project = None
        if in_dim != out_dim:
            self.project = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.bn2(out)

        if self.project is not None:
            identity = self.project(identity)

        out += identity
        out = self.relu(out)
        return out


class SIRDS_SP(nn.Module):
    """
    Symmetry-Informed Residual Deep Sets with Statistical Pooling (SI-RDS-SP).

    This model processes materials through three streams:
    1. Atomic Stream: Residual processing of local atomic environments with statistical aggregation.
    2. Global Stream: Processing of macroscopic lattice and stoichiometry features.
    3. Symmetry Stream: Embedding of crystallographic spacegroups.
    """

    def __init__(self):
        super().__init__()

        # ---------------------------------------------------------------------
        # 1. Atomic Stream (Residual Point Processor)
        # ---------------------------------------------------------------------
        # Initial projection from input features to hidden dim
        self.atom_input_proj = nn.Linear(
            Config.ATOMIC_INPUT_DIM, Config.ATOMIC_HIDDEN_DIM
        )

        # Stack of Residual Blocks
        self.atom_res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    Config.ATOMIC_HIDDEN_DIM,
                    Config.ATOMIC_HIDDEN_DIM,
                    dropout=Config.ATOMIC_DROPOUT,
                )
                for _ in range(Config.ATOMIC_RES_BLOCKS)
            ]
        )

        # ---------------------------------------------------------------------
        # 2. Global Stream (Thermodynamic Context)
        # ---------------------------------------------------------------------
        # MLP for global features
        global_layers = []
        in_dim = Config.GLOBAL_INPUT_DIM
        for _ in range(Config.GLOBAL_LAYERS):
            global_layers.append(nn.Linear(in_dim, Config.GLOBAL_HIDDEN_DIM))
            global_layers.append(nn.BatchNorm1d(Config.GLOBAL_HIDDEN_DIM))
            global_layers.append(nn.ReLU())
            global_layers.append(nn.Dropout(Config.GLOBAL_DROPOUT))
            in_dim = Config.GLOBAL_HIDDEN_DIM
        self.global_encoder = nn.Sequential(*global_layers)

        # ---------------------------------------------------------------------
        # 3. Symmetry Stream (Crystallographic Prior)
        # ---------------------------------------------------------------------
        # Embedding layer for spacegroups (1-230)
        # Add 1 to MAX_SPACEGROUP to handle 1-based indexing (0 is unused or padding)
        self.symmetry_embed = nn.Embedding(
            Config.MAX_SPACEGROUP + 1, Config.SYMMETRY_EMBEDDING_DIM
        )
        self.symmetry_proj = nn.Linear(
            Config.SYMMETRY_EMBEDDING_DIM, Config.SYMMETRY_EMBEDDING_DIM
        )

        # ---------------------------------------------------------------------
        # 4. Fusion Head
        # ---------------------------------------------------------------------
        # TriPooling outputs 3 * ATOMIC_HIDDEN_DIM (Mean, Max, Std)
        # Concatenated with Global and Symmetry embeddings
        fusion_in_dim = (
            (Config.ATOMIC_HIDDEN_DIM * 3)
            + Config.GLOBAL_HIDDEN_DIM
            + Config.SYMMETRY_EMBEDDING_DIM
        )

        fusion_layers = []
        curr_dim = fusion_in_dim
        for _ in range(Config.FUSION_LAYERS):
            fusion_layers.append(nn.Linear(curr_dim, Config.FUSION_HIDDEN_DIM))
            fusion_layers.append(nn.BatchNorm1d(Config.FUSION_HIDDEN_DIM))
            fusion_layers.append(nn.ReLU())
            fusion_layers.append(nn.Dropout(Config.FUSION_DROPOUT))
            curr_dim = Config.FUSION_HIDDEN_DIM

        self.fusion_encoder = nn.Sequential(*fusion_layers)

        # Final regressor for 2 targets (Formation Energy, Bandgap Energy)
        self.regressor = nn.Linear(Config.FUSION_HIDDEN_DIM, 2)

    def tri_pooling(self, atom_emb, batch_indices, num_graphs):
        """
        Aggregates atomic embeddings using Mean, Max, and Std Dev pooling.

        Args:
            atom_emb (Tensor): Atomic embeddings (Sum_N, Hidden).
            batch_indices (Tensor): Batch index for each atom (Sum_N,).
            num_graphs (int): Number of graphs in the batch.

        Returns:
            Tensor: Aggregated features (B, Hidden * 3).
        """
        # 1. Mean Pooling
        mean_pool = scatter_mean(atom_emb, batch_indices, dim=0, dim_size=num_graphs)

        # 2. Max Pooling
        # scatter_max returns (values, indices), we only need values
        max_pool, _ = scatter_max(atom_emb, batch_indices, dim=0, dim_size=num_graphs)

        # 3. Std Dev Pooling
        # std = sqrt( E[x^2] - (E[x])^2 )
        # E[x] is mean_pool
        # E[x^2]
        mean_sq_pool = scatter_mean(
            atom_emb**2, batch_indices, dim=0, dim_size=num_graphs
        )
        # Variance calculation with epsilon for stability
        var = mean_sq_pool - mean_pool**2
        var = torch.clamp(var, min=1e-6)  # Ensure non-negative
        std_pool = torch.sqrt(var)

        # Concatenate
        return torch.cat([mean_pool, max_pool, std_pool], dim=1)

    def forward(self, atom_features, batch_indices, global_features, spacegroups):
        """
        Args:
            atom_features: (Sum_N, ATOMIC_INPUT_DIM)
            batch_indices: (Sum_N,)
            global_features: (B, GLOBAL_INPUT_DIM)
            spacegroups: (B,)

        Returns:
            out: (B, 2) - Predicted Formation Energy and Bandgap Energy
        """
        # Determine batch size from global features
        batch_size = global_features.size(0)

        # --- 1. Atomic Stream ---
        # Initial projection
        x_atom = self.atom_input_proj(atom_features)
        x_atom = F.relu(x_atom)

        # Residual Blocks
        for block in self.atom_res_blocks:
            x_atom = block(x_atom)

        # Tri-Pooling
        x_atom_pooled = self.tri_pooling(x_atom, batch_indices, batch_size)

        # --- 2. Global Stream ---
        x_global = self.global_encoder(global_features)

        # --- 3. Symmetry Stream ---
        x_sym = self.symmetry_embed(spacegroups)
        x_sym = self.symmetry_proj(x_sym)
        x_sym = F.relu(x_sym)

        # --- 4. Fusion ---
        # Concatenate all representations
        x_fused = torch.cat([x_atom_pooled, x_global, x_sym], dim=1)

        # Fusion MLP
        x_fused = self.fusion_encoder(x_fused)

        # Regression
        out = self.regressor(x_fused)

        return out
