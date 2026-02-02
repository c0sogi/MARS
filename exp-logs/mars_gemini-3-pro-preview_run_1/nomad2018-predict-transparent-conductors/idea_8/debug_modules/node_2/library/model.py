import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Residual Block for the Atomic Stream.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> Add -> ReLU
    """

    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        # x shape: (Batch, Atoms, Dim)
        residual = x

        # Layer 1
        out = self.linear1(x)

        # BN expects (N, C, L) for 3D input, so we transpose
        out = out.transpose(1, 2)  # (B, D, A)
        out = self.bn1(out)
        out = out.transpose(1, 2)  # (B, A, D)

        out = F.relu(out)
        out = self.dropout(out)

        # Layer 2
        out = self.linear2(out)

        out = out.transpose(1, 2)
        out = self.bn2(out)
        out = out.transpose(1, 2)

        # Residual Connection
        out = out + residual
        out = F.relu(out)

        return out


class SIRDS_SP(nn.Module):
    """
    Symmetry-Informed Residual Deep Sets with Statistical Pooling (SI-RDS-SP).

    Streams:
    1. Atomic: Residual MLP processing point cloud -> Mean/Max/Std Pooling.
    2. Global: MLP processing macroscopic features.
    3. Symmetry: Embedding of Spacegroup.

    Fusion: Concatenation -> MLP -> Regression.
    """

    def __init__(self):
        super(SIRDS_SP, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Atomic Stream (Residual Point Processor)
        # ---------------------------------------------------------------------
        self.atomic_input_dim = Config.ATOMIC_INPUT_DIM
        self.atomic_hidden_dim = Config.ATOMIC_HIDDEN_DIM

        # Initial projection to hidden dim
        self.atomic_proj = nn.Linear(self.atomic_input_dim, self.atomic_hidden_dim)

        # Stack of Residual Blocks
        self.atomic_blocks = nn.ModuleList(
            [
                ResidualBlock(self.atomic_hidden_dim, Config.ATOMIC_DROPOUT)
                for _ in range(Config.NUM_RESIDUAL_BLOCKS)
            ]
        )

        # ---------------------------------------------------------------------
        # 2. Global Stream (Thermodynamic Context)
        # ---------------------------------------------------------------------
        self.global_input_dim = Config.GLOBAL_INPUT_DIM
        self.global_hidden_dim = Config.GLOBAL_HIDDEN_DIM

        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_input_dim, self.global_hidden_dim),
            nn.BatchNorm1d(self.global_hidden_dim),
            nn.ReLU(),
            nn.Dropout(Config.GLOBAL_DROPOUT),
            nn.Linear(self.global_hidden_dim, self.global_hidden_dim),
            nn.BatchNorm1d(self.global_hidden_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------------------
        # 3. Symmetry Stream (Crystallographic Prior)
        # ---------------------------------------------------------------------
        self.symmetry_embedding = nn.Embedding(
            num_embeddings=Config.NUM_SPACEGROUPS,
            embedding_dim=Config.SYMMETRY_EMBEDDING_DIM,
            padding_idx=0,
        )

        # ---------------------------------------------------------------------
        # 4. Fusion Head
        # ---------------------------------------------------------------------
        # Aggregation output: Mean (H) + Max (H) + Std (H) = 3H
        atomic_out_dim = 3 * self.atomic_hidden_dim

        fusion_input_dim = (
            atomic_out_dim + self.global_hidden_dim + Config.SYMMETRY_EMBEDDING_DIM
        )

        layers = []
        in_dim = fusion_input_dim

        for hidden_dim in Config.FUSION_HIDDEN_DIMS:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.FUSION_DROPOUT))
            in_dim = hidden_dim

        # Final regression layer
        layers.append(nn.Linear(in_dim, Config.NUM_TARGETS))

        self.fusion_head = nn.Sequential(*layers)

    def forward(self, atomic_x, atomic_mask, global_x, symmetry_x):
        """
        Args:
            atomic_x: (Batch, Max_Atoms, Atomic_Input_Dim)
            atomic_mask: (Batch, Max_Atoms) - Boolean mask (True for valid atoms)
            global_x: (Batch, Global_Input_Dim)
            symmetry_x: (Batch,) - Spacegroup IDs
        """

        # --- 1. Atomic Stream Processing ---
        # Project to hidden dim
        h = self.atomic_proj(atomic_x)  # (B, L, H)

        # Pass through residual blocks
        for block in self.atomic_blocks:
            h = block(h)

        # --- Pooling with Masking ---
        # Expand mask for broadcasting: (B, L, 1)
        mask_expanded = atomic_mask.unsqueeze(-1).float()

        # Zero out padding (ensure masked values are exactly 0)
        h_masked = h * mask_expanded

        # Count valid atoms per sample
        atom_counts = mask_expanded.sum(dim=1)  # (B, 1)
        atom_counts = torch.clamp(atom_counts, min=1.0)  # Avoid div by zero

        # Mean Pooling
        sum_pooled = h_masked.sum(dim=1)  # (B, H)
        mean_pooled = sum_pooled / atom_counts

        # Max Pooling
        # Since ReLU is used, values are >= 0. Padding is 0.
        # Max will pick up features or 0 if all features are 0.
        max_pooled = h_masked.max(dim=1)[0]  # (B, H)

        # Std Pooling
        # Var = Mean(x^2) - Mean(x)^2
        sum_sq = (h_masked**2).sum(dim=1)
        mean_sq = sum_sq / atom_counts
        var = mean_sq - (mean_pooled**2)
        var = torch.clamp(var, min=1e-6)  # Numerical stability
        std_pooled = torch.sqrt(var)

        # Concatenate Atomic Embeddings
        atomic_emb = torch.cat([mean_pooled, max_pooled, std_pooled], dim=1)  # (B, 3H)

        # --- 2. Global Stream Processing ---
        global_emb = self.global_mlp(global_x)  # (B, GH)

        # --- 3. Symmetry Stream Processing ---
        symmetry_emb = self.symmetry_embedding(symmetry_x)  # (B, SH)

        # --- 4. Fusion and Prediction ---
        fused_features = torch.cat([atomic_emb, global_emb, symmetry_emb], dim=1)
        output = self.fusion_head(fused_features)

        return output
