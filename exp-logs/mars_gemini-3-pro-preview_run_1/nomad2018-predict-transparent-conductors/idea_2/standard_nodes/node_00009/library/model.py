import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class AtomicBranch(nn.Module):
    """
    Encodes atomic composition using Deep Sets.
    Uses Immediate Feature Expansion (Cite Lesson 6) and Dual Pooling (Cite Lesson 5).
    """

    def __init__(self):
        super(AtomicBranch, self).__init__()

        self.input_dim = Config.ATOMIC_INPUT_DIM
        self.hidden_dim = Config.ATOMIC_HIDDEN_DIM

        # Wide MLP for immediate expansion (Cite Lesson 6)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Output dimension: Mean + Max pooling (Cite Lesson 5)
        self.output_dim = self.hidden_dim * 2

    def forward(self, atom_types, mask):
        """
        Args:
            atom_types: (B, N) LongTensor
            mask: (B, N) FloatTensor
        """
        # One-hot encode atom types: (B, N, 4)
        x = F.one_hot(atom_types, num_classes=self.input_dim).float()

        # Project: (B, N, hidden)
        x = self.mlp(x)

        # Masking
        mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)

        # 1. Global Mean Pooling (Cite Lesson 5)
        # Avoid Sum pooling for intensive targets (Cite Lesson 7)
        masked_sum = (x * mask_expanded).sum(dim=1)
        atom_counts = mask_expanded.sum(dim=1).clamp(min=1.0)
        global_mean = masked_sum / atom_counts

        # 2. Global Max Pooling (Cite Lesson 5)
        neg_inf = torch.ones_like(x) * -1e9
        x_masked_max = torch.where(mask_expanded > 0.5, x, neg_inf)
        global_max = x_masked_max.max(dim=1)[0]

        # Concatenate
        return torch.cat([global_mean, global_max], dim=1)


class LatticeBranch(nn.Module):
    """
    Encodes macroscopic lattice parameters.
    """

    def __init__(self):
        super(LatticeBranch, self).__init__()

        self.input_dim = Config.LATTICE_INPUT_DIM
        self.hidden_dim = Config.LATTICE_HIDDEN_DIM

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, lattice_features):
        """
        Args:
            lattice_features: (B, 6) FloatTensor
        """
        return self.mlp(lattice_features)


class LCDS(nn.Module):
    """
    Lattice-Conditioned Deep Sets (Cite Lesson 2).
    Combines atomic composition features (Deep Sets) with global lattice features.
    Avoids dense pairwise interactions (Cite Lesson 8).
    """

    def __init__(self):
        super(LCDS, self).__init__()

        self.atomic_branch = AtomicBranch()
        self.lattice_branch = LatticeBranch()

        # Calculate fusion dimension
        # Atomic: 256 * 2 = 512
        # Lattice: 64
        self.fusion_input_dim = (
            self.atomic_branch.output_dim + self.lattice_branch.hidden_dim
        )

        # Regressor MLP
        layers = []
        in_dim = self.fusion_input_dim

        for hidden_dim in Config.FUSION_HIDDEN_DIMS:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(Config.DROPOUT))
            in_dim = hidden_dim

        # Final output layer (2 targets)
        layers.append(nn.Linear(in_dim, 2))

        self.regressor = nn.Sequential(*layers)

    def forward(self, atom_types, dist_matrix, lattice_features, mask):
        """
        Args:
            atom_types: (B, N)
            dist_matrix: (B, N, N) - Ignored (Cite Lesson 8)
            lattice_features: (B, 6)
            mask: (B, N)
        """
        # 1. Get Atomic Embedding
        atom_embed = self.atomic_branch(atom_types, mask)

        # 2. Get Lattice Embedding
        latt_embed = self.lattice_branch(lattice_features)

        # 3. Fusion
        fused = torch.cat([atom_embed, latt_embed], dim=1)

        # 4. Prediction
        output = self.regressor(fused)

        return output
