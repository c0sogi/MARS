import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class AtomicBranch(nn.Module):
    """
    Encodes atomic features using Deep Sets with Dual Pooling.
    Cite Lesson 5: Dual Pooling Aggregation for Enhanced Set Representation.
    Cite Lesson 6: Immediate Feature Expansion in Deep Sets Encoders.
    """

    def __init__(self):
        super(AtomicBranch, self).__init__()

        self.num_atom_types = Config.NUM_ATOM_TYPES
        self.input_dim = Config.ATOM_INPUT_DIM
        self.hidden_dim = Config.ATOM_HIDDEN_DIM

        # MLP for atomic features (Wide Immediate Expansion)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Dimension after global pooling (Mean + Max)
        self.global_dim = self.hidden_dim * 2

    def forward(self, atom_types, mask):
        """
        Args:
            atom_types: (B, N) LongTensor
            mask: (B, N) FloatTensor (1 for atom, 0 for padding)
        """
        B, N = atom_types.shape

        # One-hot encode atom types: (B, N, num_types)
        x = F.one_hot(atom_types, num_classes=self.num_atom_types).float()

        # Process atoms: (B, N, hidden_dim)
        x = self.mlp(x)

        # Masking
        mask = mask.unsqueeze(-1)  # (B, N, 1)
        x = x * mask

        # Global Mean Pooling
        sum_x = x.sum(dim=1)  # (B, hidden)
        counts = mask.sum(dim=1)  # (B, 1)
        counts = torch.clamp(counts, min=1.0)
        mean_pool = sum_x / counts

        # Global Max Pooling
        # Set padded values to -inf
        neg_inf = torch.ones_like(x) * -1e9
        masked_max_input = torch.where(mask > 0.5, x, neg_inf)
        max_pool = masked_max_input.max(dim=1)[0]  # (B, hidden)

        # Concatenate (Dual Pooling)
        global_embedding = torch.cat([mean_pool, max_pool], dim=1)

        return global_embedding


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
    Lattice-Conditioned Deep Sets.
    Reverts to Deep Sets architecture (Cite Lesson 8) with improved components.
    """

    def __init__(self):
        super(LCDS, self).__init__()

        self.atomic_branch = AtomicBranch()
        self.lattice_branch = LatticeBranch()

        # Calculate fusion dimension
        # Atomic: 256 * 2 = 512
        # Lattice: 128
        self.fusion_input_dim = (
            self.atomic_branch.global_dim + self.lattice_branch.hidden_dim
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

    def forward(self, atom_types, lattice_features, mask):
        """
        Args:
            atom_types: (B, N)
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
