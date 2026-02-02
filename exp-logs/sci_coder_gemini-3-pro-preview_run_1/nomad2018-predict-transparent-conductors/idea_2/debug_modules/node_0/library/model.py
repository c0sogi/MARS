import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InteractionBranch(nn.Module):
    """
    Encodes atomic interactions using pairwise distances and atom types.
    Performs local aggregation (edges -> atoms) and global aggregation (atoms -> crystal).
    """

    def __init__(self):
        super(InteractionBranch, self).__init__()

        self.num_atom_types = Config.NUM_ATOM_TYPES
        self.edge_input_dim = Config.EDGE_INPUT_DIM  # 1 (dist) + 2 * num_atom_types
        self.edge_hidden_dim = Config.EDGE_HIDDEN_DIM

        # MLP to project edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.edge_hidden_dim, self.edge_hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Dimension after local pooling (Mean + Max)
        self.local_dim = self.edge_hidden_dim * 2

        # Dimension after global pooling (Mean + Max)
        self.global_dim = self.local_dim * 2

    def forward(self, atom_types, dist_matrix, mask):
        """
        Args:
            atom_types: (B, N) LongTensor
            dist_matrix: (B, N, N) FloatTensor
            mask: (B, N) FloatTensor (1 for atom, 0 for padding)
        """
        B, N = atom_types.shape

        # 1. Construct Edge Features
        # One-hot encode atom types: (B, N, num_types)
        one_hot = F.one_hot(atom_types, num_classes=self.num_atom_types).float()

        # Broadcast for edges (i, j)
        # atom_i: (B, N, 1, num_types) -> repeat over j
        atom_i = one_hot.unsqueeze(2).expand(B, N, N, self.num_atom_types)
        # atom_j: (B, 1, N, num_types) -> repeat over i
        atom_j = one_hot.unsqueeze(1).expand(B, N, N, self.num_atom_types)

        # Distance: (B, N, N, 1)
        dist = dist_matrix.unsqueeze(-1)

        # Concatenate: (B, N, N, 1 + 2*num_types)
        edge_features = torch.cat([dist, atom_i, atom_j], dim=-1)

        # 2. Process Edges
        # (B, N, N, edge_hidden_dim)
        edge_embeddings = self.edge_mlp(edge_features)

        # 3. Local Aggregation (Edges -> Atoms)
        # Mask for edges: M_ij = mask_i * mask_j
        # (B, N, 1) * (B, 1, N) -> (B, N, N)
        edge_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        edge_mask = edge_mask.unsqueeze(-1)  # (B, N, N, 1)

        # Apply mask
        # For Mean: set padded to 0
        masked_sum = (edge_embeddings * edge_mask).sum(
            dim=2
        )  # Sum over j -> (B, N, hidden)
        # Count neighbors for each atom i
        neighbor_counts = edge_mask.sum(dim=2)  # (B, N, 1)
        neighbor_counts = torch.clamp(neighbor_counts, min=1.0)
        local_mean = masked_sum / neighbor_counts

        # For Max: set padded to -inf
        # Create a large negative tensor
        neg_inf = torch.ones_like(edge_embeddings) * -1e9
        # Where mask is 1, keep embedding, else -inf
        masked_max_input = torch.where(edge_mask > 0.5, edge_embeddings, neg_inf)
        local_max = masked_max_input.max(dim=2)[0]  # Max over j -> (B, N, hidden)

        # Concatenate Local Features: (B, N, 2 * hidden)
        local_features = torch.cat([local_mean, local_max], dim=-1)

        # 4. Global Aggregation (Atoms -> Crystal)
        # Mask for atoms: (B, N, 1)
        atom_mask = mask.unsqueeze(-1)

        # Global Mean
        global_sum = (local_features * atom_mask).sum(dim=1)  # (B, 2*hidden)
        atom_counts = atom_mask.sum(dim=1)  # (B, 1)
        atom_counts = torch.clamp(atom_counts, min=1.0)
        global_mean = global_sum / atom_counts

        # Global Max
        neg_inf_global = torch.ones_like(local_features) * -1e9
        masked_global_max_input = torch.where(
            atom_mask > 0.5, local_features, neg_inf_global
        )
        global_max = masked_global_max_input.max(dim=1)[0]  # (B, 2*hidden)

        # Concatenate Global Features: (B, 4 * hidden)
        global_embedding = torch.cat([global_mean, global_max], dim=-1)

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


class IALCDS(nn.Module):
    """
    Interaction-Aware Lattice-Conditioned Deep Sets.
    Fuses structural and lattice embeddings to predict material properties.
    """

    def __init__(self):
        super(IALCDS, self).__init__()

        self.interaction_branch = InteractionBranch()
        self.lattice_branch = LatticeBranch()

        # Calculate fusion dimension
        # Interaction: (edge_hidden * 2 [local] * 2 [global]) = 128 * 4 = 512
        # Lattice: 64
        self.fusion_input_dim = (
            self.interaction_branch.global_dim + self.lattice_branch.hidden_dim
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
            dist_matrix: (B, N, N)
            lattice_features: (B, 6)
            mask: (B, N)
        """
        # 1. Get Interaction Embedding
        struct_embed = self.interaction_branch(atom_types, dist_matrix, mask)

        # 2. Get Lattice Embedding
        latt_embed = self.lattice_branch(lattice_features)

        # 3. Fusion
        fused = torch.cat([struct_embed, latt_embed], dim=1)

        # 4. Prediction
        output = self.regressor(fused)

        return output
