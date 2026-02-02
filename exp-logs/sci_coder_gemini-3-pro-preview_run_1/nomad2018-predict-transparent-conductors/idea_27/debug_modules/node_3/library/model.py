import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from library.config import Config


class AtomicEncoder(nn.Module):
    """
    Wide MLP to encode atomic features into a high-dimensional latent space.
    Applies Batch Normalization and Dropout for regularization.
    """

    def __init__(self):
        super(AtomicEncoder, self).__init__()
        input_dim = Config.ATOMIC_INPUT_DIM
        hidden_dim = Config.ATOMIC_HIDDEN_DIM
        output_dim = Config.LATENT_DIM
        dropout_prob = Config.DROPOUT

        # Wide MLP layers
        # Layer 1: Immediate Expansion
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)

        # Layer 2: Deep processing
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_prob)

        # Layer 3: Deep processing
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout_prob)

        # Output projection (Linear, no activation at the very end of encoder usually,
        # but here we project to latent space for pooling)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (Total_Atoms_In_Batch, ATOMIC_INPUT_DIM)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Final projection to latent embedding space
        x = self.fc_out(x)
        return x


class GlobalEncoder(nn.Module):
    """
    High-Capacity MLP to encode global macroscopic features.
    """

    def __init__(self):
        super(GlobalEncoder, self).__init__()
        input_dim = Config.GLOBAL_INPUT_DIM
        hidden_dim = Config.GLOBAL_HIDDEN_DIM
        output_dim = Config.LATENT_DIM
        dropout_prob = Config.DROPOUT

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_prob)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (Batch_Size, GLOBAL_INPUT_DIM)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc_out(x)
        return x


class CRR_DS_Model(nn.Module):
    """
    Chemically-Resolved Reciprocal Neighborhood Deep Sets (CRR-DS).
    Fuses local atomic context (via Dual Pooling) with global thermodynamic context.
    """

    def __init__(self):
        super(CRR_DS_Model, self).__init__()

        # Encoders
        self.atomic_encoder = AtomicEncoder()
        self.global_encoder = GlobalEncoder()

        # Fusion Head
        # Input: (Mean_Pool + Max_Pool) + Global_Embedding
        # Size: LATENT_DIM + LATENT_DIM + LATENT_DIM = 3 * LATENT_DIM
        fusion_input_dim = 3 * Config.LATENT_DIM
        hidden_dim = 256

        self.fusion_fc1 = nn.Linear(fusion_input_dim, hidden_dim)
        self.fusion_bn1 = nn.BatchNorm1d(hidden_dim)
        self.fusion_dropout1 = nn.Dropout(Config.DROPOUT)

        self.fusion_fc2 = nn.Linear(hidden_dim, 128)
        self.fusion_bn2 = nn.BatchNorm1d(128)
        self.fusion_dropout2 = nn.Dropout(Config.DROPOUT)

        # Output layer: Predicts 2 targets (formation_energy, bandgap_energy)
        self.output_layer = nn.Linear(128, 2)

    def forward(self, atomic_features, global_features, batch_indices):
        """
        Args:
            atomic_features: (Total_Atoms, ATOMIC_INPUT_DIM)
            global_features: (Batch_Size, GLOBAL_INPUT_DIM)
            batch_indices: (Total_Atoms,) indicating which crystal each atom belongs to.
        """
        # 1. Atomic Stream
        # Encode individual atoms
        atom_embeddings = self.atomic_encoder(
            atomic_features
        )  # (Total_Atoms, LATENT_DIM)

        # Dual Pooling: Aggregation
        # Scatter Mean
        mean_pool = scatter_mean(
            atom_embeddings, batch_indices, dim=0
        )  # (Batch_Size, LATENT_DIM)
        # Scatter Max (returns tuple (values, indices), we need values)
        max_pool, _ = scatter_max(
            atom_embeddings, batch_indices, dim=0
        )  # (Batch_Size, LATENT_DIM)

        # 2. Global Stream
        global_embeddings = self.global_encoder(
            global_features
        )  # (Batch_Size, LATENT_DIM)

        # 3. Fusion
        # Concatenate: [Mean_Pool, Max_Pool, Global]
        fused = torch.cat(
            [mean_pool, max_pool, global_embeddings], dim=1
        )  # (Batch_Size, 3*LATENT_DIM)

        # Regressor
        x = self.fusion_fc1(fused)
        x = self.fusion_bn1(x)
        x = F.relu(x)
        x = self.fusion_dropout1(x)

        x = self.fusion_fc2(x)
        x = self.fusion_bn2(x)
        x = F.relu(x)
        x = self.fusion_dropout2(x)

        output = self.output_layer(x)  # (Batch_Size, 2)

        return output
