import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from library.config import Config


class AtomicStream(nn.Module):
    """
    Processes atomic features (Type + Centered XYZ + RBF) using a Wide MLP.
    Projects node-level features into a latent space before pooling.
    """

    def __init__(self):
        super(AtomicStream, self).__init__()
        # Wide MLP: Input -> Hidden -> Latent
        self.fc1 = nn.Linear(Config.ATOMIC_INPUT_DIM, Config.ATOMIC_HIDDEN_DIM)
        self.fc2 = nn.Linear(Config.ATOMIC_HIDDEN_DIM, Config.ATOMIC_LATENT_DIM)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

    def forward(self, x):
        # x shape: (Total_Atoms, ATOMIC_INPUT_DIM)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # Output shape: (Total_Atoms, ATOMIC_LATENT_DIM)
        return x


class LatticeStream(nn.Module):
    """
    Processes global lattice features using a High-Capacity MLP.
    Extracts features related to unit cell dimensions and angles.
    """

    def __init__(self):
        super(LatticeStream, self).__init__()

        layers = []
        input_dim = Config.LATTICE_INPUT_DIM

        # Build MLP layers from config
        for hidden_dim in Config.LATTICE_HIDDEN_DIMS:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.DROPOUT_RATE))
            input_dim = hidden_dim

        # Final projection to output dim
        layers.append(nn.Linear(input_dim, Config.LATTICE_OUTPUT_DIM))
        # Optional: Activation on final lattice embedding? Usually embedding spaces are linear or activated.
        # Following standard MLP blocks, we'll leave the final projection linear to be fused.
        # But commonly a non-linearity is applied before fusion if it's a deep network.
        # Given the description "High-Capacity MLP", we stick to the hidden layers structure.
        # The final layer maps to LATTICE_OUTPUT_DIM.

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch_Size, LATTICE_INPUT_DIM)
        return self.mlp(x)


class RBFDualStreamDeepSets(nn.Module):
    """
    RBF-Augmented Dual-Stream Deep Sets Architecture.

    Stream 1: Atomic Stream -> Wide MLP -> Dual Pooling (Mean + Max).
    Stream 2: Lattice Stream -> High-Capacity MLP.
    Fusion: Concatenation -> Regressor MLP -> Output.
    """

    def __init__(self):
        super(RBFDualStreamDeepSets, self).__init__()

        self.atomic_stream = AtomicStream()
        self.lattice_stream = LatticeStream()

        # Regressor MLP
        layers = []
        input_dim = Config.FUSION_INPUT_DIM

        for hidden_dim in Config.REGRESSOR_HIDDEN_DIMS:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.DROPOUT_RATE))
            input_dim = hidden_dim

        # Final output layer (2 targets)
        layers.append(nn.Linear(input_dim, Config.OUTPUT_DIM))

        self.regressor = nn.Sequential(*layers)

    def forward(self, atomic_features, lattice_features, batch_indices):
        """
        Args:
            atomic_features: Tensor (Total_Atoms, ATOMIC_INPUT_DIM)
            lattice_features: Tensor (Batch_Size, LATTICE_INPUT_DIM)
            batch_indices: LongTensor (Total_Atoms,) mapping atoms to batch index

        Returns:
            predictions: Tensor (Batch_Size, 2)
        """
        # 1. Atomic Stream Processing
        # (Total_Atoms, ATOMIC_LATENT_DIM)
        atom_latent = self.atomic_stream(atomic_features)

        # 2. Dual Pooling (Aggregation)
        # Global Mean Pooling
        # (Batch_Size, ATOMIC_LATENT_DIM)
        # Ensure batch_size is derived correctly or scatter handles size automatically based on max index
        # We assume batch_indices covers 0 to Batch_Size-1
        mean_pool = scatter_mean(atom_latent, batch_indices, dim=0)

        # Global Max Pooling
        # scatter_max returns (values, indices)
        max_pool, _ = scatter_max(atom_latent, batch_indices, dim=0)

        # Concatenate aggregated features
        # (Batch_Size, ATOMIC_LATENT_DIM * 2)
        atomic_agg = torch.cat([mean_pool, max_pool], dim=1)

        # 3. Lattice Stream Processing
        # (Batch_Size, LATTICE_OUTPUT_DIM)
        lattice_latent = self.lattice_stream(lattice_features)

        # 4. Fusion
        # (Batch_Size, FUSION_INPUT_DIM)
        fused = torch.cat([atomic_agg, lattice_latent], dim=1)

        # 5. Regression
        # (Batch_Size, 2)
        output = self.regressor(fused)

        return output
