import torch

# Global configuration for training and model hyperparameters
CONFIG = {
    "seed": 42,
    "batch_size": 32,
    "epochs": 200,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "k_neighbors": 12,
    "hidden_dim_atomic": 512,
    "hidden_dim_global": 256,
    "latent_dim": 512,
}

# Physical constants for feature engineering
# Format: [Atomic Mass (u), Covalent Radius (Angstrom), Electronegativity (Pauling)]
PHYSICAL_CONSTANTS = {
    "Al": [26.98, 1.21, 1.61],
    "Ga": [69.72, 1.22, 1.81],
    "In": [114.82, 1.42, 1.78],
    "O": [16.00, 0.66, 3.44],
}

# Mapping from atom symbol to one-hot index
ATOM_MAP = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
