import os
import torch


class Config:
    # Reproducibility
    SEED = 42

    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    CACHE_DIR = "./working/idea_15"
    MODEL_PATH = os.path.join(CACHE_DIR, "best_model.pt")
    SCALER_PATH = os.path.join(CACHE_DIR, "scalers.npz")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Data Processing
    ATOM_TYPES = ["Al", "Ga", "In", "O"]
    NUM_ATOM_TYPES = len(ATOM_TYPES)

    # Atomic features:
    # 4 (One-hot) + 3 (Centered Coords) + 1 (NN Dist) + 1 (Potential Proxy) = 9
    ATOMIC_INPUT_DIM = NUM_ATOM_TYPES + 3 + 1 + 1

    # Global features:
    # 3 (Lattice lengths) + 3 (Lattice angles) + 1 (Volume) + 1 (Density) +
    # 3 (Stoichiometry) + 1 (Total Atoms) = 12
    GLOBAL_INPUT_DIM = 3 + 3 + 1 + 1 + 3 + 1

    # Model Architecture (Wide Deep Sets)
    ATOMIC_HIDDEN_DIM = 512  # Wide MLP for atomic stream
    GLOBAL_HIDDEN_DIM = 256  # High-capacity MLP for global stream
    FUSION_HIDDEN_DIM = 256  # Hidden dim for fusion head
    DROPOUT_RATE = 0.2  # Regularization

    # Training Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Regularization
    NUM_EPOCHS = 200
    PATIENCE = 20  # Early stopping

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
