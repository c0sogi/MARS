import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    # Read-only input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea
    # All intermediate files, cache, and model checkpoints go here
    WORKING_DIR = "./working/idea_27"

    # Sub-directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "model")
    PREDICTIONS_DIR = os.path.join(WORKING_DIR, "predictions")

    # Final submission file path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model checkpoint path
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")

    # -------------------------------------------------------------------------
    # Data & Preprocessing Parameters
    # -------------------------------------------------------------------------
    # Random seed for reproducibility
    SEED = 42

    # Atomic species mapping
    ATOMIC_MAP = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
    NUM_ATOM_TYPES = len(ATOMIC_MAP)

    # Atomic masses for feature engineering (approximate)
    ATOMIC_MASSES = {"Al": 26.98, "Ga": 69.72, "In": 114.82, "O": 15.999}

    # Number of nearest neighbors for local packing density calculation
    K_NEIGHBORS = 12

    # Feature Dimensions
    # Atomic Features:
    # 1. One-hot encoding (4 dims)
    # 2. Centered Coordinates (3 dims)
    # 3. Reciprocal Proximity to each type (4 dims)
    # 4. Local Packing Density (1 dim)
    ATOMIC_INPUT_DIM = 4 + 3 + 4 + 1  # Total: 12

    # Global Features:
    # 1. Lattice lengths (3 dims)
    # 2. Lattice angles (3 dims)
    # 3. Volume (1 dim)
    # 4. Atomic Density (1 dim)
    # 5. Stoichiometry (3 dims: Al, Ga, In)
    # 6. Total Atoms (1 dim)
    GLOBAL_INPUT_DIM = 3 + 3 + 1 + 1 + 3 + 1  # Total: 12

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Atomic Stream (Wide MLP)
    ATOMIC_HIDDEN_DIM = 512

    # Global Stream (High Capacity)
    GLOBAL_HIDDEN_DIM = 256

    # Projection dimension before fusion
    LATENT_DIM = 128

    # Regularization
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 200
    PATIENCE = 20  # Early stopping patience

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup_directories():
        """Creates necessary directories if they don't exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Initialize directories immediately when module is imported
Config.setup_directories()
