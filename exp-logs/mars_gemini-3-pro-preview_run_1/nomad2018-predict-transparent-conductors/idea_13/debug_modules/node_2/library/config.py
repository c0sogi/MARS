import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Cache Files (using .npz for efficient numpy storage)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data.npz")

    # Model Checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pt")

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    ATOM_TYPES = ["Al", "Ga", "In", "O"]
    # Based on data analysis, max atoms in unit cell is 80.
    # We set a slightly higher buffer or exact match.
    MAX_ATOMS = 80
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Feature Dimensions
    # -------------------------------------------------------------------------
    # Atomic Stream Features:
    # 1. One-hot encoding of atom type (4)
    # 2. Centered Cartesian coordinates (x, y, z) (3)
    # 3. Fractional coordinates (u, v, w) (3)
    # 4. Nearest Neighbor Distance (1)
    # 5. Potential Proxy (sum of inverse distances) (1)
    ATOMIC_INPUT_DIM = 4 + 3 + 3 + 1 + 1  # Total: 12

    # Global Stream Features:
    # 1. Lattice vector lengths (a, b, c) (3)
    # 2. Lattice angles (alpha, beta, gamma) (3)
    # 3. Unit Cell Volume (1)
    # 4. Atomic Density (1)
    # 5. Stoichiometry (Al, Ga, In percentages) (3)
    # 6. Total Number of Atoms (1)
    GLOBAL_INPUT_DIM = 3 + 3 + 1 + 1 + 3 + 1  # Total: 12

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Wide MLP settings for Atomic Stream
    ATOMIC_HIDDEN_DIM = 512

    # MLP settings for Global Stream
    GLOBAL_HIDDEN_DIM = 256

    # Latent dimension for fusion
    LATENT_DIM = 128

    # Regularization
    DROPOUT = 0.2

    # -------------------------------------------------------------------------
    # Training Parameters
    # -------------------------------------------------------------------------
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 200
    PATIENCE = 20  # Early stopping patience
    SEED = 42

    # -------------------------------------------------------------------------
    # Compute
    # -------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def print_config():
        """Prints the current configuration."""
        print("=" * 40)
        print("  MC-PDS CONFIGURATION")
        print("=" * 40)
        print(f"Device: {Config.DEVICE}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Epochs: {Config.EPOCHS}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Atomic Input Dim: {Config.ATOMIC_INPUT_DIM}")
        print(f"Global Input Dim: {Config.GLOBAL_INPUT_DIM}")
        print(f"Working Dir: {Config.WORKING_DIR}")
        print("=" * 40)
