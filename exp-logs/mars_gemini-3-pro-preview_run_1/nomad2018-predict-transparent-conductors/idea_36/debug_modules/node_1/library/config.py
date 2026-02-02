import os
import torch


class Config:
    """
    Configuration class for the Parsimonious Geometric Wide Deep Sets (PG-WDS) strategy.
    Centralizes all hyperparameters, file paths, and constants.
    """

    # -------------------------------------------------------------------------
    # Directory Configuration
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea to store cached data and models
    WORKING_DIR = "./working/idea_36"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Number of nearest neighbors to consider for the d_mean calculation (Local Packing Density)
    K_NEIGHBORS = 12

    # List of atomic species present in the dataset for one-hot encoding
    ATOM_TYPES = ["Al", "Ga", "In", "O"]

    # Feature Dimensions Calculation
    # Atomic Stream Features:
    #   1. One-hot encoding of atom type (len(ATOM_TYPES) = 4)
    #   2. Centered Cartesian coordinates (x, y, z) (3)
    #   3. Nearest neighbor distance d_min (1)
    #   4. Mean neighbor distance d_mean (1)
    ATOMIC_INPUT_DIM = 4 + 3 + 1 + 1  # Total: 9

    # Global Stream Features:
    #   1. Lattice vector lengths (lv1, lv2, lv3) (3)
    #   2. Lattice angles (alpha, beta, gamma) (3)
    #   3. Unit cell volume (1)
    #   4. Atomic density (1)
    #   5. Stoichiometry (Al, Ga, In composition) (3)
    #   6. Total number of atoms (1)
    GLOBAL_INPUT_DIM = 3 + 3 + 1 + 1 + 3 + 1  # Total: 12

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Wide MLP dimension for the Atomic Stream to project low-dim features into high-dim space
    ATOMIC_HIDDEN_DIM = 512

    # Hidden dimension for the Global Stream MLP
    GLOBAL_HIDDEN_DIM = 256

    # Hidden dimension for the Fusion Head MLP after concatenating streams
    FUSION_HIDDEN_DIM = 256

    # Dropout rate for regularization across the network
    DROPOUT_RATE = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Regularization for wide layers
    BATCH_SIZE = 64
    NUM_EPOCHS = 200

    # Early Stopping parameters to prevent overfitting
    EARLY_STOPPING_PATIENCE = 20

    # Learning Rate Scheduler (ReduceLROnPlateau) settings
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 10
    SCHEDULER_MIN_LR = 1e-6

    # -------------------------------------------------------------------------
    # Hardware Configuration
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Number of workers for DataLoader (optimized for 12 vCPUs)
    NUM_WORKERS = 4
