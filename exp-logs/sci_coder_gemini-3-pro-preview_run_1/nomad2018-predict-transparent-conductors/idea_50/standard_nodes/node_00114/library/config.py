import os
import torch


class Config:
    """
    Configuration for Aligned Multi-Scale Anisotropic Deep Sets (AMSA-DS).
    """

    # -------------------------------------------------------------------------
    # Random Seed for Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea
    WORKING_DIR = "./working/idea_50"

    # Sub-directories for caching processed data and saving model artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    EXECUTION_DIR = os.path.join(WORKING_DIR, "execution")
    SUBMISSION_DIR = "./submission"  # Final submission location

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Neighbor counts for multi-scale context
    # K_NEAR corresponds to the immediate coordination shell
    # K_FAR corresponds to the broader crystal field
    K_NEAR = 6
    K_FAR = 24

    # Input Feature Dimensions
    # Atomic Stream:
    #   4 (One-Hot Identity: Al, Ga, In, O)
    # + 3 (Centered Coordinates x, y, z)
    # + 1 (Nearest Neighbor Distance d_min)
    # + 2 (Multi-Scale Packing Ratios R_6, R_24)
    # + 4 (Weighted Chemical Context K=6)
    # + 4 (Weighted Chemical Context K=24)
    # = 18 Total Dimensions
    ATOMIC_FEATURE_DIM = 18

    # Global Stream:
    #   6 (Lattice lengths & angles)
    # + 1 (Volume)
    # + 1 (Atomic Density)
    # + 4 (Stoichiometry fractions)
    # + 1 (Total Number of Atoms)
    # + 3 (Lattice Aspect Ratios)
    # + 3 (Weighted Physics: Mass, Radius, Electronegativity)
    # + 1 (Angular Distortion)
    # = 20 Total Dimensions
    GLOBAL_FEATURE_DIM = 20

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Atomic Stream Encoder (Wide MLP)
    ATOMIC_HIDDEN_DIM = 512
    ATOMIC_LAYERS = 3

    # Global Stream Encoder
    GLOBAL_HIDDEN_DIM = 256
    GLOBAL_LAYERS = 2

    # Fusion Head
    FUSION_HIDDEN_DIM = 256

    # Regularization
    DROPOUT_RATE = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64  # Number of crystals per batch
    LEARNING_RATE = 1e-3  # Initial learning rate
    WEIGHT_DECAY = 1e-4  # L2 regularization
    NUM_EPOCHS = 200  # Maximum training epochs
    PATIENCE = 20  # Early stopping patience

    # Scheduler settings (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-6

    # -------------------------------------------------------------------------
    # Computation
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    # -------------------------------------------------------------------------
    # Debugging
    # -------------------------------------------------------------------------
    # If True, runs on a small subset of data to verify pipeline
    DEBUG = False
    DEBUG_SIZE = 100

    @classmethod
    def setup(cls):
        """Ensure necessary directories exist."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.EXECUTION_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on module import
Config.setup()
