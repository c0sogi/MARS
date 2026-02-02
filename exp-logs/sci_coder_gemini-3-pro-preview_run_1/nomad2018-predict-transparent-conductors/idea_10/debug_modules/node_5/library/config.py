import os
import torch


class Config:
    """
    Configuration class for the Residual Tri-Pool Deep Sets (RT-DS) project.
    Contains paths, data parameters, model architecture settings, and training hyperparameters.
    """

    # -------------------------------------------------------------------------
    # Paths and Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching processed data and saving models
    # Explicitly set to idea_10 as per requirements
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")

    # Cache Paths (for numpy/parquet files)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.npz")

    # -------------------------------------------------------------------------
    # Data Processing Configuration
    # -------------------------------------------------------------------------
    # Atom type mapping for one-hot encoding
    ATOM_MAP = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
    NUM_ATOM_TYPES = len(ATOM_MAP)

    # Feature Dimensions
    # Atomic Stream: One-hot (4) + Centered Coords (3) + PBC Nearest Neighbor Dist (1)
    ATOMIC_FEATURE_DIM = 8

    # Global Stream: Lattice Vectors (3) + Lattice Angles (3) + Volume (1) + Density (1) + Stoichiometry (4)
    GLOBAL_FEATURE_DIM = 12

    # Target Variables
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    NUM_TARGETS = len(TARGET_COLS)

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters (RT-DS)
    # -------------------------------------------------------------------------
    # Latent dimension for the atomic stream (Residual Point Processor)
    LATENT_DIM = 512

    # Hidden dimension for the global stream MLP
    GLOBAL_HIDDEN_DIM = 256

    # Number of Residual Blocks in the atomic encoder
    NUM_RES_BLOCKS = 3

    # Dropout rate for regularization
    DROPOUT_RATE = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 20

    # Debugging: Set to an integer (e.g., 100) to limit dataset size for quick testing.
    # Set to None for full training.
    DEBUG_SAMPLE_SIZE = None

    # -------------------------------------------------------------------------
    # Hardware & Execution
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for working files and submissions.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(
            f"Config: Directories ensured at {cls.WORKING_DIR} and {cls.SUBMISSION_DIR}"
        )
        print(f"Config: Device set to {cls.DEVICE}")
