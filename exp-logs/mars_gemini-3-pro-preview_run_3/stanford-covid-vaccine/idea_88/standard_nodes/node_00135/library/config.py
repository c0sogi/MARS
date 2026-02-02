import os
import torch


class Config:
    """
    Global configuration for the High-Capacity Stabilized GLU-Decoupled BiGRU strategy.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 50  # Number of samples to use in debug mode

    # =========================================================================
    # Paths
    # =========================================================================
    # Read-only input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata file paths (Parquet format)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output directories
    WORKING_DIR = "./working/idea_88"
    SUBMISSION_DIR = "./submission"

    # Submission paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache paths for processed tensors (using .npy as requested)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_cache.npy")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    SEQ_SCORED = 68
    NUM_TARGETS = 5  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    # Input Features: One-hot encoding
    # 4 (Nucleotide: A, G, C, U)
    # + 3 (Structure: (, ), .)
    # + 7 (Loop Type: S, M, I, B, H, E, X)
    INPUT_CHANNELS = 14

    # =========================================================================
    # Model Hyperparameters
    # Strategy: High-Capacity Stabilized GLU-Decoupled BiGRU
    # =========================================================================
    HIDDEN_DIM = 384  # Dimension per direction (Total 768 for BiGRU)
    NUM_LAYERS = 4  # Deep 4-layer backbone
    DROPOUT = 0.1  # Conservative regularization (0.1)

    # Convolutional Stem
    CNN_FILTERS = 256
    KERNEL_SIZE = 3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    PATIENCE = 7  # Early stopping patience
    MAX_GRAD_NORM = 1.0  # Gradient clipping (Mandatory for stability)
    WEIGHT_DECAY = 1e-4

    # Scheduler settings
    MIN_LR = 1e-6

    # =========================================================================
    # Compute Resources
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    @staticmethod
    def setup():
        """
        Ensures that the necessary working and submission directories exist.
        This is called immediately upon module import.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Initialize environment
Config.setup()
