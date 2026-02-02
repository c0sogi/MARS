import os
import torch


class Config:
    """
    Centralized configuration for the Attention-Augmented Deep Residual BiGRU strategy.
    """

    # --------------------------------------------------------------------------
    # General Settings
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2  # Number of subprocesses for data loading

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Data Files
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories (Write Allowed)
    WORKING_DIR = "./working/idea_6"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Configuration
    # --------------------------------------------------------------------------
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Features:
    # 4 (Nucleotides: A, G, U, C) +
    # 3 (Structure: (, ), .) +
    # 7 (Loop Type: S, M, I, B, H, E, X)
    INPUT_DIM = 14

    # Output Targets: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    OUTPUT_DIM = 5

    # Indices of the targets that are actually scored in the competition
    # 0: reactivity, 1: deg_Mg_pH10, 3: deg_Mg_50C
    SCORED_TARGET_INDICES = [0, 1, 3]

    # Caching Logic
    LOAD_CACHED_DATA = True

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    # 1. Convolutional Stem
    STEM_KERNEL_SIZE = 3
    STEM_FILTERS = 256

    # 2. Deep Residual BiGRU Backbone
    RNN_HIDDEN_DIM = 256
    RNN_LAYERS = 2
    RNN_DROPOUT = 0.3

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping patience
    MAX_GRAD_NORM = 1.0

    @classmethod
    def create_directories(cls):
        """Ensures that the necessary working and cache directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


# Initialize directories upon module import
Config.create_directories()
