import os
import torch


class Config:
    """
    Configuration class for the High-Capacity Stabilized Decoupled BiGRU (HCSD-BiGRU) strategy.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_58"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Paths (Metadata Parquet files)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (using .npz for numpy array storage)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npz")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Input features: 4 (ACGU) + 3 (Structure: .()) + 7 (Loop: SMIBHEX)
    INPUT_CHANNELS = 14

    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL = 3

    # Backbone (BiGRU)
    # Hidden dimension per direction is 384, resulting in 768 total output
    HIDDEN_DIM = 384
    NUM_LAYERS = 4

    # Regularization
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3

    # Gradient Clipping (Mandatory for hybrid architecture stability)
    MAX_GRAD_NORM = 1.0

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # All 5 targets used for Multi-Task Learning
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Only these 3 are used for validation metric calculation
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
