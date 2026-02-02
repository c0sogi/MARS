import os
import torch


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    # Input data is read-only
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for artifacts (checkpoints, cache)
    # Using specific idea folder as per instructions
    WORKING_DIR = "./working/idea_57"
    CACHE_DIR = WORKING_DIR

    # Metadata file paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Submission paths
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model checkpoint path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Target Columns - Multi-Task Learning on all 5
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns used for validation scoring (MCRMSE)
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    NUM_TARGETS = len(TARGET_COLS)

    # ==========================================
    # Tokenization & Features
    # ==========================================
    # One-hot encoding maps
    # Sequence (4 dims)
    TOKEN_MAP_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
    # Structure (3 dims)
    TOKEN_MAP_STRUCT = {"(": 0, ")": 1, ".": 2}
    # Loop Type (7 dims)
    TOKEN_MAP_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    # Total Feature Dimensions: 4 + 3 + 7 = 14
    INPUT_CHANNELS = 14

    # ==========================================
    # Model Architecture (SDBR-BiGRU)
    # ==========================================
    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL_SIZE = 3

    # Backbone
    HIDDEN_DIM = 384
    NUM_LAYERS = 3  # 3-Layer Backbone for stability
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Gradient Clipping (Mandatory for hybrid architecture)
    MAX_GRAD_NORM = 1.0

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR

    # Early Stopping
    PATIENCE = 5

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @staticmethod
    def setup_directories():
        """Ensure necessary directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
