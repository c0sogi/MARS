import os
import torch


class Config:
    """
    Configuration for the High-Capacity Stabilized GLU-Decoupled BiGRU experiment.
    Defines model hyperparameters, training settings, and file paths.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Number of dataloader workers

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input data (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for outputs and cache (Write Enabled)
    WORKING_DIR = "./working/idea_87"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files (Parquet format)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Raw Input Files (for reference or submission format)
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Files
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npy")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Mappings for One-Hot Encoding
    # Sequence: 4 bases
    TOKEN_MAP_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
    # Structure: 3 types
    TOKEN_MAP_STRUCT = {".": 0, "(": 1, ")": 2}
    # Predicted Loop Type: 7 types
    TOKEN_MAP_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    # Input Feature Channels
    # 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    INPUT_CHANNELS = 14

    # Targets
    # Multi-Task Learning: Train on all 5, Evaluate on 3
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = 5

    # =========================================================================
    # Model Architecture
    # High-Capacity Stabilized GLU-Decoupled BiGRU
    # =========================================================================
    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL = 3

    # Recurrent Backbone
    # 384 per direction -> 768 total hidden size
    HIDDEN_DIM = 384
    NUM_LAYERS = 4

    # Regularization
    # Conservative dropout to preserve weak signals
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    EPOCHS = 50

    # Optimization
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    # Stability
    # Strict gradient clipping is mandatory for deep hybrid architectures
    MAX_GRAD_NORM = 1.0

    # Scheduler
    T_MAX = EPOCHS  # For Cosine Annealing
