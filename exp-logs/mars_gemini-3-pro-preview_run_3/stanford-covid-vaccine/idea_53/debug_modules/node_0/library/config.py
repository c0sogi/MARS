import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Data Paths
    # =========================================================================
    # Input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-split and stratified)
    TRAIN_PARQUET = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PARQUET = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PARQUET = os.path.join(METADATA_DIR, "test.parquet")

    # Submission sample
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working directory for caching and model checkpoints
    WORKING_DIR = "./working/idea_53"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache file paths
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npy")

    # Model checkpoint path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Dimensions & Columns
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Feature Channels (One-Hot Encodings)
    # Nucleotides: A, G, C, U (4)
    # Structure: (, ), . (3)
    # Loop Type: S, M, I, B, H, E, X (7)
    INPUT_DIM = 14  # 4 + 3 + 7

    # Target Columns
    # We train on all 5 to leverage auxiliary signal (Multi-Task Learning)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Scored Columns (Subset used for validation metric and leaderboard)
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    SCORED_INDICES = [0, 1, 3]  # Indices corresponding to TARGET_COLS

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Convolutional Stem
    CNN_FILTERS = 256
    CNN_KERNEL_SIZE = 3

    # Backbone (BiGRU + Stabilized Interaction)
    HIDDEN_DIM = 384  # High capacity
    NUM_LAYERS = 4  # Deep architecture
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0  # Crucial for stability in deep RNNs

    NUM_EPOCHS = 20
    PATIENCE = 5  # Early stopping patience

    # Scheduler settings (Cosine Annealing)
    T_MAX = NUM_EPOCHS  # Cycle length
    ETA_MIN = 1e-6  # Minimum learning rate
