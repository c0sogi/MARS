import os
import torch
import numpy as np
import random


class Config:
    # ==============================
    # File Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for Idea 13 (Gated Latent Spatial Conv-BiGRU)
    WORKING_DIR = "./working/idea_13"

    # Metadata file paths (parquet format)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Submission paths
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_FILE = os.path.join(WORKING_DIR, "submission.csv")

    # Cache file paths
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_cache.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_cache.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_cache.npy")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==============================
    # Data Specifications
    # ==============================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Input Channels:
    # 4 (Sequence: A, G, U, C)
    # + 3 (Structure: (, ), .)
    # + 7 (Loop Type: S, M, I, B, H, E, X)
    INPUT_CHANNELS = 14

    # Targets
    NUM_TARGETS = 5
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Only these 3 are scored in the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==============================
    # Model Hyperparameters
    # ==============================
    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL = 3

    # Recurrent Backbone (BiGRU)
    HIDDEN_DIM = 384
    NUM_LAYERS = 3
    BIDIRECTIONAL = True

    # Regularization
    DROPOUT = 0.3

    # ==============================
    # Training Hyperparameters
    # ==============================
    SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Optimization
    GRADIENT_CLIP_VAL = 1.0
    EARLY_STOPPING_PATIENCE = 5

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # ==============================
    # Debugging & Runtime
    # ==============================
    # Set to True to train on a small subset for quick pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    # Number of workers for data loading
    NUM_WORKERS = 4

    @classmethod
    def create_dirs(cls):
        """Ensures necessary working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


def setup_reproducibility(seed=Config.SEED):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensure working directory exists
    Config.create_dirs()
