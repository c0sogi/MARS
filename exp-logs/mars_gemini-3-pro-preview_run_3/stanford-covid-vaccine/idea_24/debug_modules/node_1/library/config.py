import os
import torch


class Config:
    """
    Global configuration for the RNA Degradation Prediction task.
    Implements the settings for the Zero-Masked Non-Linear Channel-Gated BiGRU strategy.
    """

    # ==========================================
    # Paths & Directories
    # ==========================================
    # Input Metadata (Pre-generated Parquet files)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directory (for Caching and Model Checkpoints)
    WORKING_DIR = "./working/idea_24"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LEN = 107
    SEQ_SCORED = 68
    NUM_TARGETS = 5

    # Input Feature Dimensions (Strict One-Hot Encoding)
    # Sequence: A, G, U, C
    DIM_SEQ = 4
    # Structure: (, ), .
    DIM_STRUCT = 3
    # Loop Type: S, M, I, B, H, E, X
    DIM_LOOP = 7

    # Total Input Channels = 4 + 3 + 7 = 14
    INPUT_CHANNELS = DIM_SEQ + DIM_STRUCT + DIM_LOOP

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Convolutional Stem
    STEM_KERNEL_SIZE = 3
    STEM_FILTERS = 256

    # BiGRU Backbone with Zero-Masked Channel-Gating
    HIDDEN_DIM = 384  # Strictly limited to 384 to prevent optimization failure
    NUM_LAYERS = 3  # Number of Block iterations
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 32  # Moderate batch size for stability with graph ops
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Gradient Clipping (Mandatory for Recurrent/Gated architectures)
    MAX_GRAD_NORM = 1.0

    # Scheduler Settings (Cosine Annealing)
    T_MAX = EPOCHS

    # Hardware
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration settings."""
        print("=" * 40)
        print(f"Configuration ({cls.__name__})")
        print("=" * 40)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k:<20}: {v}")
        print("=" * 40)
