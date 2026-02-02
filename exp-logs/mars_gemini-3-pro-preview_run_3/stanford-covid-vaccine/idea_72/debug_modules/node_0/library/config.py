import os
import torch


class Config:
    # ==========================================
    # Data & Problem Dimensions
    # ==========================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Features:
    # 4 (One-hot Nucleotide: A, G, C, U)
    # + 3 (One-hot Structure: (, ), .)
    # + 7 (One-hot Loop Type: S, M, I, B, H, E, X)
    INPUT_CHANNELS = 14

    # Targets
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # ==========================================
    # Model Architecture (High-Capacity FFN-Enhanced GLU-BiGRU)
    # ==========================================
    # Convolutional Stem
    CNN_FILTERS = 256
    CNN_KERNEL = 3

    # Recurrent Backbone
    # Hidden size is per direction.
    # Bidirectional output size = HIDDEN_DIM * 2 = 768.
    HIDDEN_DIM = 384
    NUM_LAYERS = 4

    # Pointwise Feed-Forward Network (FFN)
    # Expands the 768-dim backbone output to 1536 before projecting back
    FFN_DIM = 1536

    # Regularization
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    EPOCHS = 25
    PATIENCE = 5  # For Early Stopping

    # LR Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR

    # ==========================================
    # System & Paths
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # Directory Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_72"
    SUBMISSION_DIR = "./submission"

    # File Paths
    TRAIN_PARQUET = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PARQUET = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PARQUET = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    @staticmethod
    def setup_directories():
        """Ensures that the working and submission directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
