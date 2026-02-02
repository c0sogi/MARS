import os
import torch


class Config:
    # =========================================================================
    # Directory & File Paths
    # =========================================================================
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Read-Only, Pre-split Parquet files)
    METADATA_DIR = "./metadata"
    TRAIN_META = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directory (Write Allowed)
    # Strategy-specific directory for caching and checkpoints
    WORKING_DIR = "./working/idea_44"
    CACHE_DIR = WORKING_DIR
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    SEED = 42
    ID_COL = "Id"
    TARGET_COL = "Cover_Type"

    # Cover_Type contains classes 1-7. We set output dim to 7 to handle
    # the mapping (typically 0-6 internally or masking).
    NUM_CLASSES = 7

    # =========================================================================
    # Model Architecture: Asymmetric Parallel Factorized-DCN-ResNet (Rank-4)
    # =========================================================================
    # General
    HIDDEN_DIM = 512

    # Branch 1: Asymmetric Factorized DCN
    DCN_RANK = 4  # Rank-4 Factorization
    DCN_LAYERS = 3  # Asymmetric Depth (shallower than backbone)
    DCN_INIT_STD = 1e-4  # Warm-Start Initialization (Near-Zero)

    # Branch 2: Deep Full Pre-Activation ResNet
    RESNET_BLOCKS = 4  # 4 Residual Blocks
    RESNET_DROPOUT = 0.2  # Dropout rate within blocks

    # =========================================================================
    # Training Configuration
    # =========================================================================
    BATCH_SIZE = 4096
    EPOCHS = 60

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.1  # Aggressive decay
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MODE = "max"  # Monitoring Accuracy

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # System / Hardware
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CUDNN_DETERMINISTIC = False  # Disabled for performance (Lesson 00070)

    @staticmethod
    def initialize():
        """
        Creates the necessary writable directories for the experiment.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
