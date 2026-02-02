import os
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_23"
    SUBMISSION_DIR = "./submission"

    # Data Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Paths (Parquet format preferred over pickle)
    TRAIN_PROCESSED_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_PROCESSED_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_PROCESSED_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")
    METADATA_CACHE_PATH = os.path.join(
        WORKING_DIR, "metadata.npy"
    )  # For vocab sizes etc.

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Embedding dimension for all categorical variables
    EMBEDDING_DIM = 16

    # Feature Engineering
    F27_SEQ_LEN = 10  # Length of string in f_27

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000  # Number of samples to use if DEBUG is True

    # ==========================================
    # Model Architecture (AV-PFE)
    # ==========================================
    # The ensemble consists of 5 streams:
    # 0: Anchor 1 (Standard Funnel, Drop 0.20)
    # 1: Anchor 2 (Standard Funnel, Drop 0.20)
    # 2: Capacity Variant (Wide Funnel, Drop 0.25)
    # 3: Aggressive Variant (Standard Funnel, Drop 0.10)
    # 4: Conservative Variant (Standard Funnel, Drop 0.30)

    NUM_STREAMS = 5

    # Stream-specific configurations
    # Format: {'hidden_dims': list, 'dropout': float}
    STREAM_CONFIGS = [
        # Anchor 1
        {"hidden_dims": [512, 256, 128], "dropout": 0.20},
        # Anchor 2
        {"hidden_dims": [512, 256, 128], "dropout": 0.20},
        # Capacity Variant (Wide)
        {"hidden_dims": [1024, 512, 256], "dropout": 0.25},
        # Aggressive Variant (Low Reg)
        {"hidden_dims": [512, 256, 128], "dropout": 0.10},
        # Conservative Variant (High Reg)
        {"hidden_dims": [512, 256, 128], "dropout": 0.30},
    ]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 1024
    EPOCHS = 50

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-2  # max_lr for OneCycleLR
    WEIGHT_DECAY = 1e-4

    # Scheduler
    PCT_START = 0.3  # Percentage of training to increase LR
    DIV_FACTOR = 25.0  # Initial LR = max_lr / div_factor
    FINAL_DIV_FACTOR = 1e4  # Min LR = initial_lr / final_div_factor

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories on import
Config.setup()
