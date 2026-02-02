import os
import torch


class Config:
    """
    Global configuration for the Sentiment Analysis Tweet Extraction task.
    Contains hyperparameters, file paths, and model settings based on the
    optimized strategy using DeBERTa-v3-Large with Multi-Sample Dropout.
    """

    # ====================================================
    # General Settings
    # ====================================================
    SEED = 42
    DEBUG = False  # Set to True for fast debugging on a subset
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use in debug mode
    NUM_WORKERS = 4  # Number of CPU workers for dataloaders

    # ====================================================
    # File Paths
    # ====================================================
    INPUT_DIR = "./input"
    WORKING_DIR = "./working"

    # Idea-specific output directory
    IDEA_NAME = "idea_7"
    OUTPUT_DIR = os.path.join(WORKING_DIR, IDEA_NAME)

    # Full Dataset Paths (for 5-Fold CV on Full Data)
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Paths (Pre-split 80/20, for reference/validation if needed)
    META_TRAIN_CSV = "./metadata/train.csv"
    META_VAL_CSV = "./metadata/val.csv"
    META_TEST_CSV = "./metadata/test.csv"

    # Final Submission Path
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.csv")

    # ====================================================
    # Model Architecture
    # ====================================================
    MODEL_PATH = "microsoft/deberta-v3-large"
    TOKENIZER_PATH = "microsoft/deberta-v3-large"

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 16  # Fits on A100 (40GB) with AMP
    VALID_BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-5

    # Optimization
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    SCHEDULER_TYPE = "cosine"  # Learning rate scheduler
    WARMUP_RATIO = 0.1

    # ====================================================
    # Strategy Specifics
    # ====================================================
    # Mixed Precision Training
    USE_AMP = True

    # Cross-Validation
    N_FOLDS = 5

    # Multi-Sample Dropout (Internal Ensembling)
    USE_MSD = True
    MSD_DROPOUT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Hybrid Loss Function
    LABEL_SMOOTHING = 0.1
    JACCARD_LOSS_WEIGHT = 0.5  # Weight for the Soft Jaccard component

    # ====================================================
    # Hardware
    # ====================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """Creates the output directory if it does not exist."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)


# Initialize environment
Config.setup()
