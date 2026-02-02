import os
import torch


class Config:
    """
    Central configuration for the Tweet Sentiment Extraction task.
    Defines model hyperparameters, file paths, and training settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # =========================================================================
    # Data Paths
    # =========================================================================
    # Using metadata files as requested to prevent data leakage and ensure correct splits
    TRAINING_FILE = "./metadata/train.csv"
    VALIDATION_FILE = "./metadata/val.csv"
    TEST_FILE = "./metadata/test.csv"
    SAMPLE_SUBMISSION_FILE = "./input/sample_submission.csv"

    # =========================================================================
    # Output Paths
    # =========================================================================
    WORKING_DIR = "./working/idea_9"

    # Directory for saving trained model weights (per fold)
    OUTPUT_MODEL_DIR = os.path.join(WORKING_DIR, "models")

    # Directory for caching processed data (numpy/parquet)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Directory for final submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Model Architecture & Tokenizer
    # =========================================================================
    MODEL_PATH = "microsoft/deberta-v3-large"
    TOKENIZER_PATH = "microsoft/deberta-v3-large"

    # =========================================================================
    # Hyperparameters
    # =========================================================================
    MAX_LEN = 128  # Sufficient for max char length of ~141

    # Batch sizes optimized for A100 40GB
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16

    EPOCHS = 5
    LEARNING_RATE = 1e-5

    # Optimizer settings
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Loss settings
    LABEL_SMOOTHING = 0.1

    # Scheduler settings
    WARMUP_RATIO = 0.1

    # =========================================================================
    # Debugging & Control
    # =========================================================================
    # Set DEBUG to True to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # Frequency of logging during training
    PRINT_FREQ = 50

    @staticmethod
    def setup_directories():
        """
        Ensures that all necessary output directories exist.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_MODEL_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup_directories()
