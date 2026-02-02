import os
import torch


class Config:
    """
    Configuration class for the Tweet Sentiment Extraction task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SIZE = 500  # Number of samples to use when DEBUG is True

    # =========================================================================
    # System & Hardware
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Data
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Using generated metadata files as per instructions
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output & Working Directories
    # Using 'idea_9' as the designated working directory for this iteration
    WORKING_DIR = "./working/idea_9"
    CACHE_DIR = WORKING_DIR  # Directory for caching processed data (numpy/parquet)

    # Model Artifacts
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.bin")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Model Architecture & Tokenizer
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-base"

    # =========================================================================
    # Hyperparameters
    # =========================================================================
    MAX_LEN = 128  # Maximum sequence length (covers max tweet length safely)
    TRAIN_BATCH_SIZE = 32  # Batch size for training
    VALID_BATCH_SIZE = 64  # Batch size for validation
    EPOCHS = 5  # Number of training epochs
    LEARNING_RATE = 2e-5  # Learning rate for the optimizer
    WEIGHT_DECAY = 0.01  # Weight decay for AdamW
    SIGMA = 1.0  # Standard deviation for Gaussian label smoothing
    DROPOUT = 0.1  # Dropout probability
    SCHEDULER = "linear"  # Learning rate scheduler type
    WARMUP_RATIO = 0.1  # Ratio of total steps for warmup

    @classmethod
    def setup(cls):
        """
        Create necessary directories if they don't exist.
        This ensures the caching and submission paths are valid.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup on import to ensure environment is ready
Config.setup()
