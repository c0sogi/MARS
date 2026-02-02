import os
import torch


class Config:
    """
    Configuration class for the Tweet Sentiment Extraction task.
    Defines model parameters, file paths, and training hyperparameters.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging: Set to True to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # =========================================================================
    # Data Paths
    # =========================================================================
    # Using metadata files as per instructions
    TRAIN_FILE = "./metadata/train.csv"
    VAL_FILE = "./metadata/val.csv"
    TEST_FILE = "./metadata/test.csv"
    SAMPLE_SUBMISSION = "./input/sample_submission.csv"

    # =========================================================================
    # Output Directories
    # =========================================================================
    # Working directory for idea_4
    BASE_OUTPUT_DIR = "./working/idea_4"

    # Ensure base output directory exists
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Path to save trained models
    MODEL_OUTPUT_DIR = BASE_OUTPUT_DIR

    # Path for caching processed data (numpy/parquet)
    CACHE_DIR = os.path.join(BASE_OUTPUT_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-large"

    # =========================================================================
    # Tokenizer & Input
    # =========================================================================
    # Max length for tokenized sequences.
    # Tweets are short (<141 chars), so 128 tokens is sufficient.
    MAX_LEN = 128

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16
    EPOCHS = 3

    # Optimizer
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Scheduler
    WARMUP_RATIO = 0.1

    # Loss Function
    # Label smoothing to prevent overfitting to noisy boundaries
    LABEL_SMOOTHING = 0.1

    # =========================================================================
    # Inference / Post-processing
    # =========================================================================
    # Heuristic: If sentiment is neutral, predict the full text.
    # Analysis shows Jaccard ~0.98 for neutral tweets with full text.
    NEUTRAL_FULL_TEXT = True
