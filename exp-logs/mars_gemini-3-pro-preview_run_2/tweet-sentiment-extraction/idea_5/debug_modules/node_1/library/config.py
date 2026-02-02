import os
import torch


class Config:
    """
    Configuration class for the Tweet Sentiment Extraction task.
    Defines hyperparameters, file paths, and model settings.
    """

    # ====================================================
    # General Settings
    # ====================================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================
    # Directories & Paths
    # ====================================================
    # Input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output directories
    WORKING_DIR = "./working"
    OUTPUT_DIR = "./working/idea_5"
    CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Data Files (using metadata splits to prevent leakage)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ====================================================
    # Model Architecture
    # ====================================================
    # Using DeBERTa-v3-large as the backbone
    BERT_PATH = "microsoft/deberta-v3-large"
    TOKENIZER_PATH = "microsoft/deberta-v3-large"

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 8  # Adjusted for A100 memory with Large model
    VALID_BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Regularization
    LABEL_SMOOTHING = 0.1
    DROPOUT = 0.1

    # Scheduler
    WARMUP_RATIO = 0.1

    # ====================================================
    # Cross Validation
    # ====================================================
    N_FOLDS = 5

    # ====================================================
    # Debugging & Development
    # ====================================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True
