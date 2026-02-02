import os
import torch


class Config:
    """
    Configuration class for the Hindi/Tamil Question Answering Task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"

    # Ensure the working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Data Files (using metadata splits to prevent leakage)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Using MuRIL (Multilingual Representations for Indian Languages)
    MODEL_NAME = "google/muril-base-cased"

    # =========================================================================
    # Tokenizer & Data Processing
    # =========================================================================
    MAX_LEN = 384  # Maximum sequence length for the model
    DOC_STRIDE = 128  # Overlap size for sliding window strategy

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 3  # Number of folds for Group K-Fold Cross-Validation
    EPOCHS = 4  # Number of training epochs
    TRAIN_BATCH_SIZE = 16  # Batch size for training
    EVAL_BATCH_SIZE = 32  # Batch size for validation/inference

    LEARNING_RATE = 2e-5  # Initial learning rate
    WEIGHT_DECAY = 0.01  # Weight decay for AdamW
    WARMUP_RATIO = 0.1  # Ratio of total steps for linear warmup
    MAX_GRAD_NORM = 1.0  # Gradient clipping value

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2  # Number of subprocesses for data loading
