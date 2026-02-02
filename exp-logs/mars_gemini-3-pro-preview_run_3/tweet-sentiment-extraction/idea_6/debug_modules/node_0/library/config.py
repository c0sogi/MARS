import os
import torch


class Config:
    """
    Central configuration for the Variance-Gated Self-Training pipeline.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use if DEBUG is True

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output directories
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Data Files
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_PATH = "microsoft/deberta-v3-large"
    MAX_LEN = 96

    # Training
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.1
    MAX_GRAD_NORM = 1.0

    # Scheduler
    SCHEDULER_TYPE = "cosine"
    NUM_WARMUP_STEPS_RATIO = 0.1

    # =========================================================================
    # Self-Training / Variance-Gating
    # =========================================================================
    # Threshold for selecting pseudo-labels from the test set based on ensemble consistency
    PSEUDO_LABEL_THRESHOLD = 0.9

    # =========================================================================
    # Hardware / System
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2  # Number of dataloader workers
    PIN_MEMORY = True
