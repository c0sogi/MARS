import os
import torch


class Config:
    """
    Configuration class for the Siamese DeBERTa-Base model pipeline.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of dataloader workers

    # =========================================================================
    # Data Paths
    # =========================================================================
    # Input Metadata (Read-only)
    METADATA_DIR = "./metadata"
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories (Write allowed)
    WORKING_DIR = "./working/idea_4"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create output directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 512
    HIDDEN_SIZE = 768  # Standard for base models
    NUM_LABELS = 3  # Classes: Model A, Model B, Tie
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 2
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Mixed Precision Training
    USE_FP16 = True

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLES = 100
