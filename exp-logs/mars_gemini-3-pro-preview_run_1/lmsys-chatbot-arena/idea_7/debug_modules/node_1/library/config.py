import os
import torch


class Config:
    """
    Configuration class for the Siamese DeBERTa-v3-Base pipeline.
    Handles paths, hyperparameters, and hardware settings.
    """

    # =========================================
    # Reproducibility & Debugging
    # =========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    DEBUG_SUBSET_SIZE = 1000  # Number of rows to use if DEBUG is True

    # =========================================
    # File Paths
    # =========================================
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Data Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories (Writeable)
    WORKING_DIR = "./working/idea_7"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Output Files
    MODEL_OUTPUT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================
    # Model Configuration
    # =========================================
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LENGTH = 512
    NUM_LABELS = 3  # Target classes: winner_model_a, winner_model_b, winner_tie

    # Architecture Specifics
    HIDDEN_SIZE = 768  # For DeBERTa-v3-base
    NUM_POOLING_LAYERS = 4  # Number of last layers to pool

    # =========================================
    # Training Hyperparameters
    # =========================================
    # Effective Batch Size = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS
    # Goal: Effective BS = 32
    TRAIN_BATCH_SIZE = 8  # Fits in A100 memory with 512 seq len
    VALID_BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 4

    # Learning Rates (Differential)
    LR_BACKBONE = 1e-5
    LR_HEAD = 1e-4

    # Optimization
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    NUM_EPOCHS = 4

    # Regularization
    PATIENCE = 2  # Early stopping patience
    DROPOUT = 0.1

    # =========================================
    # Hardware & Environment
    # =========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # For DataLoaders
    PIN_MEMORY = True

    @classmethod
    def setup(cls):
        """
        Initializes the environment by creating necessary directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
