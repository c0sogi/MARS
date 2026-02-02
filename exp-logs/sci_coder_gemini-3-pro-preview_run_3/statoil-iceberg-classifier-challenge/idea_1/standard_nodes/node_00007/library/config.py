import os
import torch


class Config:
    # ==========================================
    # File Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    # Cache directory for idea_1 specific intermediate files (e.g. processed numpy arrays)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_3")
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Specifications
    # ==========================================
    IMG_WIDTH = 75
    IMG_HEIGHT = 75
    NUM_BANDS = 2
    # Flattened dimension: 75 * 75 * 2 = 11250
    INPUT_DIM = IMG_WIDTH * IMG_HEIGHT * NUM_BANDS

    # ==========================================
    # Model Architecture (SFCN)
    # ==========================================
    # Hidden layers as specified in the idea description
    HIDDEN_UNITS = [512, 256]
    DROPOUT_RATE = 0.4

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Debugging / Development
    # Set to an integer (e.g., 100) to limit dataset size for quick debugging.
    # Set to None for full training.
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # Reproducibility and Hardware
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Number of workers for DataLoader
    NUM_WORKERS = 2

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
