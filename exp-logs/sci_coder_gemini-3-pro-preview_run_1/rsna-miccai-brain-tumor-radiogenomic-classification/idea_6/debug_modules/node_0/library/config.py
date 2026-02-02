import os
import torch


class Config:
    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # File Paths
    # ==========================================
    # Read-only input directories
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata paths (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for caching processed data and saving models
    WORKING_DIR = "./working/idea_6"
    CACHE_DIR = WORKING_DIR

    # Output directory for submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    IMG_SIZE = 224
    NUM_CHANNELS = 3  # FLAIR, T1wCE, T2w
    MODALITIES = [
        "FLAIR",
        "T1wCE",
        "T2w",
    ]  # T1w is excluded from the composite as per idea

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MAX_EPOCHS = 30
    PATIENCE = 5  # Early stopping patience
    N_FOLDS = 5
    NUM_WORKERS = 4  # Optimized for the 12 vCPU environment

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50

    @classmethod
    def setup(cls):
        """
        Creates necessary working and submission directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories ensured: {cls.WORKING_DIR}, {cls.SUBMISSION_DIR}")
