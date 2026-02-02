import os
import torch


class Config:
    """
    Global configuration for the Complementary Signal Network (CS-Net) experiment.
    """

    # -------------------------------------------------------------------------
    # Experiment Control & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 100

    # -------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------
    N_FOLDS = 5
    NUM_EPOCHS = 75  # Sufficient epochs for "Low and Slow" convergence
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-4  # Conservative start for Adam
    PATIENCE = 15  # Early stopping patience
    DROPOUT_RATE = 0.2

    # -------------------------------------------------------------------------
    # Data & Model Architecture
    # -------------------------------------------------------------------------
    IMG_WIDTH = 75
    IMG_HEIGHT = 75
    # Input channels: 3 Base (HH, HV, Avg)
    INPUT_CHANNELS = 3

    # Architecture specific: Dual Pooling results in effective doubling
    # Defined here for reference, implemented in model
    FILTER_SIZES = [32, 64, 64, 32]

    # -------------------------------------------------------------------------
    # Compute Resources
    # -------------------------------------------------------------------------
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Directories (Read-Only, Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Write Allowed)
    # Specific folder for this idea iteration
    WORK_DIR = "./working/idea_16"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    MODEL_DIR = WORK_DIR

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    @staticmethod
    def setup_directories():
        """
        Ensures that the working, cache, and submission directories exist.
        """
        os.makedirs(Config.WORK_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup_directories()
