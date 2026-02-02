import os
import torch


class Config:
    """
    Configuration class for the Ship vs. Iceberg classification task.
    Implements the settings for the Dual-Scale Normalized Simple CNN (DSN-CNN).
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SAMPLES = 100  # Number of samples to use in debug mode

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/optimized_simple_cnn"
    SUBMISSION_DIR = "./submission"

    # -------------------------------------------------------------------------
    # Input File Paths
    # -------------------------------------------------------------------------
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # Metadata File Paths (Pre-generated)
    # -------------------------------------------------------------------------
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # -------------------------------------------------------------------------
    # Caching Paths (For deterministic data processing)
    # -------------------------------------------------------------------------
    # These files will be stored in WORKING_DIR to speed up subsequent runs
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "X_train.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "y_train.npy")
    CACHE_TRAIN_ANGLE = os.path.join(WORKING_DIR, "angle_train.npy")

    CACHE_VAL_X = os.path.join(WORKING_DIR, "X_val.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "y_val.npy")
    CACHE_VAL_ANGLE = os.path.join(WORKING_DIR, "angle_val.npy")

    CACHE_TEST_X = os.path.join(WORKING_DIR, "X_test.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "ids_test.npy")
    CACHE_TEST_ANGLE = os.path.join(WORKING_DIR, "angle_test.npy")

    # -------------------------------------------------------------------------
    # Output Paths
    # -------------------------------------------------------------------------
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 75
    # Input channels: Band 1 (HH), Band 2 (HV), Average ((HH+HV)/2)
    NUM_CHANNELS = 3

    # -------------------------------------------------------------------------
    # Model Architecture (DSN-CNN)
    # -------------------------------------------------------------------------
    # Backbone channel widths: 64 -> 128 -> 128 -> 128
    CONV_CHANNELS = [64, 128, 128, 128]
    # Classification head settings
    FC_DIM = 256
    DROPOUT_RATE = 0.2

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    N_FOLDS = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3  # Constant learning rate strategy
    WEIGHT_DECAY = 1e-4  # L2 Regularization to prevent confident errors
    PATIENCE = 10  # Early stopping patience

    # -------------------------------------------------------------------------
    # Hardware / Compute
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Ensures that the necessary working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
