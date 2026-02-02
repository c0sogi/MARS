import os
import torch


class Config:
    """
    Configuration class for the SDHA-ResNet pipeline.
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Toggle to True to use a small subset of data for debugging
    DEBUG_SUBSET_SIZE = 100

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_34"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Caching Paths (for deterministic data processing)
    # -------------------------------------------------------------------------
    # These paths are used by the data loader to save/load processed numpy arrays
    CACHE_PATHS = {
        "X_train": os.path.join(WORKING_DIR, "X_train.npy"),
        "y_train": os.path.join(WORKING_DIR, "y_train.npy"),
        "angle_train": os.path.join(WORKING_DIR, "angle_train.npy"),
        "X_val": os.path.join(WORKING_DIR, "X_val.npy"),
        "y_val": os.path.join(WORKING_DIR, "y_val.npy"),
        "angle_val": os.path.join(WORKING_DIR, "angle_val.npy"),
        "X_test": os.path.join(WORKING_DIR, "X_test.npy"),
        "ids_test": os.path.join(WORKING_DIR, "ids_test.npy"),
        "angle_test": os.path.join(WORKING_DIR, "angle_test.npy"),
    }

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    IN_CHANNELS = 3  # HH, HV, and Average((HH+HV)/2)

    # -------------------------------------------------------------------------
    # Model Hyperparameters (SDHA-ResNet)
    # -------------------------------------------------------------------------
    MODEL_NAME = "SDHA_ResNet"
    STAGES = 4
    CHANNEL_WIDTHS = [64, 128, 128, 128]
    DROP_PATH_RATE = 0.2  # Max probability for Stochastic Depth (linearly increases)
    LEAKY_RELU_SLOPE = 0.1
    HEAD_DROPOUT = 0.5
    USE_BIAS = True  # Retain bias in convolutions

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    NUM_FOLDS = 5
    BATCH_SIZE = 32
    EPOCHS = 75
    PATIENCE = 12
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # L2 Regularization for AdamW

    # Compute
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
