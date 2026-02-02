import os
import torch


class Config:
    """
    Configuration class for the Max-Attention Selective Hierarchical CNN (MASH-CNN) solution.
    Defines global hyperparameters, file paths, and hardware settings.
    """

    # ---------------------------
    # General Configuration
    # ---------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    MAX_SAMPLES = None  # Number of samples to use if DEBUG is True

    # ---------------------------
    # Directory Paths
    # ---------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_22"
    SUBMISSION_DIR = "./submission"

    # ---------------------------
    # File Paths (Input)
    # ---------------------------
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Paths (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # ---------------------------
    # File Paths (Output)
    # ---------------------------
    # Cache for preprocessed numpy arrays
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Model checkpoints
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Final submission file
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ---------------------------
    # Training Hyperparameters
    # ---------------------------
    NUM_FOLDS = 5
    NUM_EPOCHS = 75
    PATIENCE = 12
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3  # Strong regularization as per strategy

    # ---------------------------
    # Model Architecture
    # ---------------------------
    INPUT_CHANNELS = 3  # Band 1 (HH), Band 2 (HV), Avg ((HH+HV)/2)
    IMAGE_SIZE = 75
    NUM_CLASSES = 1

    # ---------------------------
    # Hardware
    # ---------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    @staticmethod
    def setup_directories():
        """
        Creates necessary output directories if they don't exist.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup_directories()
