import os
import torch


class Config:
    """
    Configuration class for the Energy-Attentive Isomorphic CNN (Idea 61).
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # =========================================================================
    # Directories and File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory specific to this experiment/idea
    WORKING_DIR = "./working/idea_61"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    # Input channels: HH, HV, and Synthetic Average
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42

    # Cross-Validation
    N_FOLDS = 5

    # Optimization
    BATCH_SIZE = 32
    EPOCHS = 75
    PATIENCE = 12  # Early stopping patience

    # Optimizer settings (AdamW)
    LEARNING_RATE = 1e-3  # Constant learning rate
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # =========================================================================
    # System / Hardware
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup_directories(cls):
        """
        Creates necessary output directories if they don't exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup_directories()
