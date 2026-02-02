import os
import torch


class Config:
    """
    Configuration class for the Ship vs Iceberg classification task.
    Centralizes all hyperparameters, file paths, and system settings.
    """

    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory specific to Idea 60
    WORK_DIR = "./working/idea_60"

    # Sub-directories for caching processed data and saving models
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Specifications
    # --------------------------------------------------------------------------
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    # Input channels: Band 1 (HH), Band 2 (HV), and Average (Band 1 + Band 2) / 2
    CHANNELS = 3

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 75
    PATIENCE = 12
    NUM_FOLDS = 5

    # Regularization
    # Using L2 Regularization (Weight Decay) as specified in the idea
    WEIGHT_DECAY = 1e-4

    # --------------------------------------------------------------------------
    # System / Compute
    # --------------------------------------------------------------------------
    # Number of workers for DataLoader
    NUM_WORKERS = 4

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure for the project.
        This ensures that cache, checkpoint, and submission directories exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically create directories when the config module is imported
Config.setup()
