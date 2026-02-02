import os
import torch


class Config:
    """
    Configuration class for the Reparameterized Coordinate ResUNet (Rep-CResUNet-SR) experiment.
    """

    # --------------------------
    # Reproducibility
    # --------------------------
    SEED = 42

    # --------------------------
    # File Paths & Directories
    # --------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Experiment specific working directory
    WORKING_DIR = "./working/idea_15"

    # Sub-directories for caching and checkpoints
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Metadata file paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Model Output Paths
    # Using the specific name for the idea
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "rep_cresunet_best.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------
    # Data Parameters
    # --------------------------
    PATCH_SIZE = 128
    PATCHES_PER_IMAGE = 100
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # --------------------------
    # Model Architecture
    # --------------------------
    BASE_FILTERS = 64

    # --------------------------
    # Training Hyperparameters
    # --------------------------
    EPOCHS = 100
    BATCH_SIZE = 32  # Adjusted for A100 40GB and 128x128 patches
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Aggressive regularization as per strategy

    # --------------------------
    # System
    # --------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure for the experiment.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
