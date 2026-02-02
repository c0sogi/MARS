import os
import random
import numpy as np
import torch


class Config:
    """
    Central configuration for the Denoising project (Idea 3: Attention U-Net + Bagging).
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    PROJECT_NAME = "idea_3"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    PATCH_SIZE = 128  # Size of random crops for training
    NUM_WORKERS = 4  # Number of data loading workers

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 16  # Small batch size as per strategy
    LEARNING_RATE = 1e-3  # High learning rate for velocity
    NUM_EPOCHS = 1000  # High max epochs, relying on early stopping
    EARLY_STOPPING_PATIENCE = 20

    # Scheduler Settings (Cosine Annealing)
    COSINE_T_MAX = 1000  # Decoupled horizon for scheduler

    # =========================================================================
    # Ensemble Strategy
    # =========================================================================
    NUM_MODELS = 5  # Number of independent models for Bagging

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def seed_everything(seed=Config.SEED):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Ensure directories exist upon import
Config.setup()
