import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for python, numpy, and torch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Centralized configuration for the denoising task.
    Includes paths, data parameters, model hyperparameters, and hardware settings.
    """

    # --- Reproducibility ---
    SEED = 42

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching data and saving models
    # Using 'idea_2' as per the selected idea (U-Net)
    WORKING_DIR = "./working/idea_2"

    # Submission directory and path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Path to save the best model checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "unet_best.pth")

    # --- Data Parameters ---
    # Patch size for training (random crops)
    PATCH_SIZE = 256

    # Input and Output channels (Grayscale = 1)
    IN_CHANNELS = 1
    OUT_CHANNELS = 1

    # --- Training Hyperparameters ---
    # Batch size (A100 40GB can handle larger batches)
    BATCH_SIZE = 16

    # Number of training epochs
    NUM_EPOCHS = 100

    # Learning rate for Adam optimizer
    LEARNING_RATE = 1e-4

    # Weight decay for regularization
    WEIGHT_DECAY = 1e-5

    # Early stopping patience (epochs without improvement)
    EARLY_STOPPING_PATIENCE = 10

    # --- Hardware ---
    # Number of data loading workers
    NUM_WORKERS = 4

    # Device selection
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Debugging / Development ---
    # If set to an integer, limits the number of samples used for training/validation
    # Set to None for full training
    DEBUG_SAMPLE_SIZE = None

    @classmethod
    def setup(cls):
        """
        Initializes the environment by creating necessary directories
        and setting the random seed.
        """
        # Create working and submission directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set random seed
        seed_everything(cls.SEED)
