import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Central configuration for the Cactus Classification task.
    Defines hyperparameters, file paths, and environment settings.
    """

    # Experiment Identity
    EXPERIMENT_NAME = "idea_9"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100

    # Directory Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Caching Paths (for deterministic data loading)
    CACHE_TRAIN_IMGS = os.path.join(WORKING_DIR, "cache_train_imgs.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "cache_train_labels.npy")
    CACHE_TEST_IMGS = os.path.join(WORKING_DIR, "cache_test_imgs.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "cache_test_ids.npy")

    # Data Dimensions
    IMAGE_SIZE = 32
    NUM_CLASSES = 1
    NUM_CHANNELS = 3

    # Training Hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_FOLDS = 5

    # Regularization
    MIXUP_ALPHA = 0.2

    # Optimization (Sharpness-Aware Minimization)
    USE_SAM = True
    SAM_RHO = 0.05

    # Compute Settings
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initializes the environment by creating necessary directories and setting seeds.
        Should be called at the start of the execution pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        seed_everything(cls.SEED)
