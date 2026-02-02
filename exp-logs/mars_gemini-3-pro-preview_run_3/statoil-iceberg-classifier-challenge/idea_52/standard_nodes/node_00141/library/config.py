import os
import random
import numpy as np
import torch


class Config:
    # Experiment Identifier
    EXPERIMENT_NAME = "idea_52"

    # -----------------------
    # File Paths
    # -----------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Splits
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    # Caching processed arrays to avoid re-reading JSONs repeatedly
    CACHE_DIR = os.path.join(WORKING_DIR, EXPERIMENT_NAME)
    # Checkpoints for model weights
    CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints")
    # Final submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -----------------------
    # Model Hyperparameters
    # -----------------------
    # Input: HH, HV, Avg(HH,HV), Diff(HH,HV) -> 4 Channels
    INPUT_CHANNELS = 4
    IMAGE_SIZE = 75
    NUM_CLASSES = 1

    # -----------------------
    # Training Hyperparameters
    # -----------------------
    SEED = 42
    NUM_FOLDS = 5
    BATCH_SIZE = 64
    NUM_EPOCHS = 75
    PATIENCE = 12
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # -----------------------
    # Compute / Hardware
    # -----------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------
    # Debugging
    # -----------------------
    # Set to True to train on a small subset for quick pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    @staticmethod
    def setup():
        """
        Creates the necessary directory structure for the experiment.
        Should be called at the start of the pipeline.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
