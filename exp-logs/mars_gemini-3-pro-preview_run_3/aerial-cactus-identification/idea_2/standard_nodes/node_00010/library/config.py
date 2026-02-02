import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    """
    Centralized configuration for the Cactus Identification pipeline.
    """

    # General
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for the available 12 vCPUs

    # Directory Paths
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure mutable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Parameters
    IMAGE_SIZE = (32, 32)
    NUM_CLASSES = 1

    # Training Hyperparameters
    BATCH_SIZE = 128
    NUM_EPOCHS = 35
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 7  # Early stopping patience

    # Scheduler Parameters (Cosine Annealing)
    T_MAX = NUM_EPOCHS
    ETA_MIN = 1e-6

    # Augmentation Parameters
    # Exploiting aerial perspective (rotation/flips) and lighting invariance (color jitter)
    AUG_ROTATION_DEGREES = 30
    AUG_COLOR_JITTER_BRIGHTNESS = 0.2
    AUG_COLOR_JITTER_CONTRAST = 0.2
    AUG_COLOR_JITTER_SATURATION = 0.2
    AUG_COLOR_JITTER_HUE = 0.1
    AUG_HFLIP_PROB = 0.5
    AUG_VFLIP_PROB = 0.5

    # Mixup
    USE_MIXUP = True
    MIXUP_ALPHA = 1.0

    # Inference
    USE_TTA = True  # Use Test-Time Augmentation
