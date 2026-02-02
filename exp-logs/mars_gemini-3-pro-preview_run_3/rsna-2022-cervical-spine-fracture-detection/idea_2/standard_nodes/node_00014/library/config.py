import os
import torch
import random
import numpy as np


class Config:
    # ====================================================
    # General Settings
    # ====================================================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20  # Number of samples to use when DEBUG is True

    # ====================================================
    # Paths
    # ====================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Input Data Paths
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Generated Metadata Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Paths
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ====================================================
    # Model & Data Hyperparameters
    # ====================================================
    BACKBONE = "resnet18"
    IMG_SIZE = 256
    IN_CHANNELS = 3  # 2.5D Stacking (z-1, z, z+1)

    # Sequence Modeling (MIL)
    SEQ_LEN = 64  # Number of slices sampled per exam
    N_CLASSES = 7  # C1 to C7

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    EPOCHS = 10
    BATCH_SIZE = 8  # Adjusted for 64 slices * 256x256 on A100
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000

    # Scheduler
    # T_max set to 1.5x total epochs to prevent rapid decay
    T_MAX = int(EPOCHS * 1.5)
    MIN_LR = 1e-6

    # ====================================================
    # Hardware & Logging
    # ====================================================
    NUM_WORKERS = 12
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PRINT_FREQ = 10


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_directories():
    """
    Ensures necessary working and submission directories exist.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Execute setup immediately on import
setup_directories()
seed_everything(Config.SEED)
