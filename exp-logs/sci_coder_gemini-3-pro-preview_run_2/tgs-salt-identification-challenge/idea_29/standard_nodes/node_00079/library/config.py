import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Salt Segmentation Task.
    Centralizes hyperparameters, file paths, constants, and augmentation settings.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (generated in previous steps)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    DEPTHS_CSV = os.path.join(INPUT_ROOT, "depths.csv")

    # Working directory for caching processed data and model checkpoints
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_29")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary writeable directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Parameters
    # =========================================================================
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Padded size for U-Net divisibility (32)
    CHANNELS = 1  # Input channels (Grayscale)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4

    # Epochs for different stages of the pipeline
    NUM_EPOCHS = 50
    NUM_EPOCHS_TEACHER = 50  # Stage 1: Privileged Teacher
    NUM_EPOCHS_STUDENT = 30  # Stage 2: Distillation
    NUM_EPOCHS_FINAL = 30  # Stage 3: Self-training

    # Optimization
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10

    # Loss Weights for Multi-Task Student (Stage 2 & 3)
    LOSS_WEIGHT_SEG = 1.0  # Segmentation Loss (Lovasz + BCE)
    LOSS_WEIGHT_MSE = 0.1  # Depth Regression Auxiliary Loss
    LOSS_WEIGHT_BCE_DISTILL = 0.5  # Distillation Loss (Teacher Soft Targets)

    # =========================================================================
    # Augmentation Parameters
    # =========================================================================
    # Elastic Transform settings (Non-rigid)
    AUG_ELASTIC_P = 0.2
    AUG_ELASTIC_ALPHA = 120.0
    AUG_ELASTIC_SIGMA = 6.0

    # Geometric settings (Rigid)
    AUG_SHIFT_SCALE_ROTATE_P = 0.2

    # =========================================================================
    # Model Architecture
    # =========================================================================
    ENCODER_NAME = "resnet34"
    ENCODER_WEIGHTS = "imagenet"

    # =========================================================================
    # Compute & Debugging
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging flags to control dataset size for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use if DEBUG is True


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

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


# Initialize seeding on import to ensure consistency across all modules
seed_everything(Config.SEED)
