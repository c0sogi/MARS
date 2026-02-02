import os
import random
import numpy as np
import torch
from pathlib import Path


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


class Config:
    """
    Central configuration for the Ink Detection task.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = Path("./input")
    METADATA_DIR = Path("./metadata")
    WORKING_DIR = Path("./working")

    # Cache directory for deterministic data processing (Idea 2)
    CACHE_DIR = WORKING_DIR / "idea_2"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata files
    TRAIN_METADATA_PATH = METADATA_DIR / "train.csv"
    VAL_METADATA_PATH = METADATA_DIR / "val.csv"
    TEST_METADATA_PATH = METADATA_DIR / "test.csv"

    # Output paths
    MODEL_FILENAME = "best_model.pth"
    MODEL_PATH = CACHE_DIR / MODEL_FILENAME
    SUBMISSION_PATH = Path("submission.csv")

    # =========================================================================
    # Data Hyperparameters
    # =========================================================================
    Z_DIM = 65  # Depth of the 3D volume
    TILE_SIZE = 256  # Height/Width of input patches
    STRIDE = 128  # 50% overlap for validation/inference tiling
    NUM_WORKERS = 4  # Number of DataLoader workers

    # Normalization stats (Global mean/std calculated from EDA or set roughly)
    # Using approx values from EDA: Mean ~100, Std ~12.5
    NORM_MEAN = 100.0
    NORM_STD = 12.5

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Lean 2.5D U-Net Architecture
    ENCODER_CHANNELS = [32, 64, 64]
    EMA_DECAY = 0.99  # Exponential Moving Average decay for model weights

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 5

    # Debugging / Quick Iteration
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use if DEBUG is True

    # =========================================================================
    # Loss Hyperparameters (Tversky Loss)
    # =========================================================================
    # F0.5 score weights precision higher.
    # We penalize False Positives more heavily: alpha (FP) = 0.3, beta (FN) = 0.7
    # Note: Some implementations define alpha/beta differently.
    # Here we assume: Loss = 1 - (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    TVERSKY_ALPHA = 0.3
    TVERSKY_BETA = 0.7
    TVERSKY_SMOOTH = 1e-6

    # =========================================================================
    # Inference Hyperparameters
    # =========================================================================
    THRESHOLD = 0.5  # Probability threshold for binary mask
    TTA_STEPS = 8  # Test Time Augmentation steps (e.g., 4 rot * 2 flips)

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
