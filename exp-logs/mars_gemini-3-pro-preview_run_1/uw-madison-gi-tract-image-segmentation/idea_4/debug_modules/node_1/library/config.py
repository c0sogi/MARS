import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Stomach and Intestines MRI Segmentation task.
    Centralizes hyperparameters, file paths, and model settings.
    """

    # ==========================
    # General Settings
    # ==========================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    EXP_NAME = "idea_4"

    # ==========================
    # Directories
    # ==========================
    # Input data (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Write Allowed)
    WORKING_DIR = os.path.join("./working", EXP_NAME)
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # ==========================
    # Data Preprocessing
    # ==========================
    # Image resolution: 320x320 (Balanced for U-Net++ decoder cost)
    IMG_SIZE = (320, 320)

    # Input channels: 3
    # Strategy: 2D Single-Slice (z) replicated to (z, z, z) to fit ResNet backbone
    IN_CHANNELS = 3

    # Segmentation Classes
    CLASSES = ["large_bowel", "small_bowel", "stomach"]
    NUM_CLASSES = 3

    # Normalization
    # Robust Percentile Normalization settings
    NORM_MIN_PERCENTILE = 1.0
    NORM_MAX_PERCENTILE = 99.0

    # ==========================
    # Model Architecture
    # ==========================
    ARCH = "UnetPlusPlus"  # Nested U-Net for better boundary detection
    BACKBONE = "resnet34"  # Moderate capacity, stable convergence
    ENCODER_WEIGHTS = "imagenet"
    DEEP_SUPERVISION = True  # Enable output from intermediate decoder nodes

    # ==========================
    # Training Hyperparameters
    # ==========================
    BATCH_SIZE = 32
    EPOCHS = 15

    # Optimization
    LR = 2e-4
    WEIGHT_DECAY = 1e-2
    MIN_LR = 1e-6
    SCHEDULER = "CosineAnnealingLR"
    T_MAX = EPOCHS  # For CosineAnnealing

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MODE = "max"  # Monitoring Dice score (maximize)

    # Loss Weights (Deep Supervision requires summing losses)
    # Combined Loss = BCE_WEIGHT * BCE + DICE_WEIGHT * Dice
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5

    # ==========================
    # Hardware
    # ==========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Using 12 vCPUs available
    NUM_WORKERS = 12
    PIN_MEMORY = True

    @classmethod
    def setup_directories(cls):
        """
        Creates the necessary working directories if they don't exist.
        """
        for d in [
            cls.WORKING_DIR,
            cls.CHECKPOINT_DIR,
            cls.PREDICTION_DIR,
            cls.SUBMISSION_DIR,
        ]:
            os.makedirs(d, exist_ok=True)
        print(f"Directories created/verified at {cls.WORKING_DIR}")


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
    print(f"Random seed set to {seed}")
