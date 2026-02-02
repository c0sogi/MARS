import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the Breast Cancer Detection pipeline.
    Handles file paths, hyperparameters, and global settings.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Output Directories
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
    SUBMISSION_DIR = "./submission"

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "metadata_embedded_cnn.pth")

    # Ensure necessary writeable directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Processing & Input Configuration
    # =========================================================================
    IMG_SIZE = (512, 512)

    # Input Channels
    # 1 Channel for Grayscale Mammogram + N Channels for Metadata Maps
    NUM_IMG_CHANNELS = 1
    METADATA_COLS = ["age", "implant"]
    NUM_METADATA_CHANNELS = len(METADATA_COLS)

    # Total input channels for the modified first layer of the CNN
    TOTAL_INPUT_CHANNELS = NUM_IMG_CHANNELS + NUM_METADATA_CHANNELS

    # Debugging / Development
    # Set to a specific number (e.g., 1000) to limit dataset size for quick testing
    # Set to None for full training
    SAMPLE_SIZE = None

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "efficientnet_b0"
    NUM_CLASSES = 1  # Binary classification (Cancer vs No Cancer)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    EPOCHS = 5
    BATCH_SIZE = 32

    # Optimizer
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Loss Function (Class Imbalance Handling)
    # Target imbalance is ~1:47. Reduced pos_weight to improve probability calibration for pF1.
    POS_WEIGHT = 5.0

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR

    # =========================================================================
    # Hardware & Compute
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def set_seed(seed=42):
        """
        Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)


# Initialize seed on import
Config.set_seed(Config.SEED)
