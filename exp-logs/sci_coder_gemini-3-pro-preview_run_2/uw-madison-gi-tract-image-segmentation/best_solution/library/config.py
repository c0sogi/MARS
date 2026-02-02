import os
import random
import numpy as np
import torch


class Config:
    # ==============================
    # Directories and Paths
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Ensure working and cache directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================
    # Data Configuration
    # ==============================
    IMG_SIZE = (256, 256)
    IN_CHANNELS = 3  # 2.5D Input: Slice i-1, i, i+1
    NUM_CLASSES = 3
    CLASSES = ["large_bowel", "small_bowel", "stomach"]
    SLICE_THICKNESS = 3.0  # mm

    # Data Loading
    NUM_WORKERS = 12
    PIN_MEMORY = True

    # Dataset Control
    DEBUG = False
    DATA_FRACTION = 1.0  # Percentage of data to use (1.0 = 100%)

    # ==============================
    # Training Hyperparameters
    # ==============================
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # For AdamW
    T_MAX = 15  # For CosineAnnealingLR (matches EPOCHS)
    ETA_MIN = 1e-6

    # ==============================
    # Model Architecture
    # ==============================
    ENCODER_NAME = "ghostnet_100"

    # ==============================
    # Reproducibility
    # ==============================
    SEED = 42

    @staticmethod
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

    @classmethod
    def set_debug_mode(cls, debug=True):
        """
        Adjusts configuration for debugging purposes.
        Reduces epochs, dataset size, and batch size for rapid iteration.
        """
        cls.DEBUG = debug
        if debug:
            cls.EPOCHS = 2
            cls.DATA_FRACTION = 0.05  # Use 5% of data
            cls.BATCH_SIZE = 8
            print(
                f"Debug mode enabled: Epochs={cls.EPOCHS}, Data Fraction={cls.DATA_FRACTION}"
            )
