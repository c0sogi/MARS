import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for the Ship vs Iceberg classification task.
    Implements settings for the Delayed-Integration Dual-Pyramid Network (DIDP-Net).
    """

    # ==========================================
    # DIRECTORY & FILE PATHS
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_28"
    SUBMISSION_DIR = "./submission"

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Split Files
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Artifacts
    CACHE_PATH = os.path.join(WORK_DIR, "processed_data.npz")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_PATH_TEMPLATE = os.path.join(WORK_DIR, "didp_net_fold_{}.pth")

    # ==========================================
    # DATA HYPERPARAMETERS
    # ==========================================
    IMAGE_SIZE = 75
    # Input Channels: Band 1, Band 2, Mean(B1, B2)
    IN_CHANNELS = 3

    # ==========================================
    # MODEL HYPERPARAMETERS (DIDP-Net)
    # ==========================================
    BACKBONE_FILTERS = 128
    DROPOUT_RATE = 0.5
    NUM_CLASSES = 1  # Binary classification

    # ==========================================
    # TRAINING HYPERPARAMETERS
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 100  # Upper bound, controlled by Early Stopping
    LEARNING_RATE = 2e-4

    # Optimization Strategy
    PATIENCE = 12  # Early stopping patience
    SCHEDULER_PATIENCE = 4  # ReduceLROnPlateau patience
    SCHEDULER_FACTOR = 0.5  # Decay factor
    MIN_LR = 1e-6  # Minimum learning rate

    # ==========================================
    # SYSTEM CONFIGURATION
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def setup(cls):
        """
        Ensures necessary working and output directories exist.
        """
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
