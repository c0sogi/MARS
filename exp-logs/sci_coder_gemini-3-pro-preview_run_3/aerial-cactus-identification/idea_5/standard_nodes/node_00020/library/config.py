import os
import torch
import numpy as np
import random


class Config:
    """
    Central configuration for the Cactus Identification Task.
    Implements settings for the Triple-Architecture Heterogeneous Stacking Ensemble.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False
    # If Debug is True, limit dataset size to this number for quick testing
    DEBUG_SAMPLE_SIZE = 500

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching processed data and model checkpoints
    WORK_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    IMG_HEIGHT = 32
    IMG_WIDTH = 32
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    NUM_CLASSES = 1
    NUM_FOLDS = 5

    # ==========================================
    # Compute Environment
    # ==========================================
    # Use 4 workers as 12 vCPUs are available (safe buffer)
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Extended epochs and patience for Mixup convergence
    EPOCHS = 150
    BATCH_SIZE = 128

    # Optimization (AdamW settings)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Regularization
    PATIENCE = 35
    MIXUP_ALPHA = 1.0

    # ==========================================
    # Model Architecture
    # ==========================================
    # The four orthogonal architectures for the ensemble
    MODEL_ARCHS = ["resnet", "densenet", "efficientnet", "mobilenet"]

    @classmethod
    def setup(cls, seed=None):
        """
        Initialize the environment:
        1. Create necessary working directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seed
        if seed is None:
            seed = cls.SEED
        cls.seed_everything(seed)

    @staticmethod
    def seed_everything(seed):
        """
        Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
        Also configures cuDNN for deterministic behavior where possible,
        while enabling benchmark for hardware optimization on fixed input sizes.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Hardware Optimization:
        # benchmark=True allows cuDNN to find the best algorithm for fixed input sizes (32x32)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
