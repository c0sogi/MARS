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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # --------------------------------------------------------------------------
    # General Configuration
    # --------------------------------------------------------------------------
    PROJECT_NAME = "idea_61"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and model checkpoints
    WORKING_DIR = f"./working/{PROJECT_NAME}"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata file paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # --------------------------------------------------------------------------
    # Data Processing & Augmentation
    # --------------------------------------------------------------------------
    # Image Generation: Fixed Overlapping Orthogonal Tri-Slabs
    IMAGE_SIZE = 224  # Native resolution for EfficientNet-B0
    NUM_SLABS = 3  # Number of slabs per view (mapped to RGB channels)
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Tabular Features
    # Note: 'Weeks' is strictly excluded from input features as per solution design
    TABULAR_COLS = ["Age", "Sex", "SmokingStatus", "Percent"]

    # --------------------------------------------------------------------------
    # Model Architecture: NBC-SLN
    # --------------------------------------------------------------------------
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_PRETRAINED = True

    # Dimensions
    BACKBONE_DIM = 1280  # Output dimension of EfficientNet-B0 (GAP)
    LATENT_DIM = 128  # Shared latent vector dimension
    HIDDEN_DIM = 256  # Dimension for intermediate mixing layers

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    EPOCHS = 50
    BATCH_SIZE = 32  # A100 40GB can handle this easily
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 8  # Early stopping patience
    NUM_WORKERS = 4  # Number of dataloader workers

    # --------------------------------------------------------------------------
    # Metric & Loss Constants
    # --------------------------------------------------------------------------
    # Modified Laplace Log Likelihood parameters
    CONFIDENCE_MIN = 70.0  # Minimum clipped confidence (sigma)
    ERROR_MAX = 1000.0  # Maximum clipped absolute error

    # --------------------------------------------------------------------------
    # Hardware
    # --------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """
        Creates necessary directories and sets random seeds.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        seed_everything(cls.SEED)


# Automatically setup environment when module is imported
Config.setup()
