import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for Salt Segmentation Task.
    Implements the 'High-Fidelity Training Regime' settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on available vCPUs (12 available)

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_METADATA = "./metadata/train_metadata.csv"
    VAL_METADATA = "./metadata/val_metadata.csv"
    TEST_METADATA = "./metadata/test_metadata.csv"
    DEPTHS_CSV = "./input/depths.csv"

    # Working Directory (Write Access)
    # Using 'idea_7' as the specific workspace for this iteration
    WORKING_DIR = "./working/idea_7"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # Best model path
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    ORIG_SIZE = 101  # Original image size
    IMG_SIZE = 128  # Padded size for training (multiple of 32 for U-Net)
    IN_CHANNELS = 3  # [Seismic, Seismic, Depth]
    NUM_CLASSES = 1  # Binary segmentation (Salt vs Sediment)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "UnetPlusPlus"
    ENCODER = "resnext50_32x4d"
    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = "sigmoid"
    DEEP_SUPERVISION = True  # Enable deep supervision for U-Net++

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64  # Scaled up due to AMP
    EPOCHS = 80  # Extended training duration
    LEARNING_RATE = 1e-4  # Starting LR for ResNeXt backbone
    WEIGHT_DECAY = 1e-4

    # Loss Schedule
    # Switch from BCE+Dice to Lovasz-Hinge at this epoch
    LOVASZ_EPOCH = 15

    # Optimization
    AMP = True  # Automatic Mixed Precision
    GRAD_ACCUMULATION = 1  # Gradient accumulation steps

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-7

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 20  # Must be > LOVASZ_EPOCH to allow fine-tuning

    # =========================================================================
    # Augmentation (Conservative Geometric Regularization)
    # =========================================================================
    AUG_ROTATION = 5  # Degrees
    AUG_SHIFT = 0.05  # Fraction
    AUG_SCALE = 0.05  # Fraction
    AUG_PROB = 0.5  # Probability of applying augmentations

    @classmethod
    def setup(cls, make_dirs=True, set_seed=True):
        """
        Setup the environment: create directories and set random seeds.
        """
        if make_dirs:
            os.makedirs(cls.WORKING_DIR, exist_ok=True)
            os.makedirs(cls.CACHE_DIR, exist_ok=True)
            os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
            os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
            os.makedirs(cls.LOG_DIR, exist_ok=True)

        if set_seed:
            random.seed(cls.SEED)
            np.random.seed(cls.SEED)
            torch.manual_seed(cls.SEED)
            torch.cuda.manual_seed(cls.SEED)
            # Ensure deterministic behavior for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def get_config_dict(cls):
        """Returns a dictionary representation of the configuration."""
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
