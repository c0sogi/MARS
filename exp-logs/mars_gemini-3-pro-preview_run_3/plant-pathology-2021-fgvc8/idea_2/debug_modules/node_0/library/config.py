import os
import torch
import random
import numpy as np


class Config:
    """
    Centralized configuration for the Apple Disease Detection task.
    Handles hyperparameters, file paths, and system settings.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_ROOT = "./input"

    # Metadata paths (pre-generated)
    TRAIN_METADATA = "./metadata/train.csv"
    VAL_METADATA = "./metadata/val.csv"
    TEST_METADATA = "./metadata/test.csv"

    # Output paths
    WORKING_DIR = "./working/idea_2"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 224
    NUM_CLASSES = 6
    # Classes sorted alphabetically to ensure consistent mapping
    CLASSES = [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ]

    # DataLoader settings
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Safe default for 12 vCPUs
    PIN_MEMORY = True

    # Debugging
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    DROPOUT_RATE = 0.2

    # ==========================================
    # Training Configuration
    # ==========================================
    SEED = 42
    EPOCHS = 15

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler (CosineAnnealingLR)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # ==========================================
    # Advanced Strategies (Augmentation & Loss)
    # ==========================================
    # MixUp and CutMix
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    MIX_PROB = 0.5  # Probability of applying mixup/cutmix

    # Loss Function (Asymmetric Loss)
    USE_ASL = True
    ASL_GAMMA_NEG = 4.0
    ASL_GAMMA_POS = 0.0
    ASL_CLIP = 0.05

    # Inference
    USE_TTA = True  # Test Time Augmentation (Horizontal Flip)

    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        """
        Initialize the configuration:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        self._create_directories()
        self._set_seed()

    def _create_directories(self):
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

    def _set_seed(self):
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.SEED)
            torch.cuda.manual_seed_all(self.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
