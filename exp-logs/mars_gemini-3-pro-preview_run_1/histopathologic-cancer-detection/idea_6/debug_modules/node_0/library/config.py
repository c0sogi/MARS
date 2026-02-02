import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the Cross-Validated DenseNet Family Ensemble strategy.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    WORK_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 48  # Input size for the model
    CENTER_CROP_SIZE = 48  # Size of the center crop to extract (Hard Attention)
    NUM_WORKERS = 12  # Number of CPU workers for data loading
    PIN_MEMORY = True

    # ==========================================
    # Model Configuration
    # ==========================================
    # Ensemble of DenseNet121 and DenseNet169
    MODEL_ARCHS = ["densenet121", "densenet169"]
    PRETRAINED = True
    MODIFY_STEM = True  # Replace standard 7x7 stride-2 stem with 3x3 stride-1
    NUM_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    N_FOLDS = 5  # 5-Fold Cross-Validation
    EPOCHS = 20  # Maximum epochs per fold
    BATCH_SIZE = 128  # Batch size (A100 can handle large batches for 48x48)

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = 20  # Should match EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 6  # Relaxed patience for stability
    MIN_DELTA = 1e-4

    # ==========================================
    # Augmentation (Conservative)
    # ==========================================
    # Mild color adjustments only; no hue/saturation
    AUG_BRIGHTNESS = 0.1
    AUG_CONTRAST = 0.1

    # Geometric
    AUG_H_FLIP_PROB = 0.5
    AUG_V_FLIP_PROB = 0.5

    # ==========================================
    # Inference Configuration
    # ==========================================
    TTA_VIEWS = 4  # Test Time Augmentation views (Original, HFlip, VFlip, Rot90)

    # ==========================================
    # Debugging
    # ==========================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use if DEBUG is True

    @classmethod
    def setup(cls):
        """Creates necessary directories and sets up the environment."""
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        cls.set_seed(cls.SEED)

    @staticmethod
    def set_seed(seed=42):
        """Sets fixed random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Automatically setup directories on import
Config.setup()
