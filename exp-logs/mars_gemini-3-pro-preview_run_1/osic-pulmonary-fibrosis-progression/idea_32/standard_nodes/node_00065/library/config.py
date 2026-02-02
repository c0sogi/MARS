import os
import torch


class Config:
    """
    Global configuration for the Max-Pooled Visual-Exclusive Residual Network (MP-VER-Net).
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use when DEBUG is True
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this idea implementation
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_32")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # File paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================================
    # Data / Image Processing
    # ==========================================
    IMG_SIZE = 224
    SLAB_COUNT = 3  # Tri-slab configuration
    USE_AUGMENTATION = True

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "tf_efficientnet_b0_ns"
    FEATURE_DIM = 1280  # Native dimensionality of EfficientNet-B0

    # Progressive Expansion MLP for Tabular Data
    # Input dim depends on encoding (e.g., 9 for OHE), output matches visual dim
    TABULAR_HIDDEN_DIMS = [64, 256, 1280]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 50
    BATCH_SIZE = 32
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 8  # Strict patience for Early Stopping

    # Scheduler settings (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # ==========================================
    # Metric / Loss Constants
    # ==========================================
    MAX_ERROR = 1000
    MIN_CONFIDENCE = 70

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories to ensure file system safety.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds for reproducibility
        import random
        import numpy as np

        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
