import os
import random
import numpy as np
import torch


class Config:
    """
    Centralized configuration for the Lung Function Decline prediction task.
    Implements the settings for the GeM-Pooled Dual-Axis Transformer strategy.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for Idea 9 (Deterministic Data Processing)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_9")

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Preprocessing
    # ==========================================
    IMG_SIZE = 224  # Native resolution to avoid artifacts
    NUM_SLABS = 3  # Tri-slab approach
    SLAB_OVERLAP = 0.15  # 15% overlap

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True
    GEM_P_INIT = 3.0  # Initial p value for GeM pooling
    TOKEN_DIM = 512  # Dimension for visual and tabular tokens
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 1
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 16  # Adjusted for dual-backbone memory usage
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 8  # Strict patience to prevent overfitting on small data

    # Metric Constraints
    MAX_ERROR = 1000  # Error clipping threshold
    MIN_CONFIDENCE = 70  # Confidence clipping threshold

    # ==========================================
    # Compute & Environment
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use when DEBUG is True

    @staticmethod
    def seed_everything(seed=42):
        """
        Sets the random seed for all relevant libraries to ensure reproducibility.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def setup(cls):
        """
        Prepares the environment by creating necessary directories and setting seeds.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        cls.seed_everything(cls.SEED)
