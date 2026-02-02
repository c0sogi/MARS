import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for the Balanced-Skip Holistic Dual-Axis Network (BS-HDAN).
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==========================================
    # 1. General Settings & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a subset of data
    DEBUG_SIZE = 50  # Number of samples if DEBUG is True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # 2. File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea
    WORKING_DIR = "./working/idea_43"

    # Sub-directories for caching and artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist immediately upon config load
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 3. Data Processing (Tri-Slab)
    # ==========================================
    # Image Resolution: Matches EfficientNet-B0 native resolution
    IMG_SIZE = 224

    # Tri-Slab Generation
    SLAB_COUNT = 3
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Normalization Stats (Derived from EDA)
    # Used for standard scaling of numerical inputs
    STATS = {
        "Weeks": {"mean": 31.3751, "std": 23.4602},
        "Percent": {"mean": 76.9105, "std": 19.1970},
        "Age": {"mean": 67.5825, "std": 6.6259},
    }

    # ==========================================
    # 4. Model Architecture (BS-HDAN)
    # ==========================================
    # Backbone
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_PRETRAINED = True
    VISUAL_DIM = 1280  # Native output dim of EfficientNet-B0 (No projection)

    # Dual-Path Tabular Encoding
    # Path A: High-dim alignment for fusion with visual features
    TABULAR_FUSION_DIM = 1280
    # Path B: Low-dim balanced prior for skip connection
    TABULAR_PRIOR_DIM = 128

    # Attention Mechanism
    ATTN_HEADS = 4
    ATTN_LAYERS = 1
    DROPOUT = 0.1

    # Feature Definitions
    TABULAR_COLS = ["Weeks", "Percent", "Age", "Sex", "SmokingStatus"]
    NUM_COLS = ["Weeks", "Percent", "Age"]
    CAT_COLS = ["Sex", "SmokingStatus"]

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-2

    # Optimization Strategy
    PATIENCE = 8  # Strict early stopping
    SCHEDULER_T_MAX = EPOCHS  # Cosine Annealing cycle

    # Metric / Loss Constants (Modified Laplace Log Likelihood)
    MAX_ERROR = 1000.0
    MIN_SIGMA = 70.0

    @staticmethod
    def set_seed(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Deterministic operations usually slow down training,
            # but are good for strict reproducibility.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Initialize seed on import
Config.set_seed(Config.SEED)
