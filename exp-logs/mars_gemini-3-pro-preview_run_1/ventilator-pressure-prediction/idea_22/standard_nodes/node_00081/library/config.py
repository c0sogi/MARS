import os
import torch


class Config:
    """
    Configuration for the Wide-Projected Physics-Composite Network strategy.
    Centralizes all hyperparameters, file paths, and feature engineering flags.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Input Metadata (Pre-split)
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"

    # Working Directory for Caching and Models
    # Using specific idea folder to prevent cache collisions
    WORKING_DIR = "./working/idea_23/"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data & Feature Engineering
    # ==========================================
    # Physics-Fidelity Features
    USE_PHYSICS_INTERACTIONS = True  # R*u_in, Vol/C
    USE_LAGS = True
    LAG_STEPS = 4
    USE_DIFFS = True  # 1st and 2nd order diffs of u_in

    # Control Segregation
    # u_out is binary and must not be scaled.
    # Continuous cols will be RobustScaled.
    CONTINUOUS_COLS = ["time_step", "u_in", "R", "C"]
    # Note: R and C are technically discrete settings but treated as continuous physics properties

    # ==========================================
    # Model Architecture (Wide-Projected)
    # ==========================================
    # Bottleneck-to-Wide Topology
    STEM_DIM = 512  # Compressed initialization to filter noise
    MODEL_DIM = 1024  # High-capacity latent space

    # Backbone: Wide-State Identity Blocks
    NUM_LAYERS = 4
    LSTM_HIDDEN = 512  # Per direction (Bi-LSTM -> 1024 output matches MODEL_DIM)
    FFN_EXPANSION = 2  # 2x expansion (2048 units) for stability
    DROPOUT = 0.1

    # Deep Supervision
    AUX_LAYER_INDEX = 1  # 0-indexed: Output of Block 2 (Layer index 1)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 35  # Extended horizon for OneCycleLR annealing
    BATCH_SIZE = 512  # Sufficient updates

    # Optimization
    LR_MAX = 1e-3
    WEIGHT_DECAY = 1e-2
    GRAD_CLIP = 1.0  # Strict clipping for LSTM stability

    # Loss Weights
    LOSS_AUX_WEIGHT = 0.3  # Weight for auxiliary head

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def to_dict(cls):
        """
        Returns configuration as a dictionary for logging.
        """
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }


# Ensure directories are created upon import or usage
Config.setup()
