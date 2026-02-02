import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"

    # Raw Data Files
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    CACHE_DIR = WORKING_DIR  # For caching processed datasets
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # ==========================================
    # 3. Data Engineering Parameters
    # ==========================================
    # Physics constants/limits if needed for normalization
    PRESSURE_MIN = -2.0
    PRESSURE_MAX = 65.0

    # Feature Engineering Flags
    USE_TIME_WEIGHTED_INTEGRATION = True
    USE_PHYSICS_INTERACTIONS = True
    USE_LAG_FEATURES = True
    USE_DERIVATIVES = True

    # ==========================================
    # 4. Model Architecture (DP-GI-BiLSTM)
    # ==========================================
    # Backbone
    LSTM_LAYERS = 4
    LSTM_HIDDEN = 512
    BIDIRECTIONAL = True

    # Injection & Projection
    INPUT_DIM = 0  # To be calculated dynamically based on feature count
    USE_DUAL_PATH_INJECTION = True
    USE_DEEP_INJECTION = True

    # Regularization
    DROPOUT = 0.1
    USE_LAYER_NORM = True

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    # Long-Tail Convergence Protocol
    EPOCHS = 150
    BATCH_SIZE = 512  # Optimized for A100 40GB

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000.0

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-5

    # Loss Function Weights
    LOSS_WEIGHT_INSPIRATORY = 1.0
    LOSS_WEIGHT_EXPIRATORY = 0.1

    @staticmethod
    def setup():
        """Creates necessary directories and sets reproducibility seeds."""
        # Create directories
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            # Deterministic algorithms can be slower, but ensure reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def print_config():
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for attr in dir(Config):
            if not attr.startswith("__") and not callable(getattr(Config, attr)):
                print(f"{attr}: {getattr(Config, attr)}")
        print("=" * 30)
