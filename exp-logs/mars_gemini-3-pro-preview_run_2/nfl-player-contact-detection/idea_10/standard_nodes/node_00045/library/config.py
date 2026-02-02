import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for the Explicitly-Featurized Wide Residual Network (EF-WideResNet) pipeline.
    """

    # =========================================================================
    # File Paths and Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")
    MODEL_PATH = os.path.join(WORKING_DIR, "ef_wideresnet_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Global System Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to subsample data for rapid debugging
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Data Processing & Windowing
    # =========================================================================
    # Temporal Window: t-5 to t+5 (11 frames total)
    WINDOW_HALF = 5
    WINDOW_SIZE = 2 * WINDOW_HALF + 1

    # Ground Imputation Logic
    # 1. Position: Imputed as Player's position (Distance -> 0)
    # 2. Kinematics: Imputed as 0 (Preserves Relative Motion/Closing Speed)
    IMPUTE_GROUND_KINEMATICS_ZERO = True

    # =========================================================================
    # Feature Engineering
    # =========================================================================
    # Raw tracking columns to extract
    TRACKING_COLS = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
        "orientation",
        "sa",
    ]

    # Engineered Feature Names
    FEAT_DISTANCE = "distance"
    FEAT_LOG_DISTANCE = "log_distance"
    FEAT_REL_SPEED = "relative_speed"
    FEAT_CLOSING_SPEED = "closing_speed"
    FEAT_REL_ACCEL = "relative_acceleration"
    FEAT_IS_GROUND = "is_ground"

    # Categorical Features for Embedding
    CAT_POSITION = "position"

    # =========================================================================
    # Model Architecture (EF-WideResNet)
    # =========================================================================
    # The model is a Deep Residual MLP.
    # Input dimension is dynamic based on (Features * Window) + Embeddings.

    HIDDEN_DIM = 512  # Width of the residual blocks
    NUM_RES_BLOCKS = 3  # Depth of the network
    DROPOUT_RATE = 0.2  # Regularization

    # Embedding Configuration
    # 'position' has approx 28 unique values + unknowns
    EMBEDDING_CONFIG = {"position": {"num_embeddings": 32, "embedding_dim": 8}}

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 4096
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5

    # Focal Loss Parameters
    # Designed to handle extreme class imbalance (approx 1:72)
    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 2.0

    @staticmethod
    def set_seed(seed=42):
        """Sets fixed seeds for reproducibility across all libraries."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
