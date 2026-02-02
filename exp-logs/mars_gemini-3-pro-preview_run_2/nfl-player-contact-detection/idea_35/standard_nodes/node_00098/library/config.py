import os
import torch


class Config:
    """
    Configuration for the Adaptive Pyramidal Invariant Residual-Visual Network (APIRV-Net).
    Centralizes all paths, hyperparameters, and constants.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_35"
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Files
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "apirv_net_best.pth")
    SCALER_SAVE_PATH = os.path.join(WORKING_DIR, "scaler.joblib")

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data Processing & Feature Engineering
    # =========================================================================
    # Temporal Window: t-5 to t+5 (11 frames total)
    WINDOW_LAG = 5
    WINDOW_FUTURE = 5

    # Physical Constraints (Clamping)
    # Used to prevent outliers in derivative features from destabilizing the network
    CLAMP_MIN = -50.0
    CLAMP_MAX = 50.0

    # Feature Caching
    # Files to store processed features in the working directory
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.parquet")

    # =========================================================================
    # Model Architecture (APIRV-Net)
    # =========================================================================
    # Kinematic Stream: Pyramidal Backbone
    # Structure: Input -> 512 -> 256 -> 128 -> Logit
    KINEMATIC_HIDDEN_DIMS = [512, 256, 128]

    # Visual Stream: Shallow MLP for geometric rules
    VISUAL_HIDDEN_DIMS = [64, 32]

    # Regularization
    DROPOUT_RATE = 0.1

    # Fusion
    # Initial bias or scaling for fusion can be defined here if needed,
    # though usually handled by the network weights.

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Batch Size: Large batch size for stable BatchNorm stats
    BATCH_SIZE = 8192

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 3

    # Focal Loss Parameters
    # Alpha=0.25 balances positive/negative classes
    # Gamma=2.0 focuses on hard examples
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # =========================================================================
    # Compute Resources
    # =========================================================================
    # Use all available vCPUs for data loading
    NUM_WORKERS = 12

    # Device selection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
