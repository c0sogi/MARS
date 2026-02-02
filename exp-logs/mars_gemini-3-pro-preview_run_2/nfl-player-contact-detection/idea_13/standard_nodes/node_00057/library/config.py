import os
import torch


class Config:
    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for this solution idea
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File Paths for Caching and Outputs
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Temporal Window: Current step +/- WINDOW_SIZE
    # Total steps = 2 * WINDOW_SIZE + 1
    WINDOW_SIZE = 5

    # Columns from player tracking to use for feature engineering
    # We use these to generate lags and relative physics
    TRACKING_COLS = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
        "orientation",
        "sa",  # Signed acceleration
    ]

    # Flag to control caching behavior
    CACHE_DATA = True

    # ==========================================
    # Model Architecture (Entity-Centric Wide-Residual-MLP)
    # ==========================================
    HIDDEN_DIM = 512
    NUM_RES_BLOCKS = 3
    DROPOUT_RATE = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42

    # Large batch size for efficient MLP training on wide data
    BATCH_SIZE = 4096

    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # For AdamW

    # Focal Loss Settings
    FOCAL_LOSS_GAMMA = 2.0

    # Weighted BCE Settings (Cite solution_lesson_node_00008)
    # Imbalance is approx 1:72.5
    POS_WEIGHT = 72.5

    # Early Stopping
    PATIENCE = 5

    # ==========================================
    # Compute Settings
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
