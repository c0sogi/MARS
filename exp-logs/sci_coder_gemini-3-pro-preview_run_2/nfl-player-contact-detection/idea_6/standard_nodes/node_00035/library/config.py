import os


class Config:
    """
    Global configuration for the NFL Contact Detection task.
    Implements settings for the Conditioned Kinematic Residual Network (CK-ResNet).
    """

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Input Data Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (Parquet/NPY/Joblib)
    TRAIN_FEATURES_CACHE = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_CACHE = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_CACHE = os.path.join(WORKING_DIR, "test_features.parquet")

    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4  # Number of workers for data loading

    # ==========================================
    # Feature Engineering Constants
    # ==========================================
    # Temporal window size: +/- 5 frames.
    # Total window length = 5 (past) + 1 (curr) + 5 (future) = 11 frames.
    WINDOW_SIZE = 5

    # Raw tracking columns to extract
    TRACKING_COLS = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]

    # Conditioning feature for FiLM
    CONDITION_COL = "is_ground"

    # ==========================================
    # Model Hyperparameters (CK-ResNet)
    # ==========================================
    # Input dimension will be calculated dynamically based on feature count * window size
    HIDDEN_DIM = 512
    NUM_RES_BLOCKS = 4
    DROPOUT_RATE = 0.1
    FILM_DIM = 1  # Dimension of the conditioning vector (is_ground)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 4096  # Large batch size for tabular data
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3

    # Focal Loss Parameters
    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 2.0

    # ==========================================
    # Inference & Evaluation
    # ==========================================
    THRESHOLD_STEPS = 200  # Number of steps for grid search on validation threshold

    @staticmethod
    def setup_directories():
        """Creates necessary working and submission directories if they don't exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
