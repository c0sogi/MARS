import os


class Config:
    """
    Configuration for WIRK-Net (Wide-Input Residual Kinematic Network).
    Central source of truth for file paths, hyperparameters, and feature definitions.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # File paths for metadata
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # File paths for tracking data
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    # Cache file paths (Parquet/NPY/Joblib)
    # Using Parquet for large tabular feature sets
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Scaler and Model artifacts
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "wirk_net_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data Preprocessing & Feature Engineering
    # =========================================================================
    # Window size for the Wide Format (t - WINDOW_SIZE to t + WINDOW_SIZE)
    # Total steps per sample = 2 * WINDOW_SIZE + 1
    WINDOW_SIZE = 5

    # Debugging / Development
    # If DEBUG is True, the pipeline should use a subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50000

    # Raw kinematic features to extract from tracking data
    # These will be retrieved for both Player 1 and Player 2 (or Ground)
    RAW_KINEMATIC_FEATURES = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",  # Signed acceleration
    ]

    # Categorical features for Entity Embeddings
    CATEGORICAL_FEATURES = ["position"]

    # Derived features calculated during preprocessing
    # These capture the interaction between P1 and P2
    DERIVED_FEATURES = [
        "distance",  # Euclidean distance
        "log_distance",  # log(1 + distance) for resolution near 0
        "relative_speed",  # Magnitude of velocity difference vector
        "relative_acceleration",  # Magnitude of acceleration difference vector
        "closing_speed",  # Velocity projection onto distance vector
    ]

    # Ground Imputation Logic
    # When P2 is Ground ('G'), we impute:
    # Position -> P1 Position (Distance = 0)
    # Velocity/Accel -> 0 (Preserve relative motion)

    # =========================================================================
    # Model Architecture (WIRK-Net)
    # =========================================================================
    # Deep Residual MLP configuration
    HIDDEN_DIM = 512
    NUM_RESIDUAL_BLOCKS = 4  # Depth allows learning complex interactions
    DROPOUT_RATE = 0.2

    # Embedding dimensions
    # Map categorical cardinality to embedding size
    # 'position' has ~28 unique values
    POS_EMBEDDING_DIM = 8

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 4096  # Large batch size for efficient tabular training
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 20
    PATIENCE = 3  # Early stopping patience

    # Focal Loss Configuration
    # Designed to handle the 1:72 class imbalance
    # Alpha: Weighting factor for the rare class (Contact=1)
    # Gamma: Focusing parameter for hard examples
    FOCAL_LOSS_ALPHA = 0.75
    FOCAL_LOSS_GAMMA = 2.0

    # =========================================================================
    # Inference & Evaluation
    # =========================================================================
    # Grid search range for threshold optimization on Validation set
    THRESHOLD_SEARCH_START = 0.1
    THRESHOLD_SEARCH_END = 0.6
    THRESHOLD_SEARCH_STEP = 0.01
