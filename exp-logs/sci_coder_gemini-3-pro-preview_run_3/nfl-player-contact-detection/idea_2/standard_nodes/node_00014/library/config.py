import os


class Config:
    """
    Configuration for the NFL Contact Detection Task.
    Defines paths, hyperparameters, and feature specifications.
    """

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Input File Paths
    # =========================================================================
    # Original Data
    TRAIN_LABELS_PATH = os.path.join(INPUT_DIR, "train_labels.csv")
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")

    # Metadata (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Output / Cache Paths
    # =========================================================================
    # Caching processed features (Parquet for dataframes, NPY for arrays)
    # Train
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "train_X.parquet")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y.npy")
    CACHE_TRAIN_IDS = os.path.join(WORKING_DIR, "train_ids.npy")

    # Validation
    CACHE_VAL_X = os.path.join(WORKING_DIR, "val_X.parquet")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y.npy")
    CACHE_VAL_IDS = os.path.join(WORKING_DIR, "val_ids.npy")

    # Test
    CACHE_TEST_X = os.path.join(WORKING_DIR, "test_X.parquet")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # Model and Scaler Artifacts
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Hyperparameters
    # =========================================================================
    SEED = 42

    # Data Processing
    WINDOW_SIZE = 9  # Temporal window: +/- 4 frames (Current frame + 4 past + 4 future)
    NEGATIVE_RATIO = 10.0  # Undersampling ratio (Negatives : Positives)

    # Training
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    EPOCHS = 10

    # Model Architecture
    HIDDEN_CHANNELS = 64
    KERNEL_SIZE = 3
    DROPOUT = 0.1

    # =========================================================================
    # Features
    # =========================================================================
    # Hardcoded list of features to ensure deterministic channel ordering.
    # Total Channels: 22
    FEATURES = [
        # --- Player 1 (Subject) ---
        "x_position_p1",
        "y_position_p1",
        "speed_p1",
        "acceleration_p1",
        "sa_p1",
        "sin_direction_p1",
        "cos_direction_p1",
        "sin_orientation_p1",
        "cos_orientation_p1",
        # --- Player 2 (Object / Ground) ---
        # Note: If contact is with Ground, these will be 0-padded
        "x_position_p2",
        "y_position_p2",
        "speed_p2",
        "acceleration_p2",
        "sa_p2",
        "sin_direction_p2",
        "cos_direction_p2",
        "sin_orientation_p2",
        "cos_orientation_p2",
        # --- Interaction Dynamics ---
        "distance",
        "rel_speed",
        "rel_acceleration",
        # --- Context ---
        "is_ground",
    ]

    NUM_FEATURES = len(FEATURES)
