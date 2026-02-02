import os


class Config:
    """
    Configuration for the Center-Augmented Wide-Residual Network (CA-WRN) pipeline.
    Defines paths, hyperparameters, and feature specifications for the NFL Contact Detection task.
    """

    # ==========================================
    # File System Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    NUM_WORKERS = 12  # Utilize available vCPUs for data loading

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Window size: +/- steps around t=0.
    # Total sequence length = 2 * WINDOW_SIZE + 1 (e.g., 11 steps for size 5)
    WINDOW_SIZE = 5

    # Columns to load from the raw tracking CSVs
    TRACKING_COLS = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]

    # Features to generate temporal lags for (per player entity)
    # These will be flattened: e.g., speed_1_lag-5 ... speed_1_lag+5
    PLAYER_LAG_FEATURES = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]

    # Interaction features computed between the pair (Player1, Player2/Ground)
    INTERACTION_FEATURES = [
        "distance",
        "log_distance",
        "relative_speed",
        "relative_acceleration",
        "closing_speed",
    ]

    # ==========================================
    # Model Hyperparameters (CA-WRN)
    # ==========================================
    # Architecture Dimensions
    HIDDEN_SIZE = 512
    NUM_LAYERS = 4  # Number of Dense Residual Blocks
    DROPOUT = 0.2

    # Center-Feature Skip Connection
    # List of feature names (at t=0) to be concatenated directly to the final residual output.
    # This enforces the "Center Focus" required for contact detection by bypassing the deep stack.
    CENTER_FEATURE_NAMES = [
        "distance",
        "log_distance",
        "closing_speed",
        "relative_speed",
        "relative_acceleration",
        "speed_1",
        "speed_2",
        "acceleration_1",
        "acceleration_2",
        "is_ground",
    ]

    # ==========================================
    # Training Configuration
    # ==========================================
    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50000  # Subset size when DEBUG is True to speed up iteration

    # Optimization
    BATCH_SIZE = 4096
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 15
    PATIENCE = 3  # For Early Stopping based on Validation MCC

    # Loss Function (Focal Loss)
    # Alpha handles the ~1:72 class imbalance
    FOCAL_ALPHA = 0.75
    FOCAL_GAMMA = 2.0

    # Validation & Inference
    THRESHOLD_SEARCH_STEPS = 100  # Granularity for MCC threshold optimization
