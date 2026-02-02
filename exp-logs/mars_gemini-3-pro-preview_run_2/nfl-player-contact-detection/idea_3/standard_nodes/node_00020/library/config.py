import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for idea_3 (TR-GCN)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_3")
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Tracking Data Files
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    # Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Processing & Feature Engineering
    # =========================================================================
    SEED = 42

    # Temporal Window: 11 frames total (Target t, plus 5 frames before and 5 after)
    WINDOW_SIZE = 11

    # Raw columns to extract from tracking CSVs
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

    # Numerical features to be normalized and used in the network
    # These are per-player features. The model input will contain these for
    # Player 1 and Player 2, plus derived relative features.
    PLAYER_FEATURES = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
    ]

    # Derived features calculated during processing
    # distance, log_distance, closing_speed, etc.
    # This constant is used to determine input dimension size
    # Structure per timestep: [P1_feats, P2_feats, dist, log_dist, closing_speed, is_ground]
    # len(PLAYER_FEATURES)*2 + 4
    NUM_FEATURES_PER_TIMESTEP = len(PLAYER_FEATURES) * 2 + 4

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Training
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    EPOCHS = 10

    # Class Imbalance Handling
    # Based on data analysis: Ratio of No-Contact to Contact is approx 72.5:1
    POS_WEIGHT = 72.5

    # Architecture (TR-GCN)
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 3
    HIDDEN_DIM = 128
    DROPOUT = 0.3

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # =========================================================================
    # Debugging
    # =========================================================================
    # Set to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 10000
