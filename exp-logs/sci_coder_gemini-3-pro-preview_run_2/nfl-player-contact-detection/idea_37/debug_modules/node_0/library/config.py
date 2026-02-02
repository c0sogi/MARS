import os
import torch


class Config:
    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_37"
    SUBMISSION_DIR = "./submission"

    # Create working and submission directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Files
    TRAIN_LABELS_PATH = os.path.join(INPUT_DIR, "train_labels.csv")
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")

    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Video Metadata
    VIDEO_METADATA_TRAIN_PATH = os.path.join(INPUT_DIR, "train_video_metadata.csv")
    VIDEO_METADATA_TEST_PATH = os.path.join(INPUT_DIR, "test_video_metadata.csv")

    # Metadata Files (Pre-generated)
    METADATA_TRAIN = os.path.join(METADATA_DIR, "train.csv")
    METADATA_VAL = os.path.join(METADATA_DIR, "validation.csv")
    METADATA_TEST = os.path.join(METADATA_DIR, "test.csv")

    # Cache & Output Paths
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.parquet")

    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # 12 vCPUs available

    BATCH_SIZE = 8192
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    PATIENCE = 3  # Early stopping patience

    # Focal Loss Configuration
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # =========================================================================
    # Model Architecture: SPIRV-Net
    # =========================================================================
    # Kinematic Stream (Stochastic Pyramidal Backbone)
    # Project(512) -> ResBlock -> Project(256) -> ResBlock -> Project(128)
    KINEMATIC_HIDDEN_DIMS = [512, 256, 128]
    KINEMATIC_DROPOUT = 0.3
    INPUT_NOISE_SIGMA = 0.05  # Stochastic Input Regularization (Training only)

    # Visual Stream (Shallow Correction)
    VISUAL_HIDDEN_DIMS = [64]

    # Residual Fusion
    RESIDUAL_LAMBDA = 1.0  # Weight for additive visual residual

    # =========================================================================
    # Data & Feature Engineering
    # =========================================================================
    WINDOW_SIZE = 5  # Window t-5 to t+5 (Total 11 timesteps)

    # Raw Tracking Columns
    TRACKING_COLS = [
        "game_play",
        "game_key",
        "play_id",
        "nfl_player_id",
        "step",
        "datetime",
        "x_position",
        "y_position",
        "speed",
        "distance",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    # Raw Helmet Columns
    HELMET_COLS = [
        "game_play",
        "play_id",
        "nfl_player_id",
        "frame",
        "left",
        "width",
        "top",
        "height",
        "view",
    ]
