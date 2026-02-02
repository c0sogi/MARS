import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # The working directory for this specific idea (Idea 23)
    WORKING_DIR = "./working/idea_23"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # File paths for specific datasets
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

    SUBMISSION_PATH = os.path.join(
        os.path.dirname(WORKING_DIR), "submission.csv"
    )  # Save to parent or specific loc

    # =========================================================================
    # Data Processing & Feature Engineering
    # =========================================================================
    # Random Seed for reproducibility
    SEED = 42

    # Window Size: t-5 to t+5 (Total 11 frames)
    WINDOW_SIZE = 5

    # Raw columns to load from tracking data
    TRACKING_COLS = [
        "game_play",
        "step",
        "nfl_player_id",
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    # Raw columns to load from helmet data
    HELMET_COLS = [
        "game_play",
        "step",
        "nfl_player_id",
        "left",
        "width",
        "top",
        "height",
        "view",
    ]

    # Derived Kinematic Features (per timestep)
    # These will be flattened: [feat_t-5, ..., feat_t, ..., feat_t+5]
    KINEMATIC_FEATURES = [
        "x_position_1",
        "y_position_1",
        "speed_1",
        "direction_1",
        "orientation_1",
        "acceleration_1",
        "sa_1",
        "x_position_2",
        "y_position_2",
        "speed_2",
        "direction_2",
        "orientation_2",
        "acceleration_2",
        "sa_2",
        "distance",
        "relative_speed",
        "relative_angle",
        "closing_speed",
    ]

    # Derived Visual Features (per timestep)
    VISUAL_FEATURES = [
        "left_1",
        "width_1",
        "top_1",
        "height_1",
        "area_1",
        "left_2",
        "width_2",
        "top_2",
        "height_2",
        "area_2",
        "visual_iou",
    ]

    # Numerical Stability Parameters
    CLAMP_MIN = -50.0
    CLAMP_MAX = 50.0

    # =========================================================================
    # Model Architecture (KCVR-Net)
    # =========================================================================
    # Kinematic Stream: Deep Cross Network (DCNv2)
    DCN_NUM_CROSS_LAYERS = 3
    DCN_DEEP_LAYERS = [512, 256, 128]
    DCN_DROPOUT = 0.1

    # Visual Stream: Shallow MLP
    VISUAL_HIDDEN_LAYERS = [64, 32]
    VISUAL_DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Optimization
    BATCH_SIZE = 4096  # Large batch size for stability
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 3

    # Loss Function: Focal Loss
    # Specifically tuned to alpha=0.25, gamma=2.0 as per requirements
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # =========================================================================
    # Evaluation
    # =========================================================================
    # Threshold optimization range
    THRESHOLD_RANGE = [0.1, 0.9]
    THRESHOLD_STEPS = 100
