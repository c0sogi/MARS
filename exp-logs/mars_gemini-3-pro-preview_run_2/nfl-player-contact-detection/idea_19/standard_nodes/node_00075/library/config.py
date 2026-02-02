import os


class Config:
    # =========================================================================
    # Paths and Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_20"
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Global Constants
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4

    # Data Processing
    # Window size for temporal context: t-WINDOW_SIZE to t+WINDOW_SIZE
    WINDOW_SIZE = 5
    TOTAL_STEPS = 2 * WINDOW_SIZE + 1

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Training
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3

    # Architecture
    # Kinematic Stream (Deep Residual MLP)
    KINEMATIC_HIDDEN_DIMS = [512, 256, 128]

    # Stereoscopic Visual Stream (Comparator MLP)
    VISUAL_HIDDEN_DIMS = [256, 128, 64]

    # General
    DROPOUT_RATE = 0.1
    FOCAL_LOSS_GAMMA = 2.0
    LAMBDA_VISUAL = 1.0  # Weight for the visual residual branch

    # =========================================================================
    # Feature Definitions
    # =========================================================================

    # -------------------------------------------------------------------------
    # Kinematic Features (Tracking Data)
    # -------------------------------------------------------------------------
    # These features are generated for every step in the window.
    # Base features for Player 1
    KIN_FEATS_P1 = [
        "x_position_1",
        "y_position_1",
        "speed_1",
        "acceleration_1",
        "orientation_1",
        "direction_1",
        "sa_1",
    ]

    # Base features for Player 2 (or Ground proxy)
    KIN_FEATS_P2 = [
        "x_position_2",
        "y_position_2",
        "speed_2",
        "acceleration_2",
        "orientation_2",
        "direction_2",
        "sa_2",
    ]

    # Interaction/Relative Features
    KIN_FEATS_REL = [
        "distance",
        "log_distance",
        "relative_speed",
        "closing_speed",
        "is_ground",
    ]

    # Combined list for a single timestep
    KINEMATIC_FEATURES_SINGLE_STEP = KIN_FEATS_P1 + KIN_FEATS_P2 + KIN_FEATS_REL

    # -------------------------------------------------------------------------
    # Stereoscopic Visual Features (Helmet/Camera Data)
    # -------------------------------------------------------------------------
    # Unified View Features (Max-Pooled from Sideline/Endzone)
    # We select the "best" view (highest visibility) per timestep.
    VIS_FEATS_UNIFIED = [
        "vis_iou",
        "vis_dist",
        "vis_p1_top",
        "vis_p2_top",
        "vis_p1_area",
        "vis_p2_area",
        "vis_avail",
    ]

    # Combined list for a single timestep
    VISUAL_FEATURES_SINGLE_STEP = VIS_FEATS_UNIFIED

    # =========================================================================
    # Derived Dimensions
    # =========================================================================
    # Input dimension for the Kinematic Stream
    # (Features per step) * (Number of steps in window)
    INPUT_DIM_KINEMATIC = len(KINEMATIC_FEATURES_SINGLE_STEP) * TOTAL_STEPS

    # Input dimension for the Visual Stream
    INPUT_DIM_VISUAL = len(VISUAL_FEATURES_SINGLE_STEP) * TOTAL_STEPS
