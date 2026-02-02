import os

# Create necessary directories for caching and submission
WORKING_DIR = "./working/idea_13"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


class Config:
    """
    Global configuration for the Dual-Stream GBDT Contact Detection Pipeline.
    Implements the settings for Multi-Resolution Visual Geometry and Robust Imputation.
    """

    # =========================================================================
    # Directories and File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = WORKING_DIR

    # Output Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Files
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
    TRAIN_VIDEO_META_PATH = os.path.join(INPUT_DIR, "train_video_metadata.csv")
    TEST_VIDEO_META_PATH = os.path.join(INPUT_DIR, "test_video_metadata.csv")

    # =========================================================================
    # Global Constants
    # =========================================================================
    SEED = 42
    N_JOBS = 12  # Matches available vCPUs

    # =========================================================================
    # Feature Engineering Configuration
    # =========================================================================
    # Multi-Resolution Windows (in 0.1s steps)
    # Micro: Captures instantaneous mechanics and impact forces
    MICRO_WINDOW = 4  # +/- 0.4 seconds
    # Macro: Captures trajectory context and approach vectors
    MACRO_WINDOW = 15  # +/- 1.5 seconds

    # Explicit Lag Definitions for Feature Engineering
    TRACKING_LAGS = list(range(1, MACRO_WINDOW + 1))
    VISUAL_LAGS = list(range(1, MICRO_WINDOW + 1))

    # Base Tracking Features to be flattened/aggregated
    TRACKING_COLS = [
        "x_position",
        "y_position",
        "speed",
        "distance",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    # Visual Features to be computed and aggregated
    VISUAL_COLS = [
        "helmet_iou",  # Intersection over Union
        "helmet_dist",  # Centroid distance
    ]

    # =========================================================================
    # Stream-Specific Configurations
    # =========================================================================

    # Stream A: Interaction Model (Player vs Player)
    # Logic: Uses Tracking (P1 & P2) + Visual Geometry + Interaction Features
    STREAM_A = {
        "name": "stream_a_interaction",
        "target_type": "player",  # Train only on rows where player2 != 'G'
        "use_visuals": True,  # Enable visual features (Multi-Resolution)
        "use_interaction": True,  # Enable P1-P2 relative features (speed diff, closure)
        "neg_pos_ratio": 10.0,  # Random undersampling ratio for majority class
        "impute_visuals": -999.0,  # Sentinel value for missing visual data (Robust Imputation)
    }

    # Stream B: Impact Model (Player vs Ground)
    # Logic: Uses Tracking (P1 only) + Kinematics + Physics Derivatives
    # Constraints: Explicitly blocks Visuals and P2 features to reduce noise
    STREAM_B = {
        "name": "stream_b_impact",
        "target_type": "ground",  # Train only on rows where player2 == 'G'
        "use_visuals": False,  # Disable visual features
        "use_interaction": False,  # Disable P2 relative features
        "neg_pos_ratio": 10.0,  # Random undersampling ratio
        "compute_jerk": True,  # Enable Jerk (derivative of acceleration)
        "compute_alignment": True,  # Enable Pose-Motion alignment (tumbling detection)
    }

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # XGBoost Configuration optimized for NVIDIA A100
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",  # Optimized for GPU
        "device": "cuda",  # Use GPU
        "learning_rate": 0.02,  # Lower LR for better generalization
        "n_estimators": 3000,  # High tree count with early stopping
        "max_depth": 8,  # Deeper trees to capture complex interactions
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "n_jobs": N_JOBS,
        "random_state": SEED,
    }

    # Training Control
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 100

    # Inference
    # Default threshold, to be optimized per stream on validation set
    DEFAULT_THRESHOLD = 0.5
