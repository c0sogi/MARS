import os


class Config:
    """
    Global configuration for the Pyramidal Invariant Residual-Visual Network (PIRV-Net).
    """

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for Idea 34 (PIRV-Net) caching and artifacts
    WORKING_DIR = "./working/idea_34"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Compute & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # -------------------------------------------------------------------------
    # Data Processing Configuration
    # -------------------------------------------------------------------------
    # Temporal Context: Window of t-5 to t+5 (11 frames total at 10Hz)
    WINDOW_SIZE = 11
    HALF_WINDOW = 5

    # Physical Constraints for Numerical Stability
    # Used to clamp derived features like closing speed
    CLAMP_MIN = -50.0
    CLAMP_MAX = 50.0

    # -------------------------------------------------------------------------
    # Feature Selection
    # -------------------------------------------------------------------------
    # Raw tracking columns to load from source CSVs
    TRACKING_RAW_COLS = [
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

    # Raw helmet columns to load from source CSVs
    HELMET_RAW_COLS = [
        "game_play",
        "view",
        "frame",
        "nfl_player_id",
        "left",
        "width",
        "top",
        "height",
    ]

    # Kinematic Features (Strictly Numerical & Invariant)
    # These base features are generated for each timestep in the window.
    # Categorical embeddings (team, position) are EXCLUDED to force physical generalization.
    KINEMATIC_FEATURES = [
        # Player 1 State
        "x_position_1",
        "y_position_1",
        "speed_1",
        "acceleration_1",
        "direction_1",
        "orientation_1",
        "sa_1",
        # Player 2 State (or Ground)
        "x_position_2",
        "y_position_2",
        "speed_2",
        "acceleration_2",
        "direction_2",
        "orientation_2",
        "sa_2",
        # Relative / Interaction Features
        "distance",
        "x_rel",
        "y_rel",
    ]

    # Visual Features (Shallow Correction Stream)
    # These features are derived from the Max-Pooling Selection Strategy
    VISUAL_FEATURES = [
        "left_1",
        "width_1",
        "top_1",
        "height_1",
        "left_2",
        "width_2",
        "top_2",
        "height_2",
        "view_area_1",
        "view_area_2",
    ]

    # -------------------------------------------------------------------------
    # Model Architecture (PIRV-Net)
    # -------------------------------------------------------------------------
    # Kinematic Stream: Interleaved Pyramidal Architecture
    # Layers compress dimensions: Input -> 512 -> 256 -> 128 -> Output
    PYRAMID_LAYERS = [512, 256, 128]

    # Visual Stream: Lightweight Shallow MLP
    VISUAL_HIDDEN_DIM = 64

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Large batch size to stabilize Batch Normalization in Pyramidal layers
    BATCH_SIZE = 8192

    # Optimizer settings
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Training duration
    EPOCHS = 20
    PATIENCE = 3  # Early stopping patience

    # Focal Loss Parameters
    # Alpha=0.25 balances the 1:72 class imbalance
    # Gamma=2.0 focuses learning on hard examples
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # -------------------------------------------------------------------------
    # Inference & Evaluation
    # -------------------------------------------------------------------------
    # Number of steps for threshold grid search on validation set
    THRESHOLD_SEARCH_STEPS = 100
