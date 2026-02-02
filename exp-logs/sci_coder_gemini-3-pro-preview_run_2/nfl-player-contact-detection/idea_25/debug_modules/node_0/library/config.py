import os
import torch


class Config:
    """
    Configuration for the Squeeze-and-Excitation Residual-Visual Network (SE-RVN)
    and the associated data/training pipeline.
    """

    # =========================================================================
    # Paths and Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_25"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Reproducibility and Hardware
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    # =========================================================================
    # Data Processing Pipeline
    # =========================================================================
    # Temporal Window: t-5 to t+5 (11 frames total at 10Hz)
    WINDOW_SIZE = 5
    TOTAL_FRAMES = (WINDOW_SIZE * 2) + 1

    # Caching
    CACHE_DATA = True

    # Numerical Stability & Engineering
    # Clamping derived kinematic features to prevent unbounded gradients
    CLAMP_MIN = -50.0
    CLAMP_MAX = 50.0

    # Ground Imputation
    GROUND_PLAYER_ID = "G"

    # Feature Selection
    # Raw tracking features to utilize
    TRACKING_FEATS = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
        "orientation",
        "sa",
    ]

    # Derived kinematic features
    DERIVED_FEATS = ["distance", "rel_speed", "rel_accel", "rel_angle_o", "rel_angle_d"]

    # Visual features from helmet boxes (Endzone/Sideline/All29)
    VISUAL_FEATS = ["left", "top", "width", "height"]

    # Categorical features for Entity Embeddings
    CAT_FEATS = ["position", "team"]

    # =========================================================================
    # Model Architecture (SE-RVN)
    # =========================================================================
    # Kinematic Stream (SE-Residual Backbone)
    KIN_INPUT_DIM = 0  # Calculated dynamically based on window * features
    KIN_HIDDEN_DIM = 256
    NUM_RES_BLOCKS = 3
    SE_REDUCTION_RATIO = 4
    KIN_DROPOUT = 0.1

    # Entity Embeddings
    # Dimensions: (num_categories, embedding_dim)
    # Note: Actual vocab sizes are determined during preprocessing
    EMBEDDING_DIM_POS = 16
    EMBEDDING_DIM_TEAM = 4

    # Visual Stream (Shallow Correction)
    VIS_HIDDEN_DIM = 64
    VIS_DROPOUT = 0.1

    # Reliability Gating
    # No specific param here, logic is in the model definition

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Optimization
    BATCH_SIZE = 2048  # Large batch size for efficient tabular training
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduling & Stopping
    EPOCHS = 20
    PATIENCE = 3  # Early stopping patience

    # Loss Function (Focal Loss)
    # Alpha balances positive/negative class importance
    # Gamma focuses on hard examples
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # =========================================================================
    # Inference
    # =========================================================================
    # Threshold will be optimized on validation set, but we define a default
    DEFAULT_THRESHOLD = 0.5
