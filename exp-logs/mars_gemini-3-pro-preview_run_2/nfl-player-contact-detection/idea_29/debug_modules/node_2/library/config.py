import os
import torch


class Config:
    """
    Configuration for the Stabilized Entity-Aware Residual-Visual Network (SEA-RVN).
    Centralizes all file paths, hyperparameters, and feature definitions.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_29"
    SUBMISSION_DIR = "./submission"

    # Generated Metadata Paths
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Paths
    TRAIN_TRACKING = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TRAIN_HELMETS = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Paths (Parquet format preferred over pickle)
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.parquet")

    # =========================================================================
    # Reproducibility & Compute
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # =========================================================================
    # Data Processing Hyperparameters
    # =========================================================================
    # Debugging: Set to True to train on a small subset
    DEBUG = False
    DEBUG_SAMPLES = 10000

    # Temporal Window: t-WINDOW to t+WINDOW (Total 2*WINDOW + 1 steps)
    WINDOW_SIZE = 5

    # Numerical Stability: Clamping range for continuous kinematic features
    # Prevents outliers in derivative features from destabilizing gradients
    CLAMP_MIN = -100.0
    CLAMP_MAX = 100.0

    # =========================================================================
    # Feature Definitions
    # =========================================================================
    # Continuous Kinematic Features (to be clamped and standardized)
    # Includes base tracking metrics and derived physics
    KINEMATIC_CONT_FEATURES = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "orientation",
        "direction",
        "sa",
        # Derived features assumed to be calculated during preprocessing:
        "distance",
        "closing_speed",
        "relative_angle",
    ]

    # Categorical Kinematic Features (for Entity Embeddings)
    # Captures role-based priors (e.g., QB vs LB)
    KINEMATIC_CAT_FEATURES = ["position", "team"]

    # Visual Features (for Visual Correction Stream)
    # Derived from Max-Pooling Selection Strategy on Helmet Boxes
    VISUAL_FEATURES = ["left", "width", "top", "height"]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Embedding Dimensions for Categorical Features
    # Format: {Feature Name: (Num Categories, Embedding Dim)}
    # Note: Num Categories is approximate; Preprocessor should handle exact count + Unknown
    EMBEDDING_DIMS = {
        "position": (30, 8),  # Approx 28 positions + Unknown/Pad
        "team": (5, 4),  # Home, Away, Ground, Unknown
    }

    # Kinematic Stream (Deep Residual MLP)
    # Input dim is auto-calculated based on window size and feature counts
    KIN_HIDDEN_DIMS = [512, 256, 128]

    # Visual Stream (Shallow MLP for Correction)
    VIS_HIDDEN_DIMS = [64, 32]

    # Regularization
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Batch Size: Large batch size for stable gradients on tabular data
    BATCH_SIZE = 512

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 10  # Full dataset converges relatively quickly

    # Focal Loss Parameters (Critical for Class Imbalance)
    # alpha=0.25 balances precision/recall
    # gamma=2.0 focuses learning on hard examples
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # Evaluation
    PATIENCE = 3  # Early stopping patience
    THRESHOLD_SEARCH_STEPS = 100  # Granularity for threshold optimization

    @classmethod
    def setup(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
