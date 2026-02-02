import os
import numpy as np

# =============================================================================
# GLOBAL CONSTANTS & SETUP
# =============================================================================

SEED = 42
N_JOBS = 12
USE_GPU = True

# Set random seeds for reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

# =============================================================================
# PATH CONFIGURATION
# =============================================================================


class PathConfig:
    """Defines input and output paths for the project."""

    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Cite debug_lesson_5
    WORKING_DIR = "./working/idea_32"
    SUBMISSION_DIR = "./submission"

    # Input Files (Metadata)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Input Files (Tracking & Others)
    TRAIN_TRACKING = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Parquet/NPY)
    # These will be generated in the WORKING_DIR
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "features_train.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "features_val.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "features_test.parquet")
    CACHE_HARD_NEGATIVES = os.path.join(WORKING_DIR, "hard_negative_indices.npy")

    @classmethod
    def setup_directories(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories immediately
PathConfig.setup_directories()

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================


class FeatureConfig:
    """Configuration for Time-Domain Vector-Aligned Feature Engineering."""

    # Window size in steps (1 step = 0.1s)
    # Window of [-10, +10] corresponds to +/- 1.0 second
    WINDOW_SIZE = 10

    # Sentinel value for Ground distance (distinct non-physical value)
    GROUND_DISTANCE_SENTINEL = -1.0

    # Feature Flags
    USE_VECTOR_ALIGNED_LAGS = True  # Radial/Tangential projection
    USE_JERK = True  # Derivative of acceleration
    USE_TTC = True  # Time-To-Collision

    # Normalization / Scaling
    # If True, features might be scaled, but tree models generally don't require it.
    NORMALIZE_FEATURES = False


# =============================================================================
# GATING CONFIGURATION
# =============================================================================


class GatingConfig:
    """Configuration for Relaxed Quadratic Reachability Gating."""

    # Distance threshold in yards
    # Pairs with min(quadratic_dist) > threshold are discarded early
    REACHABILITY_THRESHOLD = 3.0

    # Time horizon for quadratic projection (steps)
    PROJECTION_STEPS = 10


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================


class TrainConfig:
    """Configuration for the Training Curriculum."""

    # Hard Negative Mining
    HARD_NEGATIVE_THRESHOLD = 0.05  # Probability threshold for mining

    # Anchored Sampling
    # Ratio of Random Easy Negatives to Hard Negatives in the final Expert set
    ANCHOR_RATIO = 1.0

    # Temporal Label Smoothing
    # Sigma for Gaussian smoothing of binary labels across time steps
    LABEL_SMOOTHING_SIGMA = 1.0

    # Class Imbalance Handling
    # Scale pos weight for XGB/LGBM
    POS_WEIGHT_SCALE = 10.0


# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# LightGBM Params (Leaf-wise)
LGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "num_leaves": 256,
    "max_depth": 10,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary",
    "metric": "binary_logloss",
    "n_jobs": N_JOBS,
    "verbose": -1,
    "random_state": SEED,
    "device": "gpu" if USE_GPU else "cpu",
}

# XGBoost Params (Level-wise)
XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "max_depth": 10,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "tree_method": "gpu_hist" if USE_GPU else "hist",
}

# Dictionary aggregating all model configs
MODEL_PARAMS = {"lgbm": LGBM_PARAMS, "xgb": XGB_PARAMS}
