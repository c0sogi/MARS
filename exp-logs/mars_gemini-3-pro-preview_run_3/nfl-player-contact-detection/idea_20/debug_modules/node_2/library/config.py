import os

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_20"

# Ensure working directory exists for caching
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Input Data Paths
TRACKING_PATH_TRAIN = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TRACKING_PATH_TEST = os.path.join(INPUT_DIR, "test_player_tracking.csv")
HELMETS_PATH_TRAIN = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
HELMETS_PATH_TEST = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

# Output Paths
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
SEED = 42
DEBUG = False  # Set to True to run on a smaller subset for debugging

# Sampling Strategy
# Targeted Majority Undersampling: Keep 100% positives, subsample negatives
NEG_POS_RATIO = 10.0

# Temporal Context
# Sparse lags for Temporal Pyramids (applied as t +/- k)
LAG_OFFSETS = [0, 1, 2, 4, 8, 15]

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Stream A: Interaction Model (Player-Player)
# Strategy: Relational Scalars + Visual Pyramids + Cyclical Angles
STREAM_A_BASE_FEATURES = [
    # Relational Scalars (Explicit, not projected)
    "dist_p1_p2",
    "speed_diff",
    "closure_rate",
    # Cyclical Orientation/Direction
    "orientation_p1_sin",
    "orientation_p1_cos",
    "orientation_p2_sin",
    "orientation_p2_cos",
    "direction_p1_sin",
    "direction_p1_cos",
    "direction_p2_sin",
    "direction_p2_cos",
    # Visual Temporal Pyramids
    "sl_iou",
    "sl_dist",
    "ez_iou",
    "ez_dist",
]

# Stream B: Impact Model (Player-Ground)
# Strategy: Raw Field-Centric + Finite-Difference Ego-Dynamics
# Exclusion: No visual features, no relational context
STREAM_B_BASE_FEATURES = [
    # Raw Field-Centric Kinematics
    "x_position",
    "y_position",
    "speed",
    "acceleration",
    "orientation_sin",
    "orientation_cos",
    "direction_sin",
    "direction_cos",
    # Finite-Difference Ego-Dynamics (New Logic)
    "v_surge",
    "v_sway",  # Projected velocity
    "a_surge",
    "a_sway",  # Derivative of projected velocity
]


def get_feature_cols(base_features, lags):
    """
    Generates the full list of feature column names including lags.
    Naming convention: {feature}, {feature}_lag_{k}, {feature}_lag_minus_{k}
    """
    cols = []
    for f in base_features:
        # Current timestep (lag 0)
        if 0 in lags:
            cols.append(f)

        # Past and Future lags
        for k in lags:
            if k == 0:
                continue
            cols.append(f"{f}_lag_{k}")  # Future context (t + k)
            cols.append(f"{f}_lag_minus_{k}")  # Past context (t - k)

    return sorted(list(set(cols)))


# Generate exhaustive feature lists
STREAM_A_COLS = get_feature_cols(STREAM_A_BASE_FEATURES, LAG_OFFSETS)
STREAM_B_COLS = get_feature_cols(STREAM_B_BASE_FEATURES, LAG_OFFSETS)

# =============================================================================
# MODEL HYPERPARAMETERS (XGBoost)
# =============================================================================

# Common parameters for GPU-accelerated training
XGB_COMMON_PARAMS = {
    "booster": "gbtree",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",  # Use NVIDIA A100
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

# Stream A Params: Standard Regularization
# Optimized for interaction detection where overfitting is a risk
STREAM_A_PARAMS = XGB_COMMON_PARAMS.copy()
STREAM_A_PARAMS.update(
    {
        "max_depth": 6,
        "learning_rate": 0.05,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "min_child_weight": 1,
        "n_estimators": 5000,  # Controlled by early stopping
    }
)

# Stream B Params: Relaxed Regularization (Asymmetric)
# Optimized for ground impact which requires capturing complex ego-motion patterns
STREAM_B_PARAMS = XGB_COMMON_PARAMS.copy()
STREAM_B_PARAMS.update(
    {
        "max_depth": 12,  # Increased depth for complex kinematic interactions
        "learning_rate": 0.05,
        "colsample_bytree": 0.9,
        "subsample": 0.9,
        "min_child_weight": 1,  # Lower weight to capture specific impact signatures
        "n_estimators": 5000,
    }
)

# Training Loop Settings
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100
