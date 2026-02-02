import os

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_12"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Data Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

# Cache Files (Parquet/NPY)
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")
HARD_NEGATIVE_INDICES_PATH = os.path.join(WORKING_DIR, "hard_negative_indices.npy")
SCOUT_PREDS_PATH = os.path.join(WORKING_DIR, "scout_predictions.npy")

# Submission
SUBMISSION_PATH = "./submission/submission.csv"
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42
NUM_CORES = 12

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
# Geometric Gating: Discard pairs further than this distance immediately
GATING_DISTANCE = 2.5  # Yards

# Temporal Window: Number of steps before/after contact to include
WINDOW_SIZE = 10  # Timesteps (+/- 1.0 second at 10Hz)

# Polar Interaction Grid
# Ego-centric grid aligned to player orientation for spatial feature pooling
POLAR_GRID_SETTINGS = {
    "num_sectors": 4,  # Front, Back, Left, Right (90 deg each)
    "radial_bands": [0, 1, 2],  # Yards: 0-1, 1-2. (Edges)
    "sector_offset": 45,  # Degrees offset to align sectors (e.g. Front is -45 to +45)
    "use_velocity_flux": True,  # Compute radial (converging) and tangential (orbiting) flux
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# 1. Scout Model (LightGBM)
# Purpose: Fast inference to mine hard negatives from the gated survivor pool.
LGBM_SCOUT_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": 8,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": SEED,
    "n_jobs": NUM_CORES,
}

# 2. Expert Model (LightGBM)
# Purpose: High-capacity final classification using deep trees.
LGBM_EXPERT_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 256,
    "max_depth": 10,
    "is_unbalance": True,  # Handle class imbalance explicitly
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": SEED,
    "n_jobs": NUM_CORES,
}

# 3. Expert Model (XGBoost)
# Purpose: Heterogeneous ensemble component.
XGB_EXPERT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.02,
    "max_depth": 10,
    "tree_method": "hist",
    "device": "cuda",  # Utilize A100 GPU
    "scale_pos_weight": 10,  # Weight for positive class (approximate for mining)
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": SEED,
    "n_jobs": NUM_CORES,
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
TRAIN_CONFIG = {
    "scout_rounds": 1000,
    "expert_rounds": 3000,
    "early_stopping_rounds": 50,
    "verbose_eval": 100,
    "hard_negative_threshold": 0.05,  # Probability threshold to mine negatives
    "negative_sampling_ratio": 1.0,  # Ratio of random negatives to positives for Scout training
}


def get_config(debug=False):
    """
    Retrieves configuration dictionaries, adjusting for debug mode if specified.

    Args:
        debug (bool): If True, reduces training rounds and model complexity for fast testing.

    Returns:
        tuple: (lgbm_scout_params, lgbm_expert_params, xgb_expert_params, train_config)
    """
    # Copy to avoid mutation
    l_scout = LGBM_SCOUT_PARAMS.copy()
    l_expert = LGBM_EXPERT_PARAMS.copy()
    x_expert = XGB_EXPERT_PARAMS.copy()
    t_config = TRAIN_CONFIG.copy()

    if debug:
        # Reduce rounds for quick validation
        t_config["scout_rounds"] = 50
        t_config["expert_rounds"] = 50

        # Add n_estimators constraint for sklearn-wrapper compatibility if used
        l_scout["n_estimators"] = 50
        l_expert["n_estimators"] = 50
        x_expert["n_estimators"] = 50

    return l_scout, l_expert, x_expert, t_config
