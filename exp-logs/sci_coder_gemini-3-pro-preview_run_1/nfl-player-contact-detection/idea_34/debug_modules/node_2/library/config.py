import os

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_34"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Tracking Data Paths
TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42
N_JOBS = 12  # Utilizing available vCPUs

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================
# Temporal Window: [-10, +10] frames around the target step
WINDOW_SIZE = 10

# Relaxed Quadratic Gating
GATING_THRESHOLD = 3.0  # Yards
GROUND_DISTANCE_SENTINEL = -1.0

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
# Base features calculated for every lag step in the window
# These correspond to the Reference-Anchored Decoupled projections
BASE_FEATURES = [
    "dist",
    # Player 1 Projected Kinetics & Orientation
    "p1_v_long",
    "p1_v_trans",
    "p1_a_long",
    "p1_a_trans",
    "p1_o_long",
    "p1_o_trans",
    # Player 2 Projected Kinetics & Orientation
    "p2_v_long",
    "p2_v_trans",
    "p2_a_long",
    "p2_a_trans",
    "p2_o_long",
    "p2_o_trans",
    # Scalar fallbacks
    "speed_p1",
    "speed_p2",
    "accel_p1",
    "accel_p2",
]

# Generate the exhaustive list of feature columns including lags
FEATURE_COLUMNS = []
for lag in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
    for feat in BASE_FEATURES:
        FEATURE_COLUMNS.append(f"{feat}_lag{lag}")

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Mining Strategy
SCOUT_THRESHOLD = 0.05  # Probability threshold for Hard Negative mining
ANCHOR_RATIO = 1.0  # 1:1 Ratio of Random Easy Negatives to Positives

# General Training Params
NUM_ESTIMATORS = 2000
LEARNING_RATE = 0.02
EARLY_STOPPING_ROUNDS = 50

# =============================================================================
# MODEL HYPERPARAMETERS (TRI-ENSEMBLE)
# =============================================================================

# 1. LightGBM (Leaf-wise growth, deep trees)
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": NUM_ESTIMATORS,
    "learning_rate": LEARNING_RATE,
    "num_leaves": 256,
    "max_depth": 10,
    "is_unbalance": True,
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "verbose": -1,
}

# 2. XGBoost (Level-wise growth, deep trees)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "booster": "gbtree",
    "n_estimators": NUM_ESTIMATORS,
    "learning_rate": LEARNING_RATE,
    "max_depth": 10,
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "verbosity": 0,
    # Note: scale_pos_weight to be calculated dynamically in training script
}

# 3. CatBoost (Symmetric/Oblivious trees)
CAT_PARAMS = {
    "loss_function": "Logloss",
    "iterations": NUM_ESTIMATORS,
    "learning_rate": LEARNING_RATE,
    "depth": 10,
    "auto_class_weights": "Balanced",
    "random_seed": SEED,
    "verbose": 0,
    "allow_writing_files": False,
    "thread_count": N_JOBS,
}
