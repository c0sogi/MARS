import os

# ==============================================================================
# Global Constants & Paths
# ==============================================================================
SEED = 42

# Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
IDEA_DIR = os.path.join(WORKING_DIR, "idea_20")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

# Ensure working directories exist
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

SUBMISSION_OUTPUT_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==============================================================================
# Feature Engineering & Gating Configuration
# ==============================================================================
# Relaxed Quadratic Reachability Gating
# Threshold relaxed to 3.0 yards to capture near-misses and high-speed impacts
GATING_THRESHOLD = 3.0

# Sentinel Value Strategy
# Distinct non-physical distance for Player-Ground interactions to allow tree splits
SENTINEL_VALUE = -1.0

# Spectral-Kinematic Features
# Window size in steps (10 steps = 1.0 second at 10Hz)
WINDOW_SIZE = 10
USE_SPECTRAL_FEATURES = True

# ==============================================================================
# Training Configuration
# ==============================================================================
# Curriculum Learning Parameters
SCOUT_TRAIN_RATIO = 1.0  # Ratio of Negatives to Positives for initial Scout training
HARD_NEGATIVE_THRESHOLD = (
    0.05  # Probability threshold to consider a negative instance "Hard"
)
EXPERT_BUFFER_RATIO = (
    1.0  # Ratio of random negatives to retain in Expert set alongside hard negatives
)
EARLY_STOPPING_ROUNDS = 50

# ==============================================================================
# Model Hyperparameters
# ==============================================================================

# 1. LightGBM (Leaf-wise growth)
# Configured for high capacity and imbalance handling
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "num_leaves": 256,  # High capacity for complex decision boundaries
    "max_depth": 10,  # Deep trees
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "is_unbalance": True,  # Explicit imbalance handling
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# 2. XGBoost (Level-wise growth)
# Configured for high capacity and approximate splitting
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "max_depth": 10,  # High capacity
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",  # Fast histogram-based training
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
    # Note: scale_pos_weight is typically handled dynamically in training script
}

# 3. CatBoost (Oblivious/Symmetric trees)
# Configured for structural diversity and native imbalance handling
CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 10,  # High capacity
    "auto_class_weights": "Balanced",  # Native imbalance handling
    "verbose": 0,
    "random_seed": SEED,
    "allow_writing_files": False,
    "thread_count": -1,
}
