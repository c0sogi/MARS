import os

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths
# Note: We will likely combine train and val for K-Fold CV to use the full dataset
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
FINAL_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Paths for processed data
CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "X_train_full_v2.parquet")
CACHE_TEST_PATH = os.path.join(WORKING_DIR, "X_test_full_v2.parquet")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42
ID_COL = "Id"
TARGET_COL = "Cover_Type"
N_FOLDS = 5  # Stratified K-Fold

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
# List of (Binary, Continuous) pairs for interaction features
# These target the weak sub-domains identified in previous iterations
INTERACTION_PAIRS = [
    ("Wilderness_Area1", "Elevation"),
    ("Wilderness_Area2", "Elevation"),
    ("Wilderness_Area3", "Elevation"),
    ("Wilderness_Area4", "Elevation"),
    ("Wilderness_Area1", "Horizontal_Distance_To_Hydrology"),
    ("Wilderness_Area2", "Horizontal_Distance_To_Hydrology"),
    ("Wilderness_Area3", "Horizontal_Distance_To_Hydrology"),
    ("Wilderness_Area4", "Horizontal_Distance_To_Hydrology"),
    ("Wilderness_Area1", "Horizontal_Distance_To_Roadways"),
    ("Wilderness_Area2", "Horizontal_Distance_To_Roadways"),
    ("Wilderness_Area3", "Horizontal_Distance_To_Roadways"),
    ("Wilderness_Area4", "Horizontal_Distance_To_Roadways"),
    ("Wilderness_Area1", "Horizontal_Distance_To_Fire_Points"),
    ("Wilderness_Area2", "Horizontal_Distance_To_Fire_Points"),
    ("Wilderness_Area3", "Horizontal_Distance_To_Fire_Points"),
    ("Wilderness_Area4", "Horizontal_Distance_To_Fire_Points"),
]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# LightGBM Parameters
# Optimized for NVIDIA A100 GPU
LGBM_PARAMS = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 3000,
    "early_stopping_rounds": 50,
    "random_state": SEED,
    "n_jobs": 12,
    "device": "cpu",
    "verbose": -1,
    # 'num_class' will be set dynamically based on the target encoding
}

# XGBoost Parameters
# Optimized for NVIDIA A100 GPU
XGB_PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "learning_rate": 0.02,
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 8000,
    "early_stopping_rounds": 50,
    "random_state": SEED,
    "n_jobs": 12,
    "tree_method": "gpu_hist",  # GPU acceleration
    "gpu_id": 0,
    # 'num_class' will be set dynamically based on the target encoding
}
