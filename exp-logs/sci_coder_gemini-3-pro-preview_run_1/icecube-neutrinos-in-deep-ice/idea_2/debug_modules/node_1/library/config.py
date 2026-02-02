import os

# ==========================================
# Global Configuration
# ==========================================
SEED = 42

# ==========================================
# Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# File Paths
# ==========================================
# Input Files
GEOMETRY_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files (Generated previously)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.parquet")

# Output Submission
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Files for Processed Features
# These store the fixed-size feature vectors extracted from the raw pulse data
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# ==========================================
# Feature Engineering
# ==========================================
# List of feature names to be used for training and inference
FEATURE_NAMES = [
    # Charge-Weighted Centroids (Center of Gravity)
    "center_x",
    "center_y",
    "center_z",
    # Signal Magnitude
    "total_charge",
    "n_pulses",
    # Temporal Evolution (Wavefront arrival and duration)
    "time_10",  # 10th percentile (early arrival)
    "time_50",  # Median time
    "time_90",  # 90th percentile (tail)
    # Spatial Spread (Compactness of the event)
    "spread_x",  # Std dev of x positions
    "spread_y",  # Std dev of y positions
    "spread_z",  # Std dev of z positions
]

# ==========================================
# Model Hyperparameters
# ==========================================
# Parameters for the LightGBM Regressors
# We train separate models for x, y, and z vector components
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": 12,
    "n_estimators": 2000,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbosity": -1,
    "n_jobs": 12,
    "seed": SEED,
}

# ==========================================
# Runtime Options
# ==========================================
# Set to an integer (e.g., 100000) to limit training data for debugging/fast iteration
# Set to None to use the full dataset
DEBUG_N_ROWS = None
