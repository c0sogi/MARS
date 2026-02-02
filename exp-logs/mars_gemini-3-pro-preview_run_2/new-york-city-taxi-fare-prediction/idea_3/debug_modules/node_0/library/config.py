import os

# Global Random Seed for Reproducibility
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Directory Definitions
# -----------------------------------------------------------------------------
# Input metadata directory containing generated Parquet files
INPUT_DIR = "./metadata"

# Working directory for caching processed data and models
WORKING_DIR = "./working/idea_3"

# Submission directory
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------
DATA_PATHS = {
    # Input Data (Metadata)
    "train_parquet": os.path.join(INPUT_DIR, "train.parquet"),
    "val_parquet": os.path.join(INPUT_DIR, "val.parquet"),
    "test_parquet": os.path.join(INPUT_DIR, "test.parquet"),
    "sample_submission": "./input/sample_submission.csv",
    # Processed Data Cache
    "train_processed": os.path.join(WORKING_DIR, "train_processed.parquet"),
    "val_processed": os.path.join(WORKING_DIR, "val_processed.parquet"),
    "test_processed": os.path.join(WORKING_DIR, "test_processed.parquet"),
    # Output Submission
    "submission": os.path.join(SUBMISSION_DIR, "submission.csv"),
}

# -----------------------------------------------------------------------------
# Data Cleaning & Filtering Parameters
# -----------------------------------------------------------------------------
CLEANING_PARAMS = {
    # Bounding Box for NYC Coordinates (Clamping/Filtering)
    # Lat 40-42, Lon -75 to -72 covers NYC and surrounding metro area
    "lat_min": 40.0,
    "lat_max": 42.0,
    "lon_min": -75.0,
    "lon_max": -72.0,
    # Consistency Filtering
    # Remove rows where Fare is suspiciously high for a very short distance
    # Rule: Drop if Fare > threshold AND Distance < threshold
    "inconsistent_fare_threshold": 100.0,  # Dollars
    "inconsistent_distance_threshold_km": 1.0,  # Kilometers
    # Post-Processing
    # Minimum fare floor to apply to final predictions
    "min_fare_floor": 2.50,
}

# -----------------------------------------------------------------------------
# Feature Engineering Parameters
# -----------------------------------------------------------------------------
FEATURE_PARAMS = {
    # Create rotated coordinates (45 degrees) to help tree models with diagonal distances
    "use_rotation": True,
    "rotation_angle": 45,
}

# -----------------------------------------------------------------------------
# Ensemble Configuration
# -----------------------------------------------------------------------------
ENSEMBLE_CONFIG = {
    # Number of independent XGBoost models to train
    "n_models": 5,
    # Strategy for data allocation:
    # 'partition': Split the training set into n_models non-overlapping chunks.
    # 'bootstrap': Sample with replacement (not used here, using partition for stability).
    "strategy": "partition",
}

# -----------------------------------------------------------------------------
# XGBoost Hyperparameters
# -----------------------------------------------------------------------------
# Optimized for GPU acceleration and RMSE minimization
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    # Hardware Acceleration
    "tree_method": "hist",
    "device": "cuda",
    # Tree Structure & Regularization
    "learning_rate": 0.05,
    "max_depth": 9,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 10,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    # Training Loop
    "n_estimators": 5000,  # High cap, controlled by early stopping
    "early_stopping_rounds": 50,  # Stop if validation score doesn't improve
    # Execution
    "n_jobs": 12,
    "random_state": RANDOM_SEED,
    "verbosity": 0,
}
