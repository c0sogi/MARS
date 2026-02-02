import os
import numpy as np

# =============================================================================
# PATHS AND DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_18"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input Data Paths (Metadata)
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache Paths for Processed Data
# These paths are used to store intermediate files to satisfy the deterministic caching requirement
PROCESSED_TRAIN_PATH = os.path.join(WORKING_DIR, "processed_train.parquet")
PROCESSED_VAL_PATH = os.path.join(WORKING_DIR, "processed_val.parquet")
PROCESSED_TEST_PATH = os.path.join(WORKING_DIR, "processed_test.parquet")
GLOBAL_STATS_PATH = os.path.join(WORKING_DIR, "global_stats.parquet")

# Output Path
SUBMISSION_OUTPUT_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONSTANTS & HYPERPARAMETERS
# =============================================================================
RANDOM_SEED = 42
NUM_CORES = 12

# Dataset Sizing
# "Select a stable 5M row subsample" for training as per the strategy
TRAIN_SAMPLE_SIZE = 5_000_000

# =============================================================================
# SPATIAL CONFIGURATION
# =============================================================================
# NYC Bounding Box: Strictly clamp coordinates to this range
# Lat 40-42, Lon -75 to -72
NYC_BOUNDING_BOX = {
    "lat_min": 40.0,
    "lat_max": 42.0,
    "lon_min": -75.0,
    "lon_max": -72.0,
}

# Grid Precision
# Used to simulate Geohash aggregation.
# 3 decimal places is approx 110m resolution, close to Geohash 7 (~150m)
GRID_PRECISION = 3

# =============================================================================
# DATA HYGIENE CRITERIA
# =============================================================================
# Wisdom Criteria: Strict filters for generating clean global statistics
WISDOM_CRITERIA = {
    "min_fare": 2.50,
    "max_fare": 200.0,
    "max_fare_per_km": 10.0,  # Filter out extreme outliers in rate
    "min_dist_km": 0.05,  # Minimum distance to avoid division by zero
}

# Learner Criteria: Loose filters for the training set to retain heavy tails
LEARNER_CRITERIA = {
    "min_fare": 2.50,
    "max_fare": 5000.0,  # Loose upper bound to catch valid high-fare rides
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# XGBoost Hyperparameters
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 5000,  # Upper limit, controlled by early stopping
    "early_stopping_rounds": 50,
    "tree_method": "hist",  # Efficient for large datasets
    "device": "cuda",  # GPU acceleration
    "n_jobs": NUM_CORES,
    "random_state": RANDOM_SEED,
    "verbosity": 0,
}

# Features to be used in the final model
MODEL_FEATURES = [
    # Raw Spatial
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    # Meta
    "passenger_count",
    # Temporal
    "hour",
    "weekday",
    "year",
    # Physics / Distance
    "dist_haversine",
    "dist_manhattan",
    # Distributional Priors (Structural Innovation)
    "route_mean_fare",
    "route_std_fare",
    # Orthogonal Features
    "temporal_fare_rate",
]
