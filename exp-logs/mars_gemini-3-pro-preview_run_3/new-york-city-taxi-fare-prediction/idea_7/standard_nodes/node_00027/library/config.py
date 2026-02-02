import os
import numpy as np

# ==========================================
# Global Configuration
# ==========================================

# Random Seed for Reproducibility across all libraries
SEED = 42

# ==========================================
# File Paths and Directories
# ==========================================

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_opt"
SUBMISSION_DIR = "./submission"

# Create necessary writable directories
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths (using Metadata Parquet files for speed)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Cleaning & Sanitation Constants
# ==========================================

# Bounding Box for NYC Area
# Used to filter out coordinate outliers that distort distance calculations
BB_MIN_LAT = 40.0
BB_MAX_LAT = 42.0
BB_MIN_LON = -75.0
BB_MAX_LON = -73.0

# Target Variable Constraints
# Used to filter training data to remove extreme fare outliers
MIN_FARE = 0.0
MAX_FARE = 500.0

# ==========================================
# Feature Engineering Constants
# ==========================================

# Coordinate Rotation
# 29 degrees aligns the coordinate system with the Manhattan street grid
ROTATION_ANGLE_DEG = 29
ROTATION_ANGLE_RAD = np.radians(ROTATION_ANGLE_DEG)

# Earth Radius in km (for Haversine calculation)
R_EARTH = 6371.0

# Key Landmarks (Lat, Lon)
# Used to calculate distance-to-hub features
LANDMARKS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "WTC": (40.7126, -74.0099),  # World Trade Center (Downtown)
    "TS": (40.7580, -73.9855),  # Times Square (Midtown)
    "MET": (40.7794, -73.9632),  # Metropolitan Museum (Uptown)
}

# Airport Bounding Boxes
# Used for creating explicit binary flags (is_JFK, is_LGA, etc.)
# Format: (min_lat, max_lat, min_lon, max_lon)
AIRPORT_BOXES = {
    "JFK": (40.62, 40.66, -73.83, -73.75),
    "LGA": (40.76, 40.79, -73.89, -73.85),
    "EWR": (40.67, 40.71, -74.19, -74.15),
}

# ==========================================
# Model Hyperparameters
# ==========================================

# XGBoost Regressor (Base Learner 1)
# Configured for NVIDIA A100 GPU acceleration
XGB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "max_depth": 12,  # Deep trees to capture complex spatial boundaries
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "device": "cuda",  # Use GPU
    "tree_method": "hist",  # Optimized histogram algorithm
    "n_jobs": 12,
    "random_state": SEED,
}

# LightGBM Regressor (Base Learner 2)
# Configured for CPU (12 vCPUs)
LGBM_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "num_leaves": 512,  # High leaf count for fine-grained splits
    "max_depth": -1,  # Unlimited depth, controlled by num_leaves
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_jobs": 12,
    "random_state": SEED,
    "verbose": -1,
}

# Ridge Regression (Meta Learner)
# Used for stacking the base learners
RIDGE_PARAMS = {
    "alpha": 10.0,
    "random_state": SEED,
}

# Training Control
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100
