import os
import numpy as np

# ==========================================
# PATH CONFIGURATION
# ==========================================
# Input Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working Directory (Cache)
# We create a specific directory for this idea's artifacts
WORKING_DIR = "./working/idea_5"
os.makedirs(WORKING_DIR, exist_ok=True)

# Data Paths
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# GLOBAL SETTINGS
# ==========================================
RANDOM_SEED = 42
DEBUG_MODE = False  # Set to True to use a small subset of data for debugging
DEBUG_SAMPLE_SIZE = 100_000  # Number of rows to use if DEBUG_MODE is True

# ==========================================
# DATA PREPROCESSING CONSTANTS
# ==========================================
# Bounding Box for NYC (Lat 40-42, Lon -73 to -75)
# Used to filter out coordinate outliers
BB_LAT_MIN = 40.0
BB_LAT_MAX = 42.0
BB_LON_MIN = -75.0
BB_LON_MAX = -73.0

# Target Value Constraints
# Filter training data to keep fare_amount within reasonable limits
FARE_MIN = 0.0
FARE_MAX = 500.0

# ==========================================
# FEATURE ENGINEERING CONSTANTS
# ==========================================
# Rotation Angle for Coordinate System Alignment (degrees)
# Aligns with the Manhattan street grid
ROTATION_ANGLE = 29.0

# Landmarks for Distance Features (Latitude, Longitude)
LANDMARKS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "WTC": (40.7126, -74.0099),  # One World Trade Center
    "MET": (40.7794, -73.9632),  # Metropolitan Museum of Art
    "TSQ": (40.7580, -73.9855),  # Times Square
}

# Earth Radius for Haversine Calculation (km)
R_EARTH_KM = 6371.0

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================
# Training Settings
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100

# 1. XGBoost Regressor Configuration
# Optimized for NVIDIA A100 GPU
XGB_PARAMS = {
    "n_estimators": 5000,
    "max_depth": 12,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "reg:squarederror",
    "tree_method": "gpu_hist",  # Use GPU acceleration
    "gpu_id": 0,
    "random_state": RANDOM_SEED,
    "n_jobs": 12,  # CPU threads for data loading/pre-processing
}

# 2. LightGBM Regressor Configuration
# Optimized for CPU (12 vCPUs)
LGBM_PARAMS = {
    "n_estimators": 5000,
    "max_depth": 12,
    "num_leaves": 512,  # 2^9, fits within max_depth=12
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "device": "cpu",
    "n_jobs": 12,  # Use all available vCPUs
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# Ensemble Weights
# Simple weighted average of the two models
WEIGHT_XGB = 0.5
WEIGHT_LGBM = 0.5
