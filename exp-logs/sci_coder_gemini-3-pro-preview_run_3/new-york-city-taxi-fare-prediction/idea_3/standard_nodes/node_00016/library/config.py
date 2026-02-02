import os

# ==========================================
# PATH CONFIGURATION
# ==========================================
METADATA_DIR = "./metadata"
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_opt")
# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
# Ensure submission directory exists
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# GLOBAL CONSTANTS
# ==========================================
SEED = 42

# Data Sanitization
MIN_FARE = 0.0
MAX_FARE = 500.0

# Spatial Feature Engineering
ROTATION_ANGLE = 29  # Degrees for coordinate rotation to align with NYC grid
N_CLUSTERS = 500  # Number of clusters for spatial target encoding

# Landmarks (Latitude, Longitude)
# Used for calculating distance features
LANDMARKS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "TSQ": (40.7580, -73.9855),  # Times Square
    "WTC": (40.7127, -74.0134),  # World Trade Center / Freedom Tower
    "NYC": (40.7141667, -74.0063889),  # General NYC Center
}

# ==========================================
# DEBUG / SAMPLING CONTROL
# ==========================================
# Set to an integer (e.g., 100000) to debug with a subset, or None for full data
MAX_TRAIN_SAMPLES = None

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================

# XGBoost Configuration
# Optimized for NVIDIA A100 GPU
XGB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "gpu_hist",  # Explicitly use GPU histogram algorithm
    "gpu_id": 0,
    "n_jobs": -1,
    "random_state": SEED,
    "early_stopping_rounds": 50,
}

# LightGBM Configuration
# Optimized for 12 vCPUs
LGBM_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "num_leaves": 512,  # High capacity for large dataset
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_jobs": 12,  # Utilize all available CPU cores
    "random_state": SEED,
    "early_stopping_rounds": 50,
    "verbosity": -1,
    "device": "cpu",
}
