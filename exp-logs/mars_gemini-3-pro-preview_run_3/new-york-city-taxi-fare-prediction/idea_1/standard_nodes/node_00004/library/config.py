import os

# -----------------------------------------------------------------------------
# Global Configuration & Reproducibility
# -----------------------------------------------------------------------------
SEED = 42

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------
# Input Data Paths (Metadata)
METADATA_DIR = "./metadata"
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Output Paths
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Directory for intermediate processing artifacts
CACHE_DIR = "./working/idea_2"

# Ensure necessary write directories exist
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Cleaning & Preprocessing Constants
# -----------------------------------------------------------------------------
# Bounding Box for New York City Area
# Coordinates falling outside this box will be treated as outliers/errors
NYC_MIN_LAT = 40.0
NYC_MAX_LAT = 42.0
NYC_MIN_LON = -75.0
NYC_MAX_LON = -73.0

# Value Constraints
MIN_FARE_AMOUNT = 0.0
MIN_PASSENGER_COUNT = 0
# While dataset max is 208, we filter unreasonable counts
MAX_PASSENGER_COUNT = 10

# -----------------------------------------------------------------------------
# Feature Engineering Constants
# -----------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
NYC_ROTATION_ANGLE = 29.0  # Degrees to align with Manhattan grid

# Key Landmarks (Lat, Lon)
LANDMARKS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "TSQ": (40.7580, -73.9855),  # Times Square
    "MET": (40.7794, -73.9632),  # Met Museum
    "WTC": (40.7126, -74.0099),  # World Trade Center
}

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
# Configuration for Histogram-based Gradient Boosting Regressor
# Optimized for large tabular datasets (speed and performance)
MODEL_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.05,  # Lower learning rate for better generalization
    "max_iter": 1000,  # Maximum number of boosting iterations (trees)
    "max_leaf_nodes": 63,  # Increased capacity for complex spatial splits
    "max_depth": None,  # No strict depth limit, controlled by leaf nodes
    "min_samples_leaf": 20,  # Minimum number of samples per leaf
    "l2_regularization": 0.0,  # L2 regularization term on weights
    "early_stopping": True,  # Enable early stopping
    "n_iter_no_change": 50,  # Patience for early stopping
    "verbose": 0,  # Silent mode
    "random_state": SEED,
}

# -----------------------------------------------------------------------------
# Execution & Compute Configuration
# -----------------------------------------------------------------------------
N_JOBS = 12  # Number of vCPUs available

# Dataset Sampling for Debugging/Fast Iteration
# Set to None to use full dataset, or an integer (e.g., 100000) for testing
TRAIN_SAMPLE_SIZE = None
