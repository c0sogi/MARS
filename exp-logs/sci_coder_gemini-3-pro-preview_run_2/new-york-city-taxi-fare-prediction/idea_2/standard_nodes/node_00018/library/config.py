import os

# ==========================================
# Global Configuration and Constants
# ==========================================

# Random Seed for Reproducibility
SEED = 42

# Directory for caching intermediate files (e.g. sanitized datasets)
CACHE_DIR = "./working/idea_optimized/"
os.makedirs(CACHE_DIR, exist_ok=True)

# File Paths
PATH_CONFIG = {
    "train_data": "./metadata/train.parquet",
    "val_data": "./metadata/val.parquet",
    "test_data": "./input/test.csv",
    "submission_output": "./submission/submission.csv",
    "model_save_path": os.path.join(CACHE_DIR, "xgb_model.json"),
}

# ==========================================
# Data Sanitization & Feature Engineering
# ==========================================

# Bounding Box for Coordinate Sanitization (NYC Area)
# Coordinates falling outside this box will be clamped to these limits.
BOUNDING_BOX = {"lat_min": 40.0, "lat_max": 42.0, "lon_min": -75.0, "lon_max": -72.0}

# Major Hub Locations for Feature Engineering
# (Latitude, Longitude)
HUB_LOCATIONS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "TSQ": (40.7580, -73.9855),  # Times Square
}

# ==========================================
# Model Hyperparameters
# ==========================================

# XGBoost Parameters
# Optimized for NVIDIA A100 GPU using 'hist' tree method on 'cuda' device.
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",  # Efficient histogram-based algorithm
    "device": "cuda",  # Enable GPU acceleration
    "learning_rate": 0.05,
    "max_depth": 10,  # Deeper trees to capture complex spatial partitions
    "subsample": 0.8,  # Row subsampling to prevent overfitting
    "colsample_bytree": 0.8,  # Column subsampling
    "min_child_weight": 10,
    "n_jobs": 12,  # Number of CPU threads for data loading/pre-processing
    "random_state": SEED,
}

# Training Loop Configuration
TRAIN_CONFIG = {
    "num_boost_round": 10000,  # Maximum number of trees
    "early_stopping_rounds": 50,  # Stop if validation RMSE doesn't improve
    "verbose_eval": 100,  # Print metrics every 100 rounds
}
