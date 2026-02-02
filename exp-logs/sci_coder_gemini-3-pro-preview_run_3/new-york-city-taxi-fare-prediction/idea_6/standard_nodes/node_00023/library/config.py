import os
import numpy as np


class Config:
    """
    Configuration module for the Taxi Fare Prediction task (Idea 6).
    Defines global constants, file paths, and model hyperparameters for the
    High-Capacity Heterogeneous Ensemble strategy.
    """

    # ==========================================
    # 1. Global Setup
    # ==========================================
    SEED = 42
    IDEA_NAME = "idea_6"

    # ==========================================
    # 2. File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Paths (Parquet Metadata)
    # These files are already generated and contain the splits
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Processed Data Cache Paths
    # Used to store feature-engineered datasets to save time on re-runs
    TRAIN_PROCESSED_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_PROCESSED_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_PROCESSED_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Output Submission Path
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. Data Cleaning & Preprocessing
    # ==========================================
    # Bounding Box Filter to remove coordinate outliers
    # NYC Longitude is negative (-75 is West of -73)
    BOUNDING_BOX = {
        "min_lat": 40.0,
        "max_lat": 42.0,
        "min_lon": -75.0,
        "max_lon": -73.0,
    }

    # Target Variable Filter (Fare Amount)
    # Filters out negative fares and extreme outliers > $500
    FARE_RANGE = (0, 500)

    # ==========================================
    # 4. Feature Engineering
    # ==========================================
    # Coordinate Rotation for Manhattan Distance (aligned with NYC grid)
    # 29 degrees is the approximate angle of the Manhattan street grid
    ROTATION_ANGLE_DEG = 29
    ROTATION_ANGLE_RAD = np.radians(ROTATION_ANGLE_DEG)

    # Landmarks for Haversine Distance Features
    # Selected major hubs to provide global spatial context
    LANDMARKS = {
        "JFK": (40.6413, -73.7781),
        "LGA": (40.7769, -73.8740),
        "EWR": (40.6895, -74.1745),
        "WTC": (40.7126, -74.0099),
        "TSQ": (40.7580, -73.9855),
    }

    # ==========================================
    # 5. Model Hyperparameters
    # ==========================================
    # XGBoost Regressor (GPU-Accelerated)
    # Configured for NVIDIA A100 usage
    XGB_PARAMS = {
        "n_estimators": 10000,
        "learning_rate": 0.05,
        "max_depth": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "gpu_hist",  # GPU Histogram algorithm
        "device": "cuda",  # Explicitly set device to CUDA
        "random_state": SEED,
        "n_jobs": -1,
    }

    # LightGBM Regressor (CPU-Optimized)
    # Configured for 12 vCPU usage
    LGBM_PARAMS = {
        "n_estimators": 5000,
        "learning_rate": 0.02,
        "num_leaves": 512,  # High leaf count for capacity
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "device": "cpu",
        "n_jobs": 12,  # Utilize all available vCPUs
        "random_state": SEED,
        "verbose": -1,  # Silent mode
    }

    # Ensemble Weights (Single Strong Model)
    ENSEMBLE_WEIGHTS = {"xgb": 1.0, "lgbm": 0.0}

    # Training Control
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    # Debugging / Sampling
    # Set to None to use full dataset (55M rows), or an integer (e.g., 100000) for debugging
    DEBUG_SAMPLE_SIZE = None
