import os

# --- Base Directories ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_optimized"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# --- Data Paths ---
DATA_PATHS = {
    "train": os.path.join(METADATA_DIR, "train.parquet"),
    "val": os.path.join(METADATA_DIR, "val.parquet"),
    "test": os.path.join(METADATA_DIR, "test.parquet"),
    "submission": os.path.join(SUBMISSION_DIR, "submission.csv"),
    "cache_dir": WORKING_DIR,
}

# --- Feature Engineering Configuration ---
FEATURE_CONFIG = {
    "target_col": "fare_amount",
    "id_col": "key",
    "datetime_col": "pickup_datetime",
    # Raw numerical features to retain
    "raw_features": [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    # Names of features to be generated during processing
    "generated_features": [
        "distance_haversine",
        "distance_manhattan",
        "abs_diff_lon",
        "abs_diff_lat",
        "pickup_hour",
        "pickup_weekday",
        "pickup_month",
        "pickup_year",
        "months_since_2009",
        "pickup_dist_jfk",
        "dropoff_dist_jfk",
        "pickup_dist_lga",
        "dropoff_dist_lga",
        "pickup_dist_ewr",
        "dropoff_dist_ewr",
    ],
    # Landmarks for distance features (Cite solution_lesson_node_00002)
    "landmarks": {
        "jfk": (40.6413, -73.7781),
        "lga": (40.7769, -73.8740),
        "ewr": (40.6895, -74.1745),
    },
    # Constraints for data cleaning
    "bounds": {
        "fare_min": 2.5,  # Minimum base fare in NYC is usually 2.50
        "fare_max": 500.0,  # Remove extreme outliers
        "passenger_min": 1,  # Rides must have passengers
        "passenger_max": 10,  # Reasonable upper bound
        "lat_min": -90.0,
        "lat_max": 90.0,
        "lon_min": -180.0,
        "lon_max": 180.0,
        # Bounding box for NYC (approximate) to filter bad coordinates if needed
        # "nyc_lat_min": 40.5, "nyc_lat_max": 41.0,
        # "nyc_lon_min": -74.3, "nyc_lon_max": -73.7
    },
}

# --- Model Hyperparameters (LightGBM) ---
MODEL_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,  # Lower learning rate for better convergence (Cite solution_lesson_node_00002)
    "num_leaves": 256,  # Increased capacity for full dataset (Cite solution_lesson_node_00002)
    "max_depth": -1,
    "min_child_samples": 100,
    "subsample": 0.8,  # Bagging fraction
    "subsample_freq": 1,
    "colsample_bytree": 0.8,  # Feature fraction
    "reg_alpha": 0.1,  # L1 regularization
    "reg_lambda": 0.1,  # L2 regularization
    "n_jobs": 12,  # Utilize all 12 vCPUs
    "verbosity": -1,
    "seed": 42,
    # Note: 'device': 'gpu' is omitted to ensure stability on standard builds,
    # but n_jobs=12 ensures high performance on CPU.
}

# --- Training Configuration ---
TRAIN_CONFIG = {
    "num_boost_round": 20000,  # Increased for lower learning rate
    "early_stopping_rounds": 50,  # Stop if validation metric doesn't improve
    "verbose_eval": 50,  # Print metrics every 50 rounds
    "random_state": 42,
    "debug_sample_size": None,  # Set to int (e.g., 100_000) for quick debugging
}
