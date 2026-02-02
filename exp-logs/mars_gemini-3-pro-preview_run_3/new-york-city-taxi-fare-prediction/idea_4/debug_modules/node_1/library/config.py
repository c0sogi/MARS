import os


class Config:
    """
    Configuration class for the Taxi Fare Prediction project (Idea 4).
    Defines project-wide constants, paths, and hyperparameters for the
    Heterogeneous Ensemble with Spatiotemporal Target Encoding.
    """

    # ==========================================
    # FILE PATHS & DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Input Files (Parquet format)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Processed Data Cache Files
    TRAIN_PROCESSED_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_PROCESSED_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_PROCESSED_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Artifacts (Models & Encoders)
    KMEANS_MODEL_PATH = os.path.join(WORKING_DIR, "kmeans_model.joblib")
    TE_MAP_PATH = os.path.join(WORKING_DIR, "te_map.joblib")  # Target Encoding Map
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # DATA SANITATION
    # ==========================================
    # Strict Bounding Box (Lat 40-42, Lon -73 to -75)
    # Filters outliers that cause distance feature explosion
    BB_LAT_MIN = 40.0
    BB_LAT_MAX = 42.0
    BB_LON_MIN = -75.0
    BB_LON_MAX = -73.0

    # Target Variable Range (Train only)
    # Filters extreme fare outliers
    FARE_MIN = 0.0
    FARE_MAX = 500.0

    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================
    # Coordinate Rotation to align with NYC grid
    ROTATION_ANGLE = 29.0

    # Spatiotemporal Target Encoding
    KMEANS_CLUSTERS = 500  # K for MiniBatchKMeans spatial clustering
    TE_FOLDS = 5  # Folds for regularization in target encoding

    # Landmarks for Distance Features (Lat, Lon)
    LANDMARKS = {
        "JFK": (40.6413, -73.7781),
        "LGA": (40.7769, -73.8740),
        "WTC": (40.7126, -74.0099),
    }

    # ==========================================
    # MODEL HYPERPARAMETERS
    # ==========================================
    SEED = 42

    # XGBoost Regressor (GPU Accelerated)
    # Configured for massive throughput on NVIDIA A100
    XGB_PARAMS = {
        "n_estimators": 10000,
        "learning_rate": 0.02,
        "max_depth": 12,  # High capacity for large data
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",  # Efficient GPU histogram algorithm
        "device": "cuda",  # Use GPU
        "n_jobs": -1,
        "random_state": SEED,
        "early_stopping_rounds": 100,
    }

    # LightGBM Regressor (CPU Optimized)
    # Configured for 12 vCPUs
    LGBM_PARAMS = {
        "n_estimators": 10000,
        "learning_rate": 0.02,
        "num_leaves": 512,  # High capacity
        "max_depth": -1,  # Unconstrained, controlled by leaves
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_jobs": 12,  # Use all available vCPUs
        "random_state": SEED,
        "verbose": -1,
        "early_stopping_rounds": 100,
    }

    # Ensemble Weights
    # Weights for the final weighted average prediction
    WEIGHT_XGB = 0.5
    WEIGHT_LGBM = 0.5
