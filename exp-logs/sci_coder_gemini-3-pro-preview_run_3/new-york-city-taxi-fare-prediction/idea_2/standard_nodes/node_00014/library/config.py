import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration constants and hyperparameters for the Taxi Fare Prediction pipeline.
    This configuration supports a Hybrid Ensemble Strategy combining Gradient Boosting
    and Deep Learning with Spatial Clustering.
    """

    # ==========================================
    # 1. Global Configuration & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100_000  # Number of samples to use in debug mode

    # ==========================================
    # 2. Directory & File Paths
    # ==========================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Input Data (Parquet Metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cached Artifacts (for deterministic processing)
    # These files store processed features and trained models
    TRAIN_PROCESSED_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_PROCESSED_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_PROCESSED_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Model & Scaler Persistence
    KMEANS_MODEL_PATH = os.path.join(WORKING_DIR, "kmeans_model.joblib")
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.joblib")
    GBDT_MODEL_PATH = os.path.join(WORKING_DIR, "gbdt_model.joblib")
    NN_MODEL_PATH = os.path.join(WORKING_DIR, "nn_model.pth")

    # ==========================================
    # 3. Data Schema & Filtering
    # ==========================================
    ID_COL = "key"
    TARGET_COL = "fare_amount"
    DATETIME_COL = "pickup_datetime"

    # Raw Feature Columns
    RAW_NUMERICAL_COLS = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]

    # Filtering Logic
    FARE_MIN = 0.0
    FARE_MAX = 500.0  # Filter out extreme outliers > $500

    # Bounding Box (NYC Area approx) - used for basic sanity checks if needed
    COORD_BOUNDS = {"lon_min": -75, "lon_max": -72, "lat_min": 39, "lat_max": 42}

    # ==========================================
    # 4. Feature Engineering Hyperparameters
    # ==========================================
    # Spatial Clustering (Neighborhood Embeddings)
    N_CLUSTERS = 100  # Number of centroids for MiniBatchKMeans

    # Coordinate Rotation (Manhattan Grid Alignment)
    ROTATION_ANGLE = 29.0  # Degrees

    # Distance Landmarks (Lat, Lon)
    LANDMARKS = {
        "JFK": (40.6413, -73.7781),
        "LGA": (40.7769, -73.8740),
        "EWR": (40.6895, -74.1745),
        "TS": (40.7580, -73.9855),  # Times Square
        "WTC": (40.7126, -74.0099),  # World Trade Center
        "MET": (40.7794, -73.9632),  # Met Museum
    }

    # ==========================================
    # 5. Model Hyperparameters
    # ==========================================

    # Stream A: Gradient Boosting (HistGradientBoostingRegressor)
    # Optimized for large tabular datasets
    GBDT_PARAMS = {
        "loss": "squared_error",
        "learning_rate": 0.1,
        "max_iter": 300,  # Number of trees (High capacity)
        "max_leaf_nodes": 255,  # Max leaves per tree
        "max_depth": 15,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
        "random_state": SEED,
        "verbose": 0,
    }

    # Stream B: Neural Network (MLP)
    # Optimized for continuous spatial manifolds
    NN_PARAMS = {
        "batch_size": 4096,  # Large batch size for GPU efficiency
        "learning_rate": 1e-3,
        "epochs": 15,
        "weight_decay": 1e-5,
        "embedding_dims": {
            "cluster": 16,  # Embedding dim for cluster IDs
            "hour": 4,
            "dow": 4,
            "year": 4,
        },
        "hidden_dims": [512, 256, 128, 64],
        "dropout": 0.1,
        "patience": 3,  # Early stopping patience
        "num_workers": 4,
        "pin_memory": True,
    }

    # Ensemble Weights (Weighted Average)
    WEIGHT_GBDT = 0.6
    WEIGHT_NN = 0.4

    # ==========================================
    # 6. Compute Resources
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Available vCPUs

    @classmethod
    def setup(cls):
        """
        Initialize the working environment:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
