import os


class Config:
    """
    Configuration for the Hierarchical Residual Gradient Boosting Taxi Fare Predictor.

    This configuration implements the strategy of using a global hierarchical prior
    (Fine Grid -> Coarse Grid -> Physics) as a base margin for an XGBoost Regressor.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"

    # Input Data Paths (from Metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths for Deterministic Processing
    # Stores the Global Knowledge Base (Aggregated Stats)
    GLOBAL_STATS_CACHE_PATH = os.path.join(WORKING_DIR, "global_stats.parquet")
    # Stores processed datasets with computed base margins and residuals
    PROCESSED_TRAIN_CACHE_PATH = os.path.join(
        WORKING_DIR, "processed_train_subsample.parquet"
    )
    PROCESSED_VAL_CACHE_PATH = os.path.join(WORKING_DIR, "processed_val.parquet")
    PROCESSED_TEST_CACHE_PATH = os.path.join(WORKING_DIR, "processed_test.parquet")

    # ==========================================
    # Global Settings
    # ==========================================
    RANDOM_SEED = 42
    NUM_WORKERS = 12  # Available vCPUs

    # ==========================================
    # Data Processing & Feature Engineering
    # ==========================================

    # 1. Physics-Consistent Filtering (Stage 1)
    # Filter logic: Remove if Fare > Threshold AND Fare/Km > Rate_Threshold
    # This removes "garbage" outliers (e.g., $90k for 100m) while preserving valid long trips.
    OUTLIER_FARE_THRESHOLD = 500.0  # Dollars
    OUTLIER_RATE_THRESHOLD = 50.0  # Dollars per Km

    # 2. Hierarchical Prior Construction
    # Fine Grid: ~100m resolution (3 decimal places)
    # Coarse Grid: ~1km resolution (2 decimal places)
    GRID_RES_FINE = 3
    GRID_RES_COARSE = 2

    # Minimum number of samples required in a grid cell to use its statistical average
    # If count < threshold, fallback to next level (Fine -> Coarse -> Physics)
    PRIOR_COUNT_THRESHOLD = 5

    # 3. Coordinate Clamping (NYC Bounding Box)
    # Strictly clamp coordinates to prevent linear extrapolation risks outside the city
    LAT_MIN, LAT_MAX = 39.60, 42.00
    LON_MIN, LON_MAX = -75.00, -72.00

    # 4. Training Data Subsampling (Stage 2)
    # Use a stable subsample of the dataset for the actual Gradient Boosting
    # to avoid convergence issues and reduce runtime while maintaining density.
    TRAIN_SUBSAMPLE_SIZE = 6_000_000

    # 5. Post-Processing
    MIN_FARE_PREDICTION = 2.50

    # ==========================================
    # Model Hyperparameters (XGBoost)
    # ==========================================
    # Objective: reg:squarederror (Standard L2 Loss)
    # We predict the residual (Target - Base_Margin), so L2 is appropriate.
    XGB_PARAMS = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 8,  # Moderate depth to capture spatial interactions
        "subsample": 0.75,  # Row subsampling to prevent overfitting
        "colsample_bytree": 0.8,  # Feature subsampling
        "min_child_weight": 10,  # Regularization
        "tree_method": "hist",  # Efficient histogram-based training
        "device": "cuda",  # Use NVIDIA A100 GPU
        "n_jobs": NUM_WORKERS,
        "random_state": RANDOM_SEED,
    }

    # Training Loop Configuration
    NUM_BOOST_ROUNDS = 5000
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set DEBUG = True to run on a tiny subset of data for pipeline verification
    DEBUG = False
    DEBUG_SIZE = 100_000

    @classmethod
    def setup(cls):
        """
        Creates necessary working and submission directories.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
