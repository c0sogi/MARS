import os


class Config:
    """
    Configuration for NYC Taxi Fare Prediction.
    Implements the settings for 'Idea: Log-Space Gradient Boosting with Spatial Clustering'.
    """

    # ==========================================
    # File Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and models (Idea 4)
    WORKING_DIR = "./working/idea_4"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Data Paths (using Metadata Parquet files)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # ==========================================
    # Data Processing & Sanitization
    # ==========================================
    # NYC Bounding Box for Coordinate Clamping
    # Used to filter/clamp outliers that cause linear extrapolation errors
    BB_LAT_MIN = 40.0
    BB_LAT_MAX = 42.0
    BB_LON_MIN = -75.0
    BB_LON_MAX = -72.0

    # Target Transformation
    # Apply log(1 + y) to fare_amount to handle heavy tails and stabilize gradients
    USE_LOG_TARGET = True

    # ==========================================
    # Feature Engineering
    # ==========================================
    # Spatial Clustering
    N_CLUSTERS = 100  # Number of clusters for MiniBatchKMeans

    # ==========================================
    # Model Hyperparameters (XGBoost)
    # ==========================================
    XGB_PARAMS = {
        "objective": "reg:squarederror",
        "tree_method": "gpu_hist",  # Use A100 GPU acceleration
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 10,  # Deeper trees to capture spatial complexities
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 12,  # CPU threads
        "random_state": 42,
        "gpu_id": 0,
    }

    # Training Loop Parameters
    NUM_BOOST_ROUND = 5000
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    # ==========================================
    # System Settings
    # ==========================================
    RANDOM_SEED = 42

    @classmethod
    def setup(cls):
        """
        Initialize the environment by creating necessary directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
