import os


class Config:
    """
    Configuration for Hierarchical Residual Dual-Hygiene Gradient Boosting.
    Defines global constants, file paths, and hyperparameters.
    """

    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for this experiment (Idea 16)
    WORKING_DIR = "./working/idea_16"

    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Paths (using generated metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Submission Paths
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Global Reproducibility & Compute
    # --------------------------------------------------------------------------
    SEED = 42
    N_JOBS = 12

    # --------------------------------------------------------------------------
    # Geographic Constants (NYC Bounding Box)
    # --------------------------------------------------------------------------
    # Used for sanitizing input coordinates
    MIN_LON = -74.50
    MAX_LON = -72.80
    MIN_LAT = 40.50
    MAX_LAT = 41.80

    # --------------------------------------------------------------------------
    # Dual-Hygiene Strategy Thresholds
    # --------------------------------------------------------------------------

    # 1. Wisdom Set (Strict Filtering)
    # Used to calculate the 'Base Margin' statistics (clean physics priors)
    STRICT_MIN_FARE = 2.5
    STRICT_MAX_FARE = 200.0
    STRICT_MAX_FARE_PER_KM = 10.0  # Exclude unrealistic price/distance ratios
    STRICT_MIN_DIST_KM = 0.05  # Exclude static noise

    # 2. Learner Set (Loose Filtering)
    # Used to train the XGBoost model (includes heavy tails)
    LOOSE_MIN_FARE = 2.5
    LOOSE_MAX_FARE = 1000.0  # Allow valid high-fare trips (e.g., long distance)

    # Training Subsample Size (for efficiency within 24h limit)
    TRAIN_SUBSAMPLE_SIZE = 5_000_000

    # --------------------------------------------------------------------------
    # Hierarchical Margin & Feature Engineering
    # --------------------------------------------------------------------------
    # Geohash Precision Levels
    GEOHASH_PRECISION_MICRO = 7  # Approx 150m x 150m
    GEOHASH_PRECISION_MESO = 6  # Approx 1.2km x 0.6km

    # Waterfall Logic
    # If count(Micro) > Threshold -> Use Micro Mean
    # Else If count(Meso) > Threshold -> Use Meso Mean
    # Else -> Use Physics Baseline (Global Rate * Distance)
    MARGIN_COUNT_THRESHOLD = 5

    # --------------------------------------------------------------------------
    # XGBoost Hyperparameters
    # --------------------------------------------------------------------------
    XGB_PARAMS = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "device": "cuda",  # Leverage A100 GPU
        "tree_method": "hist",  # Efficient histogram algorithm
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,  # L1 Regularization
        "reg_lambda": 1.0,  # L2 Regularization
        "n_jobs": N_JOBS,
        "random_state": SEED,
    }

    # Training Loop Settings
    NUM_BOOST_ROUND = 5000
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 50
