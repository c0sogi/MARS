import os


class Config:
    """
    Configuration for the Multi-Moment Hierarchical Dual-Hygiene Gradient Boosting strategy.
    """

    # ==========================================
    # Global Reproducibility
    # ==========================================
    RANDOM_SEED = 42

    # ==========================================
    # Directory & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching intermediate files (Parquet/Numpy)
    WORKING_DIR = "./working/idea_18"
    # Directory for final submission
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Paths (Generated Metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths for Deterministic Processing
    # These store the engineered features to avoid re-computation
    CACHE_WISDOM_STATS = os.path.join(WORKING_DIR, "wisdom_stats.parquet")
    CACHE_PROCESSED_TRAIN = os.path.join(WORKING_DIR, "processed_train.parquet")
    CACHE_PROCESSED_VAL = os.path.join(WORKING_DIR, "processed_val.parquet")
    CACHE_PROCESSED_TEST = os.path.join(WORKING_DIR, "processed_test.parquet")

    # ==========================================
    # Data Hygiene & Preprocessing
    # ==========================================
    # Bounding Box for NYC Clamping
    # Used to sanitize GPS coordinates and prevent linear extrapolation artifacts.
    # Covers major NYC airports (JFK, LGA, EWR) and boroughs.
    BB_MIN_LON = -74.25
    BB_MAX_LON = -73.75
    BB_MIN_LAT = 40.50
    BB_MAX_LAT = 40.90

    # Geohash Hierarchies
    # Precision levels for generating Multi-Moment Priors (Mean, Std, Count)
    # L5 (~5km error), L6 (~1km error), L7 (~150m error)
    GEOHASH_LEVELS = [5, 6, 7]

    # Wisdom Set Filters (Strict)
    # Used exclusively to generate statistical priors.
    WISDOM_MIN_FARE = 2.50
    WISDOM_MAX_FARE = 200.00
    WISDOM_MAX_FARE_PER_KM = 10.00  # Sanity check for short, expensive trips

    # Learner Set Filters (Loose)
    # Used for training the Gradient Boosting model.
    # Allows heavy-tailed outliers (high fares) to persist for RMSE optimization.
    LEARNER_MIN_FARE = 2.50
    # Note: No upper bound on Learner fare to capture true outliers.

    # Subsampling for Learner Set
    # Training on 5M rows is sufficient given the high-quality priors from 55M rows.
    LEARNER_SUBSAMPLE_SIZE = 5_000_000

    # K-Fold Strategy
    # Used for Vectorized Subtraction of priors in the training set to prevent leakage.
    NUM_FOLDS = 5

    # ==========================================
    # Model Hyperparameters (XGBoost)
    # ==========================================
    XGB_PARAMS = {
        "objective": "reg:squarederror",  # Standard L2 loss for RMSE
        "eval_metric": "rmse",
        "tree_method": "hist",  # Efficient histogram-based training
        "device": "cuda",  # Leverage NVIDIA A100 GPU
        "learning_rate": 0.05,  # Lower rate for better generalization
        "max_depth": 9,  # Deeper trees to capture spatial interactions
        "subsample": 0.85,  # Row sampling to prevent overfitting
        "colsample_bytree": 0.85,  # Feature sampling
        "min_child_weight": 10,  # Control leaf purity
        "gamma": 0.1,  # Minimum loss reduction
        "reg_alpha": 0.1,  # L1 Regularization
        "reg_lambda": 1.0,  # L2 Regularization
        "n_jobs": 12,  # CPU threads for data loading/pre-processing
        "random_state": RANDOM_SEED,
        "n_estimators": 5000,  # Upper limit, controlled by early stopping
    }

    # Training Loop Control
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100
