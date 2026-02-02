import os


class ProjectConfig:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for Idea 20 (Sanitized Target + Scaled Training)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_20")
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================================
    # Global Constants
    # ==========================================
    SEED = 42

    # NYC Bounding Box [Lon Min, Lat Min, Lon Max, Lat Max]
    # Used for strict clamping to prevent linear extrapolation on GPS errors
    NYC_BOUNDING_BOX = [-74.5, 40.5, -72.8, 41.95]

    # Geohash Levels for Hierarchical Priors
    # L5 (~5km): Macro regional trends
    # L6 (~1km): Meso neighborhood trends
    # L7 (~150m): Micro street-level trends
    GEOHASH_LEVELS = [5, 6, 7]

    # ==========================================
    # Dual-Hygiene Strategy Thresholds
    # ==========================================
    # Wisdom Set (Strict): Used exclusively to generate clean statistical priors (Mean, Std, Count)
    # Excludes extreme outliers to prevent polluting the spatial maps.
    WISDOM_MIN_FARE = 2.50
    WISDOM_MAX_FARE = 200.00
    WISDOM_MAX_FARE_PER_KM = (
        10.00  # Heuristic to filter unrealistic short-distance high-fare trips
    )

    # Learner Set (Loose): Used for training the Gradient Boosting model.
    # Includes high-fare valid trips to allow the model to learn heavy-tail predictions.
    LEARNER_MIN_FARE = 2.50
    # Note: No upper bound on Learner fare to capture true outliers (e.g. $500 trips)

    # ==========================================
    # Data Processing Configuration
    # ==========================================
    # Subsample size for the Learner Set to fit in memory and train within time limits
    TRAIN_SUBSAMPLE_SIZE = 5_000_000

    # Number of folds for Vectorized Subtraction (Target Encoding)
    # Ensures statistics for fold K are derived only from folds 1..K-1 to prevent leakage.
    NUM_FOLDS = 5

    # ==========================================
    # Model Hyperparameters (XGBoost)
    # ==========================================
    # Optimized for NVIDIA A100 GPU
    XGB_PARAMS = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",  # Efficient histogram-based algorithm
        "device": "cuda",  # Explicitly utilize GPU
        "max_depth": 9,  # Deep trees to capture complex spatial interactions
        "learning_rate": 0.03,  # Lower learning rate for robust convergence
        "n_estimators": 3000,  # Sufficient trees, controlled by early stopping
        "subsample": 0.85,  # Row subsampling to prevent overfitting
        "colsample_bytree": 0.85,  # Column subsampling
        "reg_alpha": 0.1,  # L1 Regularization
        "reg_lambda": 2.0,  # L2 Regularization
        "n_jobs": 12,  # Number of CPU threads for data loading/pre-processing
        "verbosity": 0,  # Silent mode
    }

    # Training Control
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    # Post-Processing
    PRED_MIN_FARE = 2.50
