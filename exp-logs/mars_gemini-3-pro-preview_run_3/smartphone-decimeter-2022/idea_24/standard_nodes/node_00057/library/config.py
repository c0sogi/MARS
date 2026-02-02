import os


class Config:
    """
    Global configuration for the GNSS Localization pipeline.
    Includes file paths, physical constants, model hyperparameters, and optimization settings.
    """

    # --- 1. File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_24"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # --- 2. Physical Constants ---
    SPEED_OF_LIGHT = 299792458.0  # m/s
    GPS_L1_FREQ = 1575.42e6  # Hz
    GPS_L5_FREQ = 1176.45e6  # Hz

    # --- 3. Reproducibility ---
    SEED = 42

    # --- 4. Debugging & Data Control ---
    # Set DEBUG to True to limit dataset size for rapid prototyping
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000  # Number of rows to process if DEBUG is True

    # --- 5. Feature Engineering Settings ---
    # Minimum Carrier-to-Noise density (dB-Hz) to consider a signal valid
    CN0_THRESHOLD = 20.0
    # Minimum elevation angle (degrees) to consider a satellite
    ELEVATION_MASK = 5.0

    # --- 6. Model Hyperparameters (LightGBM) ---
    # Using MAE (L1 loss) to be robust against outliers in the geometric residuals
    LGBM_PARAMS = {
        "objective": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 5000,  # High cap, controlled by early stopping
        "learning_rate": 0.05,  # Moderate learning rate for stability
        "num_leaves": 128,  # Allow sufficient complexity for geometric mapping
        "max_depth": -1,  # No strict depth limit
        "min_child_samples": 20,
        "subsample": 0.8,  # Bagging fraction
        "colsample_bytree": 0.8,  # Feature fraction
        "reg_alpha": 0.1,  # L1 regularization
        "reg_lambda": 0.1,  # L2 regularization
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Training control
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 100

    # --- 7. Graph Optimization Parameters ---
    # Weight (Lambda) for the odometry (kinematic) term relative to the anchor (ML) term.
    # Higher values trust the shape of the trajectory (TDCP) more than the absolute ML predictions.
    ODOMETRY_LAMBDA = 5.0

    # Huber Loss Deltas (in meters)
    # Defines the threshold where the loss transitions from quadratic (L2) to linear (L1).
    # Allows the optimizer to be robust against large outliers (cycle slips or ML blunders).
    HUBER_DELTA_ANCHOR = 5.0  # Trust ML anchors within ~5m, linear penalty beyond
    HUBER_DELTA_ODOM = 0.5  # Trust TDCP odometry within ~0.5m, linear penalty beyond

    # --- 8. Caching Utilities ---
    @classmethod
    def get_cache_path(cls, name: str) -> str:
        """
        Generates a standard path for cached parquet files in the working directory.

        Args:
            name: The base name of the file (e.g., 'train_features').

        Returns:
            Absolute path to the cache file.
        """
        return os.path.join(cls.WORKING_DIR, f"{name}.parquet")
