import os


class Config:
    """
    Project-wide configuration and constants for the Taxi Fare Prediction task.
    Implements settings for Dual-Hygiene data processing and Hierarchical Geohashing.
    """

    # -------------------------------------------------------------------------
    # File System & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate artifacts (Parquet/Numpy)
    # Specific to the current experimental idea
    WORKING_DIR = "./working/idea_22"

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Geographic Constraints (Sanitization)
    # -------------------------------------------------------------------------
    # Bounding Box for NYC and surrounding airports (JFK, LGA, EWR).
    # Used to clamp coordinates and prevent model extrapolation on garbage GPS data.
    # Lat Range: ~South Staten Island to North Bronx/Westchester border
    # Lon Range: ~West of Newark to East of Queens/Nassau border
    BB_MIN_LAT = 39.60
    BB_MAX_LAT = 41.30
    BB_MIN_LON = -74.50
    BB_MAX_LON = -72.80

    # Earth Radius for Haversine distance calculations
    EARTH_RADIUS_KM = 6371.0

    # -------------------------------------------------------------------------
    # Dual-Hygiene Filtering Strategy
    # -------------------------------------------------------------------------
    # 1. Wisdom Set (Background Knowledge):
    #    Used to generate statistical fingerprints (Mean/Std/Count).
    #    STRICT filters ensure these priors are robust and free of noise.
    WISDOM_MIN_FARE = 2.50
    WISDOM_MAX_FARE = 200.00
    WISDOM_MAX_FARE_PER_KM = (
        10.0  # Cap price/km to remove traffic jams/errors from priors
    )

    # 2. Learner Set (Foreground Training):
    #    Used to train the XGBoost model.
    #    LOOSE filters allow the model to learn valid heavy-tail events (surges).
    LEARNER_MIN_FARE = 2.50
    # Note: No LEARNER_MAX_FARE is set, to preserve high-fare outliers for RMSE optimization.

    # -------------------------------------------------------------------------
    # Feature Engineering: Hierarchical Geohashing
    # -------------------------------------------------------------------------
    # Geohash precision levels to generate fingerprints for.
    # L5: ~4.9km x 4.9km (Macro/Regional)
    # L6: ~1.2km x 0.6km (Meso/Neighborhood)
    # L7: ~153m x 153m   (Micro/Street Corner)
    GEOHASH_PRECISIONS = [5, 6, 7]

    # -------------------------------------------------------------------------
    # Dataset Sizing
    # -------------------------------------------------------------------------
    # The Wisdom set uses the full available training data (filtered).
    # The Learner set is a stable subsample to manage training time while maintaining convergence.
    LEARNER_SAMPLE_SIZE = 5_000_000

    # -------------------------------------------------------------------------
    # Model Hyperparameters (XGBoost)
    # -------------------------------------------------------------------------
    # Optimized for NVIDIA A100-SXM4-40GB
    XGB_PARAMS = {
        "n_estimators": 5000,  # High cap, controlled by early stopping
        "learning_rate": 0.02,  # Low LR for robust convergence on noisy data
        "max_depth": 8,  # Deep enough to capture spatial interactions
        "subsample": 0.85,  # Row sampling to prevent overfitting
        "colsample_bytree": 0.85,  # Feature sampling
        "objective": "reg:squarederror",  # L2 Loss aligns with RMSE metric
        "n_jobs": 12,  # CPU threads for data loading/pre-processing
        "tree_method": "hist",  # Histogram-based method (required for GPU)
        "device": "cuda",  # Enable GPU acceleration
        "random_state": SEED,
        "eval_metric": "rmse",
    }

    # Training Loop Configuration
    NUM_FOLDS = 5
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 100

    @staticmethod
    def setup():
        """
        Initialize the environment by creating necessary directories.
        Should be called at the start of the pipeline.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
