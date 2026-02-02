import os


class Config:
    """
    Configuration for Idea 15: Factorized Spatiotemporal Dual-Hygiene Gradient Boosting.
    Defines global constants, file paths, filtering thresholds, and model hyperparameters.
    """

    # -------------------------------------------------------------------------
    # Directory Structure
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # -------------------------------------------------------------------------
    # Input Data Paths (using pre-generated metadata)
    # -------------------------------------------------------------------------
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # Output Paths
    # -------------------------------------------------------------------------
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache files for reproducible pipeline execution
    # Stores the calculated global statistics (Wisdom Set)
    CACHE_WISDOM_STATS = os.path.join(WORKING_DIR, "wisdom_stats.parquet")
    # Stores the feature-engineered datasets
    CACHE_PROCESSED_TRAIN = os.path.join(WORKING_DIR, "processed_train.parquet")
    CACHE_PROCESSED_VAL = os.path.join(WORKING_DIR, "processed_val.parquet")
    CACHE_PROCESSED_TEST = os.path.join(WORKING_DIR, "processed_test.parquet")
    # Stores the trained model artifact
    MODEL_OUTPUT_PATH = os.path.join(WORKING_DIR, "xgb_model.json")

    # -------------------------------------------------------------------------
    # Global Constants & Geometry
    # -------------------------------------------------------------------------
    SEED = 42

    # NYC Bounding Box (Used for clamping and initial cleaning)
    # Covers NYC and major airports (JFK, LGA, EWR)
    NYC_LAT_MIN = 40.5
    NYC_LAT_MAX = 41.8
    NYC_LON_MIN = -74.5
    NYC_LON_MAX = -72.8

    # Grid / Geohash Simulation (Bin Sizes in Degrees)
    # Approximations: 1 deg lat ~ 111km, 1 deg lon ~ 85km at NYC latitude
    GRID_SIZES = {
        "L5": 0.045,  # Approx 5km (Neighborhood/District)
        "L6": 0.009,  # Approx 1km (Major Blocks)
        "L7": 0.00135,  # Approx 150m (Street Segment)
    }

    # -------------------------------------------------------------------------
    # Dual-Hygiene Filtering Strategy
    # -------------------------------------------------------------------------

    # 1. Wisdom Set Filter (Strict):
    # Used to generate global statistics (Mean Fare, Fare/Km).
    # Removes all potential noise to ensure priors are high-quality.
    STRICT_FILTER = {
        "min_fare": 2.5,
        "max_fare": 200.0,
        "max_fare_per_km": 10.0,  # Exclude traffic jams/errors causing infinite rates
        "min_dist_km": 0.2,  # Remove very short trips (noise)
        "min_passenger": 1,
    }

    # 2. Learner Set Filter (Loose):
    # Used for training the model.
    # Retains heavy tails and outliers to minimize RMSE on difficult samples.
    LOOSE_FILTER = {
        "min_fare": 2.5,
        "max_fare": 1000.0,  # Allow high value trips
        "max_fare_per_km": 500.0,
        "min_dist_km": 0.0,  # Allow short trips if fare is valid
        "min_passenger": 0,
    }

    # -------------------------------------------------------------------------
    # Dataset Sizing
    # -------------------------------------------------------------------------
    # Target size for the training set (subsampled from 55M rows)
    LEARNER_SAMPLE_SIZE = 5_000_000

    # Debugging configuration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100_000

    # -------------------------------------------------------------------------
    # Model Hyperparameters (XGBoost)
    # -------------------------------------------------------------------------
    # Optimized for A100 GPU (device='cuda')
    XGB_PARAMS = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 9,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 10,
        "n_estimators": 10000,  # Controlled by early stopping
        "early_stopping_rounds": 50,
        "n_jobs": 12,
        "device": "cuda",  # Enable GPU acceleration
        "tree_method": "hist",  # Efficient histogram-based training
        "random_state": SEED,
    }

    @classmethod
    def setup(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
