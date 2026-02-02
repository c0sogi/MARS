import os


class Config:
    """
    Configuration class for the High-Capacity Smoothed Semantic Regressor (HC-SSR).
    Defines global constants, file paths, and hyperparameters.
    """

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # -------------------------------------------------------------------------
    # Input Data Paths (Metadata)
    # -------------------------------------------------------------------------
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # -------------------------------------------------------------------------
    # Output/Cache Paths
    # -------------------------------------------------------------------------
    # Parquet files for caching extracted features
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model artifacts
    FINE_TUNED_MODEL_PATH = os.path.join(WORKING_DIR, "fine_tuned_mpnet")
    LGBM_MODEL_PATH = os.path.join(WORKING_DIR, "lgbm_model.txt")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to use a smaller subset of data for debugging
    DEBUG_SAMPLE_SIZE = 1000  # Number of notebooks to use when DEBUG is True

    # -------------------------------------------------------------------------
    # Semantic Backbone (MPNet) Hyperparameters
    # -------------------------------------------------------------------------
    MODEL_CHECKPOINT = "sentence-transformers/all-mpnet-base-v2"
    MAX_LENGTH = 128
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 64
    EPOCHS = 1
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01

    # -------------------------------------------------------------------------
    # Feature Engineering Hyperparameters
    # -------------------------------------------------------------------------
    # 1D Smoothing Convolution Kernel for similarity vectors
    # Used to identify the "dense center" of semantic matches
    SMOOTHING_KERNEL = [0.25, 0.5, 0.25]

    # -------------------------------------------------------------------------
    # LightGBM Regressor Hyperparameters
    # -------------------------------------------------------------------------
    LGBM_PARAMS = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    LGBM_EARLY_STOPPING_ROUNDS = 50

    @classmethod
    def setup(cls):
        """
        Ensures that the necessary working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
