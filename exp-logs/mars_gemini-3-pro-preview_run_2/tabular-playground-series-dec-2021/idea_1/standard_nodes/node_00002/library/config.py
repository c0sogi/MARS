import os


class Config:
    """
    Global configuration for the Forest Cover Type prediction pipeline.
    """

    # --- Random Seed ---
    SEED = 42

    # --- File Paths ---
    # Input metadata paths (Parquet format)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Working directory for caching and model artifacts
    WORKING_DIR = "./working/idea_1"
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "lgbm_model.txt")

    # --- Dataset Configuration ---
    ID_COL = "Id"
    TARGET_COL = "Cover_Type"

    # Class mapping based on EDA (Classes 1, 2, 3, 4, 6, 7 present)
    # Mapping to 0-indexed integers for LightGBM
    TARGET_MAPPING = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5}
    # Inverse mapping for creating submission
    INVERSE_TARGET_MAPPING = {v: k for k, v in TARGET_MAPPING.items()}
    NUM_CLASSES = len(TARGET_MAPPING)

    # --- Model Hyperparameters (LightGBM) ---
    # Optimized for speed and accuracy on tabular data with GPU acceleration
    MODEL_PARAMS = {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 128,  # Higher capacity for large dataset
        "max_depth": -1,  # No limit, rely on num_leaves
        "min_data_in_leaf": 100,  # Prevent overfitting
        "feature_fraction": 0.8,  # Subsample features
        "bagging_fraction": 0.8,  # Subsample data
        "bagging_freq": 1,
        "n_jobs": 12,  # Use available vCPUs
        "device": "gpu",  # Leverage NVIDIA A100
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "verbose": -1,  # Suppress warnings
        "seed": SEED,
        "deterministic": True,
    }

    # --- Training Settings ---
    NUM_BOOST_ROUND = 5000  # Max rounds, controlled by early stopping
    EARLY_STOPPING_ROUNDS = 100  # Stop if validation metric doesn't improve
    VERBOSE_EVAL = 100  # Print metrics every 100 rounds

    # Debugging / Development
    DEBUG = False  # Set to True to use a subset of data
    DEBUG_SAMPLE_SIZE = 50000  # Number of rows to use in debug mode
