import os


class Config:
    """
    Configuration class for the Distribution-Aware Semantic Regressor (DASR) pipeline.
    Centralizes file paths, model hyperparameters, and execution settings.
    """

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Paths (Metadata)
    # =========================================================================
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Cache Paths (Intermediate Data)
    # =========================================================================
    # Parquet files for caching processed features and pairs
    TRAIN_PAIRS_PATH = os.path.join(WORKING_DIR, "train_pairs.parquet")
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # =========================================================================
    # Model Artifact Paths
    # =========================================================================
    FINE_TUNED_MODEL_PATH = os.path.join(WORKING_DIR, "fine_tuned_mpnet")
    LGBM_MODEL_PATH = os.path.join(WORKING_DIR, "lgbm_model.txt")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Model Hyperparameters (Transformer)
    # =========================================================================
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    MAX_LEN = 128
    BATCH_SIZE = 32
    NUM_EPOCHS = 1
    LEARNING_RATE = 2e-5

    # =========================================================================
    # Feature Engineering & Data Volume
    # =========================================================================
    # Size of the global structural heatmap vector (resampled similarity distribution)
    HEATMAP_SIZE = 20

    # Number of notebooks to use for the contrastive fine-tuning stage
    # Limited to 50k to fit within runtime constraints while maximizing volume
    FINE_TUNE_SUBSET_SIZE = 50000

    # =========================================================================
    # LightGBM Hyperparameters
    # =========================================================================
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,  # Silent mode
        "n_jobs": -1,
        "seed": 42,
    }

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4
