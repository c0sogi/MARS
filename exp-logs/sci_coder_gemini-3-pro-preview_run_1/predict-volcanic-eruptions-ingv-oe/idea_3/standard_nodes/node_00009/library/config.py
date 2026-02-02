import os
import torch


class Config:
    """
    Central configuration for the Seismic Eruption Prediction pipeline.
    Includes paths, global constants, and hyperparameters for both
    LightGBM (Branch A) and 1D-ResNet (Branch B).
    """

    # =========================================================================
    # PATHS
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DATA_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching intermediate features (Idea 3)
    WORKING_DIR = "./working/idea_3"

    # Cache file paths
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # GLOBAL SETTINGS
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    NUM_WORKERS = 4  # For DataLoaders
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # DATA SPECIFICS
    # =========================================================================
    SENSORS = [f"sensor_{i}" for i in range(1, 11)]
    NUM_SENSORS = 10
    SEQ_LEN = 60001  # Fixed length of sensor readings

    # =========================================================================
    # BRANCH A: FEATURE ENGINEERING & LIGHTGBM
    # =========================================================================
    # Feature Engineering
    WAVELET_FAMILY = "db4"  # Daubechies 4 wavelet for DWT

    # LightGBM Hyperparameters
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 10000,
        "early_stopping_rounds": 100,
        "verbose": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    # =========================================================================
    # BRANCH B: DEEP LEARNING (1D-RESNET)
    # =========================================================================
    # Training Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Architecture
    RESNET_BASE_FILTERS = 64
    RESNET_KERNEL_SIZE = 7

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def print_summary(cls):
        """
        Prints a summary of the configuration.
        """
        print("=" * 40)
        print("CONFIGURATION SUMMARY")
        print("=" * 40)
        print(f"Device: {cls.DEVICE}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print(f"Submission Path: {cls.SUBMISSION_PATH}")
        print(f"Folds: {cls.N_FOLDS}")
        print(f"Sensors: {cls.NUM_SENSORS}")
        print(f"LGBM Learning Rate: {cls.LGBM_PARAMS['learning_rate']}")
        print(f"DL Batch Size: {cls.BATCH_SIZE}")
        print(f"DL Epochs: {cls.EPOCHS}")
        print("=" * 40)
