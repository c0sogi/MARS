import os
import json
import hashlib
import torch


class Config:
    # =========================================================================
    # PATHS & DIRECTORIES
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (features, spectrograms)
    WORKING_DIR = "./working/idea_5"

    # Submission directory and path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # GLOBAL SETTINGS
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set to True to run on a small subset of data for debugging
    DEBUG = False

    # =========================================================================
    # DATA SPECIFICS
    # =========================================================================
    SENSORS = [f"sensor_{i}" for i in range(1, 11)]
    SAMPLING_RATE = 100  # 60001 rows over 10 minutes implies ~100Hz
    SIGNAL_LENGTH = 60001

    # =========================================================================
    # BRANCH A: EXPERT FEATURE REGRESSOR (LightGBM)
    # =========================================================================
    # Configuration for tabular feature extraction
    TABULAR_CONFIG = {
        "quantiles": [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
        "abs_quantiles": [0.05, 0.25, 0.50, 0.75, 0.95],
        "sampling_rate": SAMPLING_RATE,
        "sensors": SENSORS,
        "feature_version": "v2_corr_domfreq",
    }

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
        "verbose": -1,
        "n_jobs": -1,
        "random_state": SEED,
        "verbosity": -1,
    }
    LGBM_EARLY_STOPPING_ROUNDS = 100

    # =========================================================================
    # BRANCH B: TIME-FREQUENCY VISION MODEL (2D-CNN)
    # =========================================================================
    # Spectrogram Generation Parameters
    SPECTROGRAM_PARAMS = {
        "n_fft": 1024,
        "hop_length": 256,
        "n_mels": 128,
        "fmin": 0,
        "fmax": 50,  # Nyquist frequency is 50Hz
        "power": 2.0,
        "sampling_rate": SAMPLING_RATE,
        "top_db": 80.0,
        "cache_version": "v2_sanitized",
    }

    # CNN Training Hyperparameters
    CNN_PARAMS = {
        "model_name": "tf_efficientnet_b0",  # timm model name
        "in_channels": 10,  # One channel per sensor
        "batch_size": 32,
        "epochs": 20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "scheduler_T_max": 20,
        "scheduler_eta_min": 1e-6,
    }

    # =========================================================================
    # META-LEARNER (STACKING)
    # =========================================================================
    META_RIDGE_ALPHA = 1.0

    # =========================================================================
    # UTILITIES
    # =========================================================================
    @classmethod
    def get_tabular_hash(cls):
        """
        Generates a unique hash based on the tabular feature configuration.
        Used for caching feature matrices.
        """
        # Include sensors and config in hash to invalidate cache if they change
        config_dict = {"sensors": cls.SENSORS, "tabular_config": cls.TABULAR_CONFIG}
        return cls._compute_hash(config_dict)

    @classmethod
    def get_spectrogram_hash(cls):
        """
        Generates a unique hash based on the spectrogram configuration.
        Used for caching spectrogram tensors.
        """
        config_dict = {
            "sensors": cls.SENSORS,
            "spectrogram_params": cls.SPECTROGRAM_PARAMS,
        }
        return cls._compute_hash(config_dict)

    @staticmethod
    def _compute_hash(config_dict):
        """Helper to compute MD5 hash of a dictionary."""
        # Sort keys to ensure consistent ordering for hashing
        s = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    @classmethod
    def get_cache_path(cls, prefix, file_ext):
        """
        Constructs a cache file path using the appropriate hash.

        Args:
            prefix (str): 'train', 'val', or 'test'
            file_ext (str): 'parquet' or 'npy'
        """
        if file_ext == "parquet":
            # Tabular features
            h = cls.get_tabular_hash()
            name = f"{prefix}_features_{h}.parquet"
        elif file_ext == "npy":
            # Spectrograms or raw data
            h = cls.get_spectrogram_hash()
            name = f"{prefix}_spectrograms_{h}.npy"
        else:
            name = f"{prefix}_data.cache"

        return os.path.join(cls.WORKING_DIR, name)
