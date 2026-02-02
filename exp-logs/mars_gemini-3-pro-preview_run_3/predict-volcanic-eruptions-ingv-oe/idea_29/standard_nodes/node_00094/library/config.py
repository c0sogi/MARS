import os
import numpy as np


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this idea iteration
    # Using 'idea_30' as specified in the prompt requirements
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_30")

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    SEED = 42
    NUM_SENSORS = 10
    SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]

    # Sampling rate calculation: 60,000 samples / 600 seconds (10 mins) = 100 Hz
    FS = 100

    # ==========================================
    # Signal Processing Configuration (Tri-Stream)
    # ==========================================

    # Stream A: Trend Extraction (Savitzky-Golay)
    # Large window size to isolate low-frequency baseline drift
    SG_WINDOW = 51
    SG_POLYORDER = 3

    # Stream B: Texture Extraction (Wavelet)
    # Discrete Wavelet Transform settings
    DWT_WAVELET = "db4"

    # Stream C: High-Resolution Spectral Structure (PSD)
    # Welch's Method settings
    # nperseg=1024 ensures ~0.1Hz resolution, critical for the 0.1-3Hz band
    PSD_NPERSEG = 1024
    PSD_NOVERLAP = 512  # 50% overlap

    # Frequency Bands (Hz)
    # Closed intervals for band power calculation
    PSD_BANDS = {"low": (0.1, 3.0), "mid": (3.0, 10.0), "high": (10.0, 45.0)}

    # Stream C: Flattened Temporal Profiling
    # Number of non-overlapping windows to divide the signal into
    TEMPORAL_WINDOWS = 10

    # ==========================================
    # Model Configuration (High-Capacity LightGBM)
    # ==========================================
    N_FOLDS = 5

    # LightGBM Hyperparameters
    # High capacity (num_leaves=128) with L2 loss (objective='regression')
    LGBM_PARAMS = {
        "objective": "regression",  # L2 Loss (MSE)
        "metric": "mae",  # Evaluation metric
        "boosting_type": "gbdt",
        "learning_rate": 0.01,  # Low learning rate (< 0.05)
        "n_estimators": 10000,  # High number of trees
        "num_leaves": 128,  # High capacity
        "max_depth": -1,
        "feature_fraction": 0.7,  # Subsample features (Reduced for regularization)
        "bagging_fraction": 0.7,  # Subsample data (Reduced for regularization)
        "bagging_freq": 1,
        "lambda_l1": 10.0,  # Increased Regularization (Cite solution_lesson_node_00064)
        "lambda_l2": 10.0,  # Increased Regularization
        "verbosity": -1,
        "n_jobs": -1,
        "seed": SEED,
    }

    # Training Loop Controls
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 100

    @classmethod
    def setup(cls):
        """Ensure necessary directories exist."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
