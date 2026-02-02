import os
import numpy as np


class Config:
    """
    Global configuration for the Volcanic Eruption Prediction task.
    Implements the Dual-Resolution Spectral Energy Stacking Strategy.
    """

    # ==========================================
    # 1. PATHS & DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSVs
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Cache (Idea 12)
    # Stores intermediate features (parquet) and processed spectrograms (npy)
    CACHE_DIR = "./working/idea_12"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Submission Output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. GLOBAL DATA CONSTANTS
    # ==========================================
    SEED = 42
    NUM_SENSORS = 10
    SAMPLING_RATE = 100  # Hz (Derived from 60001 samples over 10 mins)
    SIGNAL_LENGTH = 60001

    # Global Max Reading for Log-Max Scaling
    # Used to normalize spectrograms while preserving absolute energy differences.
    # Value ~32767 covers the int16 range of normalized sensor readings.
    GLOBAL_MAX_READING = 32767.0

    # ==========================================
    # 3. FEATURE ENGINEERING (BRANCH A - TABULAR)
    # ==========================================
    # Sub-band Energy: Log-sum-squared energy in specific frequency bands
    # Nyquist is 50Hz. We use 10 linearly spaced bands.
    NUM_SUBBANDS = 10
    SUBBAND_WIDTH = 5  # Hz
    # Edges: [0, 5, 10, ..., 50]
    SUBBAND_EDGES = [i * SUBBAND_WIDTH for i in range(NUM_SUBBANDS + 1)]

    # Robust MFCC Configuration
    # Using low-order coefficients (1-13) to capture timbre/texture
    MFCC_N_MFCC = 13
    MFCC_N_FFT = 1024
    MFCC_HOP_LENGTH = 512

    # ==========================================
    # 4. SPECTROGRAMS (BRANCH B - VISION)
    # ==========================================
    # Dual-Resolution Strategy:
    # 1. Wide-Band: Short window for high time resolution (Impulsive Shocks)
    # 2. Narrow-Band: Long window for high freq resolution (Harmonic Tremors)

    # Wide-Band Settings (Window ~0.64s)
    N_FFT_WIDE = 64
    HOP_WIDE = 16

    # Narrow-Band Settings (Window ~2.56s)
    N_FFT_NARROW = 256
    HOP_NARROW = 64

    # CNN Input Dimensions
    # Spectrograms will be resized to this resolution
    IMG_SIZE = (256, 256)

    # Input Channels: 10 Sensors * 2 Views (Wide/Narrow) = 20 Channels
    IN_CHANNELS = 20

    # ==========================================
    # 5. MODEL HYPERPARAMETERS
    # ==========================================
    # Branch A: LightGBM Regressor
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    # Branch B: EfficientNet CNN
    CNN_BACKBONE = "efficientnet_b0"
    CNN_PRETRAINED = True

    # Meta-Learner: Ridge Regression
    RIDGE_ALPHA = 1.0

    # ==========================================
    # 6. TRAINING SETTINGS
    # ==========================================
    NUM_FOLDS = 5

    # Compute
    BATCH_SIZE = 32  # Optimized for A100 40GB
    NUM_WORKERS = 4

    # CNN Optimization
    CNN_EPOCHS = 35
    CNN_LR = 1e-3
    CNN_WEIGHT_DECAY = 1e-4
    CNN_PATIENCE = 7  # Early stopping patience

    # Debugging Flag
    # If True, runs on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100
