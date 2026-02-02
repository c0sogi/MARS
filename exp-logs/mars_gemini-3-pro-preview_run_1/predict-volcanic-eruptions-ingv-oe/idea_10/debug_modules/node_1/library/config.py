import os
import torch


class Config:
    """
    Global configuration for the Robust Audio-Seismic & Contrast-Normalized Stacking Strategy.
    Acts as a central dependency for parameter consistency across feature engineering,
    training, and inference modules.
    """

    # ==========================================
    # PATHS & DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # GLOBAL SETTINGS
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    DEBUG = False  # Toggle for debugging with smaller subsets
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # DATA SPECIFICATIONS
    # ==========================================
    NUM_SENSORS = 10
    # Data is 10 minutes at 100Hz -> ~60,000 samples
    SEGMENT_LENGTH = 60001
    SAMPLING_RATE = 100

    # ==========================================
    # FEATURE ENGINEERING (BRANCH A: TABULAR)
    # ==========================================
    # Mel-Frequency Cepstral Coefficients
    # Limit to first 13 coefficients to capture timbre/envelope, rejecting high-freq noise
    N_MFCC = 13
    N_FFT = 2048
    HOP_LENGTH = 512

    # Statistical Aggregation
    # Compute only robust statistics to ignore outliers
    ROBUST_QUANTILES = [0.05, 0.95]
    # Explicitly exclude Min/Max statistics (noise artifacts)
    USE_MIN_MAX = False

    # ==========================================
    # PREPROCESSING (BRANCH B: VISION)
    # ==========================================
    # Spectrogram Parameters
    N_MELS = 128
    FMIN = 0
    FMAX = 50  # Nyquist frequency for 100Hz sampling

    # Image Normalization & Target Scaling
    # Normalize each spectrogram instance independently (X - mean) / std
    INSTANCE_STANDARDIZATION = True
    # Apply log1p scaling to the target variable to aid convergence
    TARGET_LOG_SCALE = True

    # ==========================================
    # MODEL HYPERPARAMETERS
    # ==========================================

    # Branch A: LightGBM (Robust Audio-Seismic Regressor)
    LGB_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_estimators": 5000,
        "early_stopping_rounds": 100,
        "n_jobs": -1,
        "random_state": SEED,
    }

    # Branch B: CNN (Contrast-Normalized Vision Model)
    CNN_PARAMS = {
        "model_name": "efficientnet_b0",
        "in_channels": 10,
        "num_classes": 1,
        "batch_size": 32,
        "epochs": 25,  # Extended duration to ensure convergence
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "eta_min": 1e-6,
        "t_max": 25,  # For Cosine Annealing scheduler
    }

    # Meta-Learner (Ridge Regression)
    META_PARAMS = {"alpha": 1.0, "random_state": SEED}  # Regularization strength
