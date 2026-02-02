import os
import numpy as np


class Config:
    """
    Configuration for the Volcano Eruption Prediction Task.
    Implements the settings for the Magnitude-Injected Hybrid Ensemble.
    """

    # ==========================================
    # 1. PATHS & DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific directory for this experimental iteration (Idea 13)
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_13")
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Directories
    TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DATA_DIR = os.path.join(INPUT_DIR, "test")

    # Output Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary write directories exist
    os.makedirs(IDEA_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. DATA SPECIFICATIONS
    # ==========================================
    SEED = 42
    NUM_SENSORS = 10
    SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]

    # Signal Properties
    # 60001 samples over 10 minutes implies ~100 Hz sampling rate
    SAMPLING_RATE = 100
    SIGNAL_LENGTH = 60001
    DURATION = 600  # seconds

    # ==========================================
    # 3. PREPROCESSING HYPERPARAMETERS
    # ==========================================
    # --- Vision Branch (Spectrograms) ---
    N_FFT = 1024
    HOP_LENGTH = 256
    N_MELS = 128
    FMIN = 0
    FMAX = 50  # Nyquist frequency at 100Hz sampling

    # Image size for the CNN (Height=Mels, Width=TimeSteps)
    # Width = ceil(60001 / 256) ≈ 235. We can resize or pad to a fixed size like 256.
    IMG_SIZE = (128, 256)

    # --- Tabular Branch (Feature Engineering) ---
    # Sub-band Energy definitions (Frequency ranges in Hz)
    # Used to capture magnitude distribution across the spectrum
    SUBBANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50)]

    # MFCC Configuration
    # Using coefficients 1-13 to capture timbre while avoiding 0 (energy) which is handled separately
    N_MFCC = 13

    # ==========================================
    # 4. MODEL ARCHITECTURE
    # ==========================================
    # --- Vision Model (EfficientNet) ---
    CNN_MODEL_NAME = "efficientnet_b0"
    IN_CHANNELS = 10  # One channel per sensor

    # Scalar Injection Layer Dimensions
    # 10 sensors * 3 stats (Log-Total-Energy, Global Max, Crest Factor) = 30 features
    SCALAR_INPUT_DIM = 30

    # ==========================================
    # 5. TRAINING HYPERPARAMETERS
    # ==========================================
    N_FOLDS = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 35  # Extended training for convergence
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE = 7  # Early stopping patience

    # Meta-Learner (Ridge Regression)
    META_ALPHA = 1.0

    # Debugging / Development Flags
    # Set DEBUG to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda"  # Assumes NVIDIA A100 is available

    # ==========================================
    # 6. LIGHTGBM PARAMETERS (Branch A)
    # ==========================================
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 5000,
        "early_stopping_rounds": 100,
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": -1,
    }

    # ==========================================
    # 7. CACHE FILE PATHS
    # ==========================================
    # Paths for caching processed features to avoid re-computation
    CACHE_TRAIN_FEATURES = os.path.join(IDEA_DIR, "train_features.parquet")
    CACHE_VAL_FEATURES = os.path.join(IDEA_DIR, "val_features.parquet")
    CACHE_TEST_FEATURES = os.path.join(IDEA_DIR, "test_features.parquet")

    # Path to store the global maximum value observed in training (for normalization)
    CACHE_GLOBAL_MAX = os.path.join(IDEA_DIR, "global_max_spectrogram.npy")

    # Directories for caching individual spectrogram arrays
    CACHE_SPECTROGRAMS_TRAIN = os.path.join(IDEA_DIR, "spectrograms_train")
    CACHE_SPECTROGRAMS_VAL = os.path.join(IDEA_DIR, "spectrograms_val")
    CACHE_SPECTROGRAMS_TEST = os.path.join(IDEA_DIR, "spectrograms_test")

    # Ensure spectrogram cache directories exist
    for d in [
        CACHE_SPECTROGRAMS_TRAIN,
        CACHE_SPECTROGRAMS_VAL,
        CACHE_SPECTROGRAMS_TEST,
    ]:
        os.makedirs(d, exist_ok=True)
