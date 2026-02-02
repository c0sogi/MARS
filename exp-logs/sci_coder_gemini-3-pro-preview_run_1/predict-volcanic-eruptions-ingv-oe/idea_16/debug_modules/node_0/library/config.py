import os
import torch


class Config:
    # ==========================================
    # 1. PATHS & DIRECTORIES
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"
    SUBMISSION_DIR = "./submission"

    # Metadata files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache paths for intermediate data
    CACHE_DIR = WORKING_DIR
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    SPECTROGRAM_CACHE_DIR = os.path.join(WORKING_DIR, "spectrograms")

    # ==========================================
    # 2. DATA SPECIFICATIONS
    # ==========================================
    N_SENSORS = 10
    SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]
    SEGMENT_ID_COL = "segment_id"
    TARGET_COL = "time_to_eruption"

    # Signal properties
    # 60001 samples over 10 minutes (600 seconds) -> ~100 Hz
    SAMPLE_RATE = 100
    SIGNAL_LENGTH = 60001

    # Global Max for Normalization (int16 max, confirmed by data analysis)
    GLOBAL_MAX_READING = 32767.0

    # ==========================================
    # 3. PREPROCESSING PARAMETERS
    # ==========================================
    # Branch A: Tabular / Trend Features
    # Frequency bands for energy calculation (Hz)
    FREQ_BANDS = {
        "band_0_2": (0, 2),
        "band_2_5": (2, 5),
        "band_5_10": (5, 10),
        "band_10_20": (10, 20),
        "band_20_50": (20, 50),  # Up to Nyquist
    }

    # Trend Feature Segmentation
    # We divide the signal into 3 overlapping blocks for trend calculation
    N_BLOCKS = 3

    # Branch B: Vision / Spectrograms
    N_FFT = 1024
    HOP_LENGTH = 256
    # EfficientNet-B0 standard input size
    IMG_SIZE = (224, 224)

    # ==========================================
    # 4. MODEL HYPERPARAMETERS
    # ==========================================
    SEED = 42
    N_FOLDS = 5

    # LightGBM (Branch A)
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 2000,
        "early_stopping_rounds": 100,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": SEED,
    }

    # EfficientNet (Branch B)
    BATCH_SIZE = 32
    EPOCHS = 35  # Extended training as per strategy
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Meta-Learner
    META_ALPHA = 1.0  # Ridge regularization strength

    @classmethod
    def make_dirs(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SPECTROGRAM_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
