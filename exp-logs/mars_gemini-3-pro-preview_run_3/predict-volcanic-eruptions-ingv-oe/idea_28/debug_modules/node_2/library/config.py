import os


class Config:
    """
    Configuration for the High-Resolution Hybrid-Transform Bagging Ensemble.
    Stores all file paths, signal processing parameters, and model hyperparameters.
    """

    # ==========================================
    # Project Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_28"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Global Configuration
    # ==========================================
    SEED = 42
    N_CORES = 12
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200  # Number of samples to use when DEBUG is True

    # ==========================================
    # Data Definitions
    # ==========================================
    SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]
    # Sampling rate calculation: 60001 samples / 600 seconds (10 mins) ≈ 100 Hz
    SAMPLING_RATE = 100.0

    # ==========================================
    # Signal Processing Parameters
    # ==========================================

    # 1. Trend Extraction (Savitzky-Golay)
    # Large window to isolate low-frequency baseline drift
    SG_WINDOW = 51
    SG_POLYORDER = 2

    # 2. Texture Extraction (Wavelet Transform)
    # 'db4' captures harmonic complexity and transient structure
    WAVELET_NAME = "db4"

    # 3. Spectral Analysis (Welch's Method)
    # High nperseg to ensure resolution in low frequency bands (0.1-3Hz)
    WELCH_NPERSEG = 1024

    # Frequency Bands (Hz)
    FREQ_BANDS = {"low": (0.1, 3.0), "mid": (3.0, 10.0), "high": (10.0, 45.0)}

    # 4. Temporal Profiling
    # Divide signal into non-overlapping windows to capture temporal evolution
    NUM_TEMPORAL_WINDOWS = 10

    # ==========================================
    # Model Hyperparameters (LightGBM)
    # ==========================================
    N_FOLDS = 5

    # High-Capacity LightGBM Regressor
    # Optimized for L2 Loss (MSE) with high capacity to capture non-linearities
    MODEL_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 128,  # High capacity
        "learning_rate": 0.03,  # Low learning rate for convergence
        "n_estimators": 10000,  # High number of trees (relies on early stopping)
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "min_child_samples": 20,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": SEED,
    }

    # Training Loop Parameters
    TRAIN_PARAMS = {"early_stopping_rounds": 100, "verbose_eval": 100}
