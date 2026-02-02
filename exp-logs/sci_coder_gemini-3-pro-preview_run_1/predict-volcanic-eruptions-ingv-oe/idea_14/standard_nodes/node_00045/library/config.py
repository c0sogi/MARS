import os
import torch


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    PROJECT_NAME = "volcano_eruption_prediction"
    IDEA_NAME = "idea_14"
    SEED = 42
    N_FOLDS = 5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200  # Number of samples to use if DEBUG is True

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Create working directory immediately
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission
    SUBMISSION_PATH = "./submission/submission.csv"

    # Cache Paths (for deterministic processing)
    # We use Parquet for tabular features and NPY for spectrograms/scalars
    TABULAR_TRAIN_CACHE = os.path.join(WORKING_DIR, "train_features.parquet")
    TABULAR_VAL_CACHE = os.path.join(WORKING_DIR, "val_features.parquet")
    TABULAR_TEST_CACHE = os.path.join(WORKING_DIR, "test_features.parquet")

    SPECTROGRAM_TRAIN_DIR = os.path.join(WORKING_DIR, "spectrograms_train")
    SPECTROGRAM_VAL_DIR = os.path.join(WORKING_DIR, "spectrograms_val")
    SPECTROGRAM_TEST_DIR = os.path.join(WORKING_DIR, "spectrograms_test")

    # Path to store/load the computed global max constant if dynamic calculation is desired
    GLOBAL_MAX_PATH = os.path.join(WORKING_DIR, "global_max_spectrogram.npy")

    # ==========================================
    # Data Properties
    # ==========================================
    NUM_SENSORS = 10
    SIGNAL_LENGTH = 60001
    SAMPLING_RATE = 100  # Hz (Approximation: 60k samples / 600 seconds)

    # ==========================================
    # Branch A: Tabular (LightGBM)
    # ==========================================
    # Feature Engineering
    MFCC_N = 13
    # Sub-bands for Log-Subband Energy features (Hz)
    # Designed to capture specific seismic signatures (tremors vs shocks)
    SUBBAND_FREQS = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, 45)]

    # LightGBM Hyperparameters
    LGB_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "n_jobs": -1,
        "random_state": SEED,
    }
    LGB_EARLY_STOPPING_ROUNDS = 100

    # ==========================================
    # Branch B: Vision (EfficientNet)
    # ==========================================
    # Spectrogram Generation
    # Dual-Resolution: Stack Short (Time) and Long (Freq) windows
    N_MELS = 64
    IMG_SIZE = (128, 128)  # Input size for the CNN (H, W)

    # Short Window (High Time Resolution)
    N_FFT_SHORT = 256
    HOP_LENGTH_SHORT = 64

    # Long Window (High Frequency Resolution)
    N_FFT_LONG = 1024
    HOP_LENGTH_LONG = 256

    FMIN = 0
    FMAX = 50  # Nyquist limit

    # Global Log-Max Scaling Constant (M_global)
    # Used for: X_norm = log(X + 1) / log(M_global + 1)
    # A fixed constant ensures absolute magnitude is preserved across samples.
    # 1e6 is a safe upper bound for FFT magnitude of int16 seismic data.
    GLOBAL_MAX_MAGNITUDE = 1000000.0

    # Model Architecture
    MODEL_NAME = "efficientnet_b0"
    IN_CHANNELS = 20  # 10 sensors * 2 views (Short + Long)
    NUM_CLASSES = 1

    # Normalized Scalar Fusion
    USE_SCALARS = True
    # Dimensions: 10 sensors * 3 stats (Log-Total-Energy, Global Max, Crest Factor)
    SCALAR_DIM = 30

    # Training Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 35
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 8  # For early stopping

    # Target Scaling
    # Apply log1p to target to handle large dynamic range of time_to_eruption
    LOG_SCALE_TARGET = True

    # ==========================================
    # Meta-Learner
    # ==========================================
    META_MODEL_ALPHA = 10.0  # Ridge Regression Regularization
