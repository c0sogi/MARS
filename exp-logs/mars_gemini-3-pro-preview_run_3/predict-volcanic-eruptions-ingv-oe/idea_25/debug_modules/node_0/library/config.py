import os


class Config:
    """
    Configuration class for the High-Resolution Spectral-Kinematic Ensemble
    with Hierarchical Volatility Profiling strategy.
    """

    # ==========================================
    # File Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate features (Idea 25)
    WORKING_DIR = "./working/idea_25"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42

    # Debugging / Development flags
    # Set DEBUG to True to use a smaller subset of data for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200  # Number of files to process in debug mode

    # ==========================================
    # Signal Processing Constants
    # ==========================================
    SAMPLING_RATE = 100  # Hz

    # Savitzky-Golay Filter (Trend Extraction)
    # Large window size to strictly isolate low-frequency baseline drift
    SG_WINDOW_SIZE = 51
    SG_POLY_ORDER = 3

    # Welch's Method (PSD Calculation)
    # High nperseg to ensure sufficient frequency resolution in low bands
    WELCH_NPERSEG = 1024

    # Frequency Bands for Spectral Power features (Hz)
    FREQ_BANDS = {"low": (0.1, 3), "mid": (3, 10), "high": (10, 45)}

    # Wavelet Transform (Texture Analysis)
    WAVELET_NAME = "db4"

    # Hierarchical Volatility Profiling
    # Divide the 60001-sample signal into N non-overlapping windows
    # 60 windows corresponds to ~10 seconds (1000 samples) per window
    VOLATILITY_NUM_WINDOWS = 60

    # ==========================================
    # Model Hyperparameters (LightGBM)
    # ==========================================
    # 5-Fold Stratified Cross-Validation
    N_FOLDS = 5

    # High-Capacity LightGBM Regressor parameters
    LGBM_PARAMS = {
        "num_leaves": 128,  # High capacity for complex interactions
        "learning_rate": 0.02,  # Low learning rate for better convergence
        "objective": "regression_l2",  # MSE Loss (L2) for stable gradients
        "metric": "mae",  # MAE for evaluation
        "n_estimators": 10000,  # Large number of trees (relies on early stopping)
        "boosting_type": "gbdt",
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    # Training Loop Controls
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 100

    @staticmethod
    def setup():
        """
        Ensures that the necessary working and submission directories exist.
        Should be called at the start of execution.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
