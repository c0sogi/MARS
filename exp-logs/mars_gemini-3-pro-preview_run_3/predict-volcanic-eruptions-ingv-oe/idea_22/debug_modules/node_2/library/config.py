import os


class Config:
    """
    Central Configuration for the Volcano Eruption Prediction Task.
    Implements the 'Hybrid-Transform Orthogonal Decomposition with Regularized Dense Profiling' strategy.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching features (Idea 22)
    WORKING_DIR = "./working/idea_22"
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Global Configuration
    # ==========================================
    SEED = 42
    N_SENSORS = 10

    # Debugging: Set to True to process a small subset of data for testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200

    # ==========================================
    # Feature Engineering Configuration
    # ==========================================

    # 1. Kinematic Trend (Savitzky-Golay)
    # Strategy requires window size > 50 for stable derivatives
    SG_WINDOW = 51
    SG_POLYORDER = 2  # Order 2 allows for accurate 1st (Velocity) and 2nd (Acceleration) derivatives

    # 2. Multi-Resolution Texture (Wavelets)
    # Using Discrete Wavelet Transform (db4)
    WAVELET_TYPE = "db4"

    # 3. Dense Profiling (Quantiles)
    # Granular distribution capture as requested: 1%, 5%, 25%, 50%, 75%, 95%, 99%
    QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

    # 4. Structural Spectral Features (PSD)
    # Welch's method window size
    WELCH_NPERSEG = 256

    # 5. Temporal Evolution
    # Number of non-overlapping windows to split the signal into for flattening (RMS/Mean)
    TEMPORAL_WINDOWS = 10

    # ==========================================
    # Model Configuration (LightGBM)
    # ==========================================
    # Strategy: Single, Highly-Optimized LightGBM Regressor
    # Key Requirements:
    # - L2 or Huber Loss (to avoid L1 gradient issues)
    # - Low learning rate (< 0.01)
    # - Explicit L1/L2 Regularization

    LGBM_PARAMS = {
        "objective": "regression",  # L2 Loss (Mean Squared Error)
        "metric": "mae",  # Evaluation metric
        "boosting_type": "gbdt",
        "learning_rate": 0.005,  # Strict requirement: < 0.01
        "n_estimators": 10000,  # High number, controlled by early stopping
        "num_leaves": 127,  # Increased capacity for dense feature set
        "max_depth": -1,
        "reg_alpha": 0.1,  # L1 Regularization
        "reg_lambda": 0.1,  # L2 Regularization
        "subsample": 0.7,  # Bagging fraction
        "colsample_bytree": 0.7,  # Feature fraction
        "n_jobs": -1,
        "verbose": -1,
        "random_state": SEED,
    }

    # Training Loop Parameters
    EARLY_STOPPING_ROUNDS = 200
    VERBOSE_EVAL = 100

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
