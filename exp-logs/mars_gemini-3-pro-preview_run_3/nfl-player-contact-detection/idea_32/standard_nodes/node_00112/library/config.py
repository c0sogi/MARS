import os


class Config:
    """
    Global configuration for the Scale-Aligned Dual-Stream GBDT pipeline.
    Centralizes file paths, hyperparameters, and feature engineering constants.
    """

    # =========================================================================
    # PATHS & DIRECTORIES
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Cache Directory (Idea 32 specific)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_32")

    # Input Files
    TRAIN_LABELS_PATH = os.path.join(INPUT_DIR, "train_labels.csv")

    TRACKING_PATH_TRAIN = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TRACKING_PATH_TEST = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    HELMETS_PATH_TRAIN = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    HELMETS_PATH_TEST = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

    VIDEO_META_TRAIN = os.path.join(INPUT_DIR, "train_video_metadata.csv")
    VIDEO_META_TEST = os.path.join(INPUT_DIR, "test_video_metadata.csv")

    # Generated Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # GLOBAL SETTINGS
    # =========================================================================
    SEED = 42
    N_JOBS = 12  # 12 vCPUs available
    USE_GPU = True

    # =========================================================================
    # FEATURE ENGINEERING CONFIGURATION
    # =========================================================================
    # Global window size for temporal flattening (t +/- WINDOW_SIZE)
    WINDOW_SIZE = 15

    # Sampling frequency of the data (10Hz) implies 1 step = 0.1s
    # Window of 15 steps = +/- 1.5 seconds context

    # Sparse Lags for Visual Consensus (Stream A)
    # Captures immediate contact, short-term approach, and long-term context
    VISUAL_CONSENSUS_LAGS = [0, 4, 8, 15]

    # =========================================================================
    # TRAINING CONFIGURATION
    # =========================================================================
    # Targeted Majority Undersampling Ratio (Negative : Positive)
    # 10:1 ratio as specified in the idea description
    UNDERSAMPLE_RATIO = 10.0

    # Stream A: The Interaction Model (Player-Player)
    # Backbone: Conditional Early-Fusion GBDT
    # Logic: Complex interaction logic requires standard depth
    XGB_PARAMS_STREAM_A = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",  # Use A100 GPU
        "max_depth": 6,  # Standard Depth
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 5000,
        "early_stopping_rounds": 50,
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "verbosity": 0,
    }

    # Stream B: The Impact Model (Player-Ground)
    # Backbone: Uni-Modal Hybrid GBDT with Rotational Energy
    # Logic: Noisy sensor derivatives require conservative learning to prevent overfitting
    XGB_PARAMS_STREAM_B = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",  # Use A100 GPU
        "max_depth": 7,  # Explicitly requested depth
        "learning_rate": 0.02,  # Conservative Learning Rate
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 5000,
        "early_stopping_rounds": 50,
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "verbosity": 0,
    }

    @classmethod
    def setup(cls):
        """
        Initialize the working environment.
        Ensures that the cache and submission directories exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Configured Cache Directory: {cls.CACHE_DIR}")
