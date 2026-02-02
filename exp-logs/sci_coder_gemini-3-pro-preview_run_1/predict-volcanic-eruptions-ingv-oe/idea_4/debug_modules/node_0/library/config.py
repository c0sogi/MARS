import os
import torch


class Config:
    # ==========================================
    # Global Settings
    # ==========================================
    PROJECT_NAME = "volcano_eruption_prediction"
    IDEA_NAME = "idea_4"
    SEED = 42
    N_FOLDS = 5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching processed features/spectrograms
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing
    # ==========================================
    USE_CACHE = True
    SENSORS = [f"sensor_{i}" for i in range(1, 11)]
    SAMPLE_RATE = 100  # 60,000 samples / 600 seconds (10 mins)

    # Target Scaling for Neural Network (Branch B)
    # Options: 'log1p' (np.log1p), 'none'
    TARGET_SCALING = "log1p"

    # ==========================================
    # Branch A: Expert Features (LightGBM)
    # ==========================================
    LGB_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": 5000,
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "random_state": SEED,
        "n_jobs": -1,
    }
    LGB_EARLY_STOPPING_ROUNDS = 100

    # ==========================================
    # Branch B: Spectrograms + CNN
    # ==========================================
    # Spectrogram Generation
    N_FFT = 1024
    HOP_LENGTH = 256  # 60000 / 256 approx 234 time steps
    N_MELS = 128  # Height of the image
    F_MIN = 0
    F_MAX = None  # None implies Nyquist (Sample Rate / 2)

    # Image resizing for CNN input (Height, Width)
    # Width is determined by time steps, we can resize to fixed or keep native
    IMG_SIZE = (128, 256)

    # Model Architecture
    CNN_MODEL_NAME = "tf_efficientnet_b0_ns"
    CNN_IN_CHANNELS = 10  # One channel per sensor

    # Training Hyperparameters
    CNN_BATCH_SIZE = 32
    CNN_EPOCHS = 20
    CNN_LR = 1e-3
    CNN_WEIGHT_DECAY = 1e-4
    CNN_PATIENCE = 5  # Early stopping patience

    # ==========================================
    # Meta-Learner (Stacking)
    # ==========================================
    META_ALPHA = 1.0  # Ridge Regression Alpha

    @classmethod
    def create_dirs(cls):
        """Creates necessary directories for working and submission."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
