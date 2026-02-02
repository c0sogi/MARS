import os


class Config:
    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (Parquet/NPY)
    # We define paths here for other modules to use
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    NUM_WORKERS = 2  # For data loading

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Target Frame Rate for resampling both Video and Audio features
    TARGET_FPS = 20

    # Skeleton Data
    NUM_JOINTS = 20
    JOINTS_DIM = 3
    SKELETON_FEATURE_DIM = NUM_JOINTS * JOINTS_DIM  # 60

    # Audio Data
    AUDIO_FEATURE_DIM = 13  # Number of MFCCs

    # Combined Input Dimension
    INPUT_DIM = SKELETON_FEATURE_DIM + AUDIO_FEATURE_DIM  # 73

    # Sliding Window for Training
    WINDOW_SIZE = 64  # Number of frames per sequence sample
    STRIDE = 32  # Overlap stride

    # Classes
    # 20 Gestures + 1 Background (Class 0)
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # ==========================================
    # Model Architecture
    # ==========================================
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5

    # Class Weights (Optional, can be calculated dynamically or set here)
    # Background class (0) usually dominates, so we might weight it less
    # or weight gestures more.
    USE_CLASS_WEIGHTS = True

    # ==========================================
    # Post-Processing
    # ==========================================
    # Minimum number of consecutive frames to consider a valid gesture prediction
    MIN_GESTURE_LENGTH = 5
    # Kernel size for median filtering of predictions
    SMOOTHING_KERNEL_SIZE = 7

    # ==========================================
    # Label Mapping
    # ==========================================
    # Maps gesture names to IDs (1-20). 0 is reserved for background.
    LABEL_MAP = {
        "vattene": 1,
        "vieniqui": 2,
        "perfetto": 3,
        "furbo": 4,
        "cheduepalle": 5,
        "chevuoi": 6,
        "daccordo": 7,
        "seipazzo": 8,
        "combinato": 9,
        "freganiente": 10,
        "ok": 11,
        "cosatifarei": 12,
        "basta": 13,
        "prendere": 14,
        "noncenepiu": 15,
        "fame": 16,
        "tantotempo": 17,
        "buonissimo": 18,
        "messidaccordo": 19,
        "sonostufo": 20,
    }

    # Reverse mapping for decoding
    ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

    @classmethod
    def setup(cls):
        """Ensure necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Run setup on import to ensure directories exist
Config.setup()
