import os


class Config:
    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_28"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (Parquet/NPY)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_features.npy")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_features.npy")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_features.npy")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Data Processing Hyperparameters
    # ==========================================
    RANDOM_SEED = 42

    # Sampling
    WINDOW_SIZE = 64
    STRIDE_TRAIN = 32  # Moderate stride to avoid overfitting
    STRIDE_TEST = 32  # 50% overlap for sliding window inference

    # Input Dimensions
    NUM_JOINTS = 20  # Kinect Skeleton joints
    CHANNELS_PER_JOINT = 9  # (Pos(3) + Vel(3) + Acc(3))
    AUDIO_FEATURES = 13  # MFCCs (example default, adjusted in feature extraction)

    # The input dimension to the model will be (NUM_JOINTS * CHANNELS_PER_JOINT) + AUDIO_FEATURES
    # 20 * 9 + 13 = 193

    # ==========================================
    # 3. Model Architecture Hyperparameters
    # ==========================================
    # Bi-GRU Encoder
    ENCODER_HIDDEN_SIZE = 128  # Per direction (Total 256)
    ENCODER_LAYERS = 1

    # MS-TCN Refinement
    TCN_KERNEL_SIZE = 3
    # Monotonically increasing dilation schedule
    # Receptive field calculation: 1 + 2*(1+2+4+8+16) = 1 + 62 = 63 frames
    # Fits within WINDOW_SIZE (64)
    TCN_DILATIONS = [1, 2, 4, 8, 16]
    TCN_CHANNELS = 64

    # Classes: 0=Background, 1-20=Gestures
    NUM_CLASSES = 21

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Loss Weights
    WEIGHT_BACKGROUND = 0.2  # Suppress background dominance
    WEIGHT_SMOOTHING = 0.15  # Weight for the smoothing loss
    SMOOTHING_THRESHOLD = 1.0  # Truncation threshold for log-space smoothing

    # ==========================================
    # 5. Post-Processing
    # ==========================================
    MIN_GESTURE_DURATION = 5  # Minimum frames to keep a prediction

    # ==========================================
    # 6. Vocabulary Mapping
    # ==========================================
    # 0 is reserved for background
    GESTURE_MAP = {
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
    ID_TO_NAME = {v: k for k, v in GESTURE_MAP.items()}

    @classmethod
    def setup_directories(cls):
        """Ensures that working and submission directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_gesture_name(cls, gesture_id):
        """Returns the name of the gesture for a given ID."""
        return cls.ID_TO_NAME.get(gesture_id, "Unknown")


# Initialize directories on import
Config.setup_directories()
