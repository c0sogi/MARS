import os


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching processed features (numpy/parquet)
    # Using 'idea_2' as per the specific iteration plan
    CACHE_DIR = "./working/idea_2"
    CACHE_TRAIN_DIR = os.path.join(CACHE_DIR, "cache_train")
    CACHE_VAL_DIR = os.path.join(CACHE_DIR, "cache_val")
    CACHE_TEST_DIR = os.path.join(CACHE_DIR, "cache_test")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4  # For DataLoader

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Gestures: 1-20. Class 0 is reserved for "Background" / "Null"
    NUM_CLASSES = 21

    # Label Mapping
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
    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

    # Skeleton
    # We use all 20 joints provided in the Kinect format
    JOINTS = [
        "HipCenter",
        "Spine",
        "ShoulderCenter",
        "Head",
        "ShoulderLeft",
        "ElbowLeft",
        "WristLeft",
        "HandLeft",
        "ShoulderRight",
        "ElbowRight",
        "WristRight",
        "HandRight",
        "HipLeft",
        "KneeLeft",
        "AnkleLeft",
        "FootLeft",
        "HipRight",
        "KneeRight",
        "AnkleRight",
        "FootRight",
    ]
    NUM_JOINTS = len(JOINTS)
    SKELETON_CHANNELS = 3  # (x, y, z)
    # HipCenter is index 0 in the list above, used for relative normalization
    HIP_CENTER_INDEX = 0

    # Audio
    # Video is 20 FPS. Audio is 16000 Hz.
    # To align audio features with video frames: Hop Length = 16000 / 20 = 800 samples.
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_N_MFCC = 13
    AUDIO_N_FFT = 2048
    AUDIO_HOP_LENGTH = 800

    # ==========================================
    # Model Architecture
    # ==========================================
    # Input dimensions
    INPUT_DIM_SKELETON = NUM_JOINTS * SKELETON_CHANNELS
    INPUT_DIM_AUDIO = AUDIO_N_MFCC

    # Feature Stem / Encoder
    HIDDEN_DIM = 256
    PROJECTION_DIM = 128
    STEM_KERNEL_SIZE = 7
    DROPOUT = 0.3

    # Recurrent Layer
    GRU_LAYERS = 2
    BIDIRECTIONAL = True

    # Refinement Stage
    REFINE_KERNEL_SIZE = 5
    REFINE_LAYERS = 2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    WEIGHT_DECAY = 1e-4

    # Scheduler
    LR_FACTOR = 0.5
    LR_PATIENCE = 3
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # Loss Configuration
    LABEL_SMOOTHING = 0.1
    # Weight for the background class (0) to balance recall/precision
    BACKGROUND_WEIGHT = 0.5

    # Augmentation
    AUG_MASK_CHANNEL_PROB = 0.5
    AUG_MASK_CHANNEL_RATIO = 0.1
    AUG_NOISE_SIGMA = 0.01

    # ==========================================
    # Post-Processing
    # ==========================================
    # Minimum duration (in frames) to consider a predicted gesture valid
    MIN_GESTURE_LENGTH = 5
    # Size of median filter kernel for smoothing predictions
    MEDIAN_FILTER_KERNEL = 7

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_TRAIN_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_VAL_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_TEST_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
