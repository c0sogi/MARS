import os


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/run
    WORK_DIR = "./working/idea_21"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORK_DIR, "submission")

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission file
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    SEED = 42

    # Video/Skeleton
    FRAME_RATE = 20
    NUM_JOINTS = 20
    # Root-relative coordinates (x, y, z) -> 3 channels
    SKELETON_CHANNELS = 3

    # Audio (Physics-Based Alignment)
    AUDIO_SAMPLERATE = 16000
    # Hop length matches video frame rate: 16000 / 20 = 800
    AUDIO_HOP_LENGTH = 800
    # Larger window for overlap
    AUDIO_N_FFT = 2048
    AUDIO_N_MFCC = 13  # Compact MFCCs

    # Augmentation
    TEMPORAL_RESAMPLE_MIN = 0.8
    TEMPORAL_RESAMPLE_MAX = 1.2
    CHANNEL_MASK_PROB = 0.1

    # ==========================================
    # Model Hyperparameters (SR-IIN)
    # ==========================================
    HIDDEN_DIM = 256
    KERNEL_SIZE = 7  # Large receptive field
    DROPOUT = 0.3
    NUM_LAYERS = 2  # BiGRU layers

    # ==========================================
    # Training Parameters
    # ==========================================
    BATCH_SIZE = 8  # Micro-batching
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05  # Aggressive regularization

    # Loss Configuration
    BG_WEIGHT = 0.5  # Prevent model collapse on background
    LABEL_SMOOTHING = 0.1

    # Inference
    MEDIAN_FILTER_KERNEL = 5
    MIN_SEGMENT_LENGTH = 5

    # ==========================================
    # Label Mapping
    # ==========================================
    # 0 is reserved for Background
    LABEL_MAP = {
        "background": 0,
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
    NUM_CLASSES = len(LABEL_MAP)

    @classmethod
    def setup(cls):
        """Ensure necessary working directories exist."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
