import os


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea (Idea 18)
    WORK_DIR = "./working/idea_18"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4  # Adjust based on vCPUs (12 available)

    # ==========================================
    # Data Processing
    # ==========================================
    FPS = 20
    AUDIO_SAMPLING_RATE = 16000

    # Physics-Based Audio Alignment
    # Hop length = SR / FPS = 16000 / 20 = 800 samples per frame
    AUDIO_HOP_LENGTH = 800
    AUDIO_N_FFT = 2048  # Large window for overlap
    AUDIO_N_MELS = 64  # Number of Mel bands

    # Skeleton
    NUM_JOINTS = 20
    NUM_CHANNELS = 3  # x, y, z

    # Classes: 20 Gestures + 1 Background (Index 0)
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # Label Map (Name -> ID). Note: Input labels are 1-20.
    # We will map 1-20 to 1-20, and use 0 for background.
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
    ID_TO_NAME[0] = "background"

    # ==========================================
    # Model Architecture
    # ==========================================
    HIDDEN_DIM = 256
    DROPOUT = 0.3

    # Skeleton Stem
    SKELETON_INPUT_DIM = NUM_JOINTS * NUM_CHANNELS  # 60
    KERNEL_SIZE_SKELETON = 7

    # Audio Stem
    AUDIO_INPUT_DIM = AUDIO_N_MELS
    KERNEL_SIZE_AUDIO = 5  # Slightly smaller for audio features

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 8  # Micro-batching strategy
    LEARNING_RATE = 1e-3  # Initial LR
    WEIGHT_DECAY = 0.05  # Aggressive regularization
    EPOCHS = 50  # Max epochs
    PATIENCE = 10  # Early stopping patience

    # Loss Weights
    BG_WEIGHT = 0.5  # Crucial fix from Lesson 00095
    LABEL_SMOOTHING = 0.1  # Handle boundary ambiguity

    # ==========================================
    # Augmentation (Global Manifold)
    # ==========================================
    RESAMPLE_LOW = 0.8
    RESAMPLE_HIGH = 1.2
    CHANNEL_MASK_RATIO = 0.1

    # ==========================================
    # Inference / Post-Processing
    # ==========================================
    MEDIAN_FILTER_WINDOW = 5
    MIN_SEGMENT_LENGTH = 5

    # Debugging
    DEBUG_SUBSET_SIZE = None  # Set to int (e.g. 10) to debug on small data
