import os


class Config:
    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea (idea_11)
    WORK_DIR = "./working/idea_11"

    # Cache directory for processed tensors (npy/parquet)
    CACHE_DIR = os.path.join(WORK_DIR, "cache")

    # Checkpoint directory for saving model weights
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Ensure directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Video
    VIDEO_FPS = 20.0

    # Audio
    AUDIO_SAMPLE_RATE = 16000
    # Physics-Based Hop Length: Ensures 1-to-1 mapping between audio frames and video frames
    # Hop = SampleRate / VideoFPS = 16000 / 20 = 800 samples per frame
    AUDIO_HOP_LENGTH = int(AUDIO_SAMPLE_RATE / VIDEO_FPS)
    N_MFCC = 13
    N_FFT = 2048

    # Skeleton
    # 20 joints * 3 coordinates (X, Y, Z)
    NUM_JOINTS = 20
    SKELETON_INPUT_DIM = NUM_JOINTS * 3

    # Normalization
    # These should be computed from training set, but placeholders or logic to compute them can be used
    # The pipeline will likely compute these and save to stats.npz in the cache dir

    # Caching Flag
    LOAD_CACHED_DATA = True

    # ==========================================
    # Label Configuration
    # ==========================================
    # 20 Gesture Classes + 1 Background Class (Index 0)
    NUM_CLASSES = 21

    # Mapping from Name to ID (1-20). 0 is reserved for background.
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
    # Reverse mapping
    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}
    ID_TO_NAME[0] = "background"

    # ==========================================
    # Model Architecture (RCGR-Net)
    # ==========================================
    # Input Stems
    KERNEL_SIZE_SKELETON = 7
    KERNEL_SIZE_AUDIO = 7

    # Backbone
    HIDDEN_DIM = (
        512  # Projected dimension, wider than input (Cite solution_lesson_node_00061)
    )
    NUM_LAYERS = 2  # Number of Recursive Gated-Residual Blocks
    DROPOUT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Micro-Batch Optimization
    BATCH_SIZE = 8

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05  # Aggressive regularization
    EPOCHS = 50

    # Loss Function
    LABEL_SMOOTHING = 0.1
    BG_WEIGHT = 0.5  # Weight for class 0 (background) to balance recall/precision

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # ==========================================
    # Inference / Post-Processing
    # ==========================================
    MEDIAN_FILTER_KERNEL = 5
    MIN_GESTURE_LENGTH = 5  # Frames

    # ==========================================
    # Debugging
    # ==========================================
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SUBSET_SIZE = 10
