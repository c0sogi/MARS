import os
import torch


class Config:
    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORK_DIR = "./working/idea_opt"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    SEED = 42

    # Audio
    AUDIO_SAMPLE_RATE = 16000
    N_MFCC = 13
    N_FFT = 2048
    HOP_LENGTH = 512

    # Skeleton
    VIDEO_FPS = 20
    N_JOINTS = 20
    COORDS_PER_JOINT = 3
    INPUT_DIM_SKELETON = N_JOINTS * COORDS_PER_JOINT  # 60
    INPUT_DIM_AUDIO = N_MFCC

    # Labels
    NUM_CLASSES = 21  # 20 gestures + 1 background
    BACKGROUND_CLASS_ID = 0

    # Label Map (Name to ID)
    # Note: Dataset labels are 1-20. We map them to 1-20, and use 0 for background/null.
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
    # ID to Name
    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Architecture
    HIDDEN_DIM = 256
    NUM_HEADS = 4
    KERNEL_SIZE = 7  # For temporal convolution stems
    DROPOUT = 0.3

    # Training
    BATCH_SIZE = 8
    NUM_EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05

    # Loss
    LABEL_SMOOTHING = 0.1
    BG_WEIGHT = 0.5  # Weight for background class in Loss

    # Scheduler
    T_MAX = 60  # For CosineAnnealingLR

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
