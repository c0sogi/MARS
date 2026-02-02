import os
import torch


class Config:
    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on vCPUs (12 available)

    # ==========================================
    # File Paths
    # ==========================================
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directories (Write Allowed)
    # Using specific idea directory as requested
    WORKING_DIR = "./working/idea_28"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure writable directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    # Gestures
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
    # Inverse map for decoding
    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}
    NUM_CLASSES = 20

    # Special Tokens / Padding
    # We use 0 as background/padding class for training targets
    BACKGROUND_CLASS_ID = 0

    # Skeleton Structure (Kinect v2)
    SKELETON_JOINTS = [
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
    NUM_JOINTS = 20
    ROOT_JOINT_NAME = "HipCenter"
    ROOT_JOINT_IDX = SKELETON_JOINTS.index(ROOT_JOINT_NAME)

    # Audio / Video Physics
    VIDEO_FPS = 20.0
    AUDIO_SAMPLE_RATE = 16000

    # Audio Feature Extraction
    # Hop length calculated to align audio frames with video frames: 16000 / 20 = 800
    AUDIO_HOP_LENGTH = 800
    AUDIO_N_FFT = 2048  # Window size ~2.5x hop length
    AUDIO_N_MFCC = 13  # Compact MFCCs

    # Input Dimensions
    # Skeleton: 20 joints * 3 coordinates (x, y, z)
    INPUT_DIM_SKELETON = NUM_JOINTS * 3
    INPUT_DIM_AUDIO = AUDIO_N_MFCC

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Backbone
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT_RATE = 0.3

    # Gated Stem
    STEM_KERNEL_SIZE = 7

    # Heads
    USE_BOUNDARY_HEAD = True

    # ==========================================
    # Training Configuration
    # ==========================================
    BATCH_SIZE = 8
    NUM_EPOCHS = 50  # Upper limit, controlled by Early Stopping
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05

    # Loss Weights
    LOSS_WEIGHT_CLASS = 1.0
    LOSS_WEIGHT_BOUNDARY = 0.5
    LABEL_SMOOTHING = 0.1

    # Optimization
    GRADIENT_CLIP_VAL = 1.0
    EARLY_STOPPING_PATIENCE = 10

    # Augmentation
    TEMPORAL_RESAMPLE_MIN = 0.8
    TEMPORAL_RESAMPLE_MAX = 1.2
    CHANNEL_MASK_PROB = 0.1

    @classmethod
    def print_config(cls):
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"Audio Hop: {cls.AUDIO_HOP_LENGTH}")
        print(f"Audio Window: {cls.AUDIO_N_FFT}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print("=" * 30)
