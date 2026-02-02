import os
import torch


class Config:
    # ==========================================
    # System & Paths
    # ==========================================
    SEED = 42

    # Input Data Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working & Output Paths
    # Using 'idea_26' as the designated working folder for this iteration
    WORKING_DIR = "./working/idea_26"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    STATS_PATH = os.path.join(WORKING_DIR, "stats.npz")

    # Device Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2  # Adjust based on vCPUs (12 available)

    # ==========================================
    # Dataset & Labels
    # ==========================================
    # 0 is reserved for Background/Padding
    BACKGROUND_LABEL = 0

    # Vocabulary of 20 Italian gestures
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

    # Inverse mapping for decoding
    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

    # Total classes = 20 gestures + 1 background
    NUM_CLASSES = len(LABEL_MAP) + 1

    # ==========================================
    # Preprocessing Hyperparameters
    # ==========================================
    # Audio
    AUDIO_SAMPLE_RATE = 16000
    MFCC_N_FFT = 2048  # ~128ms window (Cite solution_lesson_node_00128)
    MFCC_HOP_LENGTH = 800  # Physics-based: 16000Hz / 20FPS = 800 samples per frame
    MFCC_N_MFCC = 13  # Compact MFCCs

    # Video / Skeleton
    VIDEO_FPS = 20.0
    SKELETON_JOINTS = 20
    SKELETON_CHANNELS = 3  # (X, Y, Z)

    # Augmentation
    TEMPORAL_RESAMPLE_MIN = 0.8
    TEMPORAL_RESAMPLE_MAX = 1.2
    CHANNEL_MASK_PROB = 0.1

    # ==========================================
    # Model Architecture (MPWI-Net)
    # ==========================================
    HIDDEN_DIM = 256
    KERNEL_SIZE = 7
    DROPOUT_RATE = 0.3
    GRU_LAYERS = 2
    BIDIRECTIONAL = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 8  # Micro-batching
    NUM_EPOCHS = 50  # Max epochs, relies on early stopping
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05  # Aggressive regularization

    # Loss Configuration
    BACKGROUND_WEIGHT = 0.5  # Prevent model collapse on silence
    LABEL_SMOOTHING = 0.1  # Handle boundary ambiguity

    # ==========================================
    # Inference / Post-processing
    # ==========================================
    MEDIAN_FILTER_WINDOW = 5
    MIN_SEGMENT_LENGTH = 5


# Ensure directories exist
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
