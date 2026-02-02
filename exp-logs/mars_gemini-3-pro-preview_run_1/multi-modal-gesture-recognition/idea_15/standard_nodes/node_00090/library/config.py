import os
import torch


class Config:
    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching and checkpoints (Idea 15)
    WORKING_DIR = "./working/idea_15"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Constants
    # ==========================================
    SEED = 42

    # Label Mapping (1-20 based on prompt)
    # Class 0 is reserved for BACKGROUND / NULL
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

    # Reverse mapping for decoding
    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

    # Total classes = 20 gestures + 1 background
    NUM_CLASSES = len(LABEL_MAP)
    BACKGROUND_CLASS_ID = 0

    # ==========================================
    # 3. Data Processing Hyperparameters
    # ==========================================
    # Video
    VIDEO_FPS = 20.0

    # Audio
    AUDIO_SAMPLE_RATE = 16000
    # Hop length aligned to video frame rate: 16000 / 20 = 800 samples per frame
    AUDIO_HOP_LENGTH = int(AUDIO_SAMPLE_RATE / VIDEO_FPS)
    AUDIO_N_FFT = 2048  # Large window for overlap
    AUDIO_N_MFCC = 13

    # Skeleton
    # 20 Joints * 3 Coordinates (X, Y, Z)
    NUM_JOINTS = 20
    INPUT_DIM_SKELETON = NUM_JOINTS * 3
    INPUT_DIM_AUDIO = AUDIO_N_MFCC

    # ==========================================
    # 4. Model Hyperparameters (GCA-IIN)
    # ==========================================
    # Feature Extraction
    SKELETON_EMBED_DIM = 64
    AUDIO_EMBED_DIM = 64
    KERNEL_SIZE = 7  # Temporal Conv1d kernel size

    # Recurrent Backbone
    HIDDEN_DIM = 256
    NUM_RNN_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.3

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 8  # Micro-batching strategy
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05

    # Augmentation
    AUGMENT_PROB = 0.5
    NOISE_SIGMA = 0.01
    MASK_TIME_PROB = 0.2
    MASK_TIME_MAX_FRAMES = 10
    MASK_CHANNEL_PROB = 0.2
    MASK_CHANNEL_RATIO = 0.1

    # Loss
    LABEL_SMOOTHING = 0.1
    # Weight for background class (0) vs others
    # We keep background weight lower or moderate to prevent it from dominating
    BACKGROUND_WEIGHT = 0.5

    # Early Stopping
    PATIENCE = 10

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 6. Inference
    # ==========================================
    MEDIAN_FILTER_KERNEL = 5
    MIN_GESTURE_LENGTH = 5  # Frames
