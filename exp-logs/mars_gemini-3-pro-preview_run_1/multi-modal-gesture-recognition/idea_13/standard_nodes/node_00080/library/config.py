import os
import random
import numpy as np
import torch


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and models
    WORKING_DIR = "./working/idea_13"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Stats file for normalization
    STATS_PATH = os.path.join(WORKING_DIR, "stats.npz")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    SEED = 42

    # Label Configuration
    # 0 is reserved for Background. 1-20 are the provided gestures.
    BACKGROUND_LABEL = 0
    NUM_CLASSES = 21  # 20 gestures + 1 background

    # Mapping from Gesture Name to ID (1-20)
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
    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}
    ID_TO_NAME[BACKGROUND_LABEL] = "background"

    # Signal Processing
    VIDEO_FPS = 20.0
    AUDIO_SAMPLE_RATE = 16000
    # Physics-Based Hop Length: SR / FPS ensures 1 audio frame per video frame
    AUDIO_HOP_LENGTH = int(AUDIO_SAMPLE_RATE / VIDEO_FPS)
    N_MFCC = 13

    # Skeleton
    # We expect 20 joints * 3 coordinates = 60 features
    SKELETON_INPUT_SIZE = 60

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Micro-batching strategy
    BATCH_SIZE = 8

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05
    EPOCHS = 100  # High ceiling, controlled by Early Stopping
    EARLY_STOPPING_PATIENCE = 15

    # Loss Configuration
    LABEL_SMOOTHING = 0.1
    BACKGROUND_WEIGHT = 0.5  # Weight for class 0

    # Model Architecture
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3

    # Inference
    MEDIAN_FILTER_KERNEL = 5
    MIN_GESTURE_LENGTH = 5


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
