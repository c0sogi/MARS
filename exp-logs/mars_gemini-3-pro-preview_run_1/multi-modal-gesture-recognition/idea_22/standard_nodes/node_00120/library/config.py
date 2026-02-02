import os
import torch
import numpy as np
import random


class Config:
    """
    Central configuration for the Multi-Scale Context-Gated Input-Injected Network (MSC-IIN).
    """

    # ==========================================
    # Reproducibility & Environment
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to limit dataset size for debugging
    MAX_SAMPLES = None  # If not None, limits the number of samples loaded

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_22"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Specific file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Processing Constants
    # ==========================================
    # Video
    VIDEO_FPS = 20

    # Audio
    AUDIO_SAMPLE_RATE = 16000
    # Physics-Based Hop Length: SampleRate / FPS = 16000 / 20 = 800
    # This ensures 1 audio feature vector corresponds exactly to 1 video frame
    AUDIO_HOP_LENGTH = 800
    N_MFCC = 13
    N_FFT = 2048

    # Skeleton
    # 20 joints * 3 coordinates (X, Y, Z)
    NUM_JOINTS = 20
    SKELETON_INPUT_CHANNELS = 60

    # Label Map
    # 0 is reserved for the background/null class
    BACKGROUND_CLASS_ID = 0
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
    # Total classes = 20 gestures + 1 background
    NUM_CLASSES = 21

    # ==========================================
    # Model Architecture (MSC-IIN)
    # ==========================================
    # Multi-Scale Temporal Stems
    KERNEL_SIZES = [3, 7, 11]

    # Feature Dimensions
    AUDIO_EMBED_DIM = 64
    SKELETON_EMBED_DIM = 64
    # Fused dimension before backbone
    FUSED_DIM = 128

    # Backbone
    HIDDEN_DIM = 256
    DROPOUT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 8  # Micro-batching
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3

    # Optimization & Regularization
    WEIGHT_DECAY = 0.05  # Aggressive regularization
    LABEL_SMOOTHING = 0.1
    BACKGROUND_WEIGHT = 0.5  # Weight for class 0 to prevent collapse

    # Augmentation
    TEMPORAL_RESAMPLE_RANGE = (0.8, 1.2)
    CHANNEL_MASK_PROB = 0.1

    # ==========================================
    # Inference / Post-Processing
    # ==========================================
    MEDIAN_FILTER_WINDOW = 5
    MIN_SEGMENT_LENGTH = 5

    @staticmethod
    def set_seed():
        """
        Sets fixed seeds for reproducibility across random, numpy, and torch.
        """
        seed = Config.SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def get_device():
        """
        Returns the appropriate torch device.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
