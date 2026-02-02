import os
import torch
import random
import numpy as np


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    STATS_PATH = os.path.join(WORKING_DIR, "stats.npz")

    # Ensure directories exist
    for d in [WORKING_DIR, CACHE_DIR, CHECKPOINT_DIR, SUBMISSION_DIR]:
        os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Audio
    AUDIO_SAMPLE_RATE = 16000
    VIDEO_FPS = 20.0
    # Physics-based hop length: SR / FPS = 16000 / 20 = 800 samples per frame
    AUDIO_HOP_LENGTH = int(AUDIO_SAMPLE_RATE / VIDEO_FPS)
    AUDIO_N_FFT = 2048
    AUDIO_N_MFCC = 13

    # Skeleton
    # 20 joints * 3 coordinates (X, Y, Z)
    NUM_JOINTS = 20
    SKELETON_INPUT_DIM = NUM_JOINTS * 3

    # -------------------------------------------------------------------------
    # Model Architecture (SCR-Net)
    # -------------------------------------------------------------------------
    NUM_CLASSES = 21  # 20 gestures + 1 background class (index 0)

    # Dimensions
    SKELETON_EMBED_DIM = 64
    AUDIO_EMBED_DIM = 64
    FUSION_DIM = 128  # Sum of embed dims
    HIDDEN_DIM = 256  # Backbone width (wider than input)

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05

    # Regularization
    DROPOUT = 0.3
    LABEL_SMOOTHING = 0.1

    # Loss Weights
    # Background class (0) gets 0.7 weight to prevent insertion errors
    BACKGROUND_WEIGHT = 0.7

    # -------------------------------------------------------------------------
    # Augmentation
    # -------------------------------------------------------------------------
    AUG_CHANNEL_MASK_RATE = 0.1  # Probability of masking a channel
    AUG_TIME_MASK_PROB = 0.5  # Probability of applying time masking
    AUG_TIME_MASK_LEN_MIN = 5  # Min frames to mask
    AUG_TIME_MASK_LEN_MAX = 15  # Max frames to mask
    AUG_GAUSSIAN_NOISE_STD = 0.01  # Std dev for additive noise on skeleton

    # -------------------------------------------------------------------------
    # Inference / Post-processing
    # -------------------------------------------------------------------------
    MEDIAN_FILTER_WINDOW = 5
    MIN_GESTURE_LENGTH = 5

    # -------------------------------------------------------------------------
    # Compute
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading

    # -------------------------------------------------------------------------
    # Label Mapping
    # -------------------------------------------------------------------------
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

    # Reverse mapping: 0 is reserved for background
    ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}
    ID_TO_NAME[0] = "background"


# Set seeds immediately upon import
seed_everything(Config.SEED)
