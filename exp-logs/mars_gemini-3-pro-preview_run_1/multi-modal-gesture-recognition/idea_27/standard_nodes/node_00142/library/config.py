import os
import torch


class Config:
    """
    Configuration class for the Magnitude-Preserving Context-Injected Network (MPC-Net).
    Handles paths, hyperparameters, and global constants.
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea execution
    WORKING_DIR = "./working/idea_27"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure all working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Label Configuration
    # -------------------------------------------------------------------------
    # Class 0 is strictly reserved for Background/Padding
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
    NUM_CLASSES = len(LABEL_MAP)  # 21 classes (0-20)
    BACKGROUND_CLASS_ID = 0

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Audio
    AUDIO_SAMPLE_RATE = 16000
    VIDEO_FPS = 20
    # Physics-Based Hop Length: SampleRate / VideoFPS = 16000 / 20 = 800
    # This ensures 1 audio frame corresponds exactly to 1 video frame
    HOP_LENGTH = 800
    # Window Size ~ 2.5x Hop Length for sufficient overlap
    N_FFT = 2048
    # Compact MFCCs as requested
    N_MFCC = 20

    # Skeleton
    # 20 Joints * 3 coordinates (x, y, z)
    NUM_JOINTS = 20
    SKELETON_CHANNELS = 60

    # Augmentation
    TEMPORAL_RESAMPLE_MIN = 0.8
    TEMPORAL_RESAMPLE_MAX = 1.2
    CHANNEL_MASK_PROB = 0.1

    # -------------------------------------------------------------------------
    # Model Architecture (MPC-Net)
    # -------------------------------------------------------------------------
    STEM_KERNEL_SIZE = 7
    STEM_CHANNELS = 256
    BACKBONE_HIDDEN_DIM = 256
    BACKBONE_LAYERS = 2
    DROPOUT_RATE = 0.3  # Base dropout rate for regularization

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 8
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05
    LABEL_SMOOTHING = 0.1

    # Class Weights: Background gets 0.5 to prevent collapse, others get 1.0
    BACKGROUND_WEIGHT_VALUE = 0.5

    # Debugging
    # Set to an integer (e.g., 50) to limit dataset size for quick debugging, or None for full run
    DEBUG_SUBSET_SIZE = None

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
