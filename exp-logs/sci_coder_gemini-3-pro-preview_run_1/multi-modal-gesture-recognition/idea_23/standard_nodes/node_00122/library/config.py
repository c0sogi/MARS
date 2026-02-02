import os
import torch


class Config:
    """
    Configuration for the Deep Wide-Spectrum Anchored Input-Injected Network (DW-AIIN).
    Handles file paths, hyperparameters, and constants for the gesture recognition task.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Idea 23)
    WORK_DIR = "./working/idea_23"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Constants
    # ==========================================
    # 20 Gestures + 1 Background
    NUM_CLASSES = 21
    BACKGROUND_LABEL = 0

    # Label Mapping
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

    # Physics-Based Audio Parameters
    VIDEO_FPS = 20
    AUDIO_SAMPLE_RATE = 16000

    # Hop Length = SampleRate / VideoFPS = 16000 / 20 = 800 samples per frame
    # This aligns audio features exactly with video frames
    HOP_LENGTH = int(AUDIO_SAMPLE_RATE / VIDEO_FPS)

    # Large Window Size (4 x Hop) for better frequency resolution
    N_FFT = 4 * HOP_LENGTH  # 3200

    # Compact MFCCs
    N_MFCC = 13

    # Skeleton Input Dimensions
    # 20 joints * 3 coordinates (X, Y, Z)
    SKELETON_NUM_JOINTS = 20
    SKELETON_CHANNELS = 3
    SKELETON_INPUT_DIM = SKELETON_NUM_JOINTS * SKELETON_CHANNELS  # 60

    # ==========================================
    # Model Hyperparameters (DW-AIIN)
    # ==========================================
    HIDDEN_DIM = 256

    # Wide Single-Scale Stem
    WIDE_STEM_KERNEL_SIZE = 9

    # Deep Input-Injected Backbone
    BACKBONE_LAYERS = 3  # 3-layer BiGRU

    DROPOUT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 8  # Micro-batching
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05  # Aggressive regularization

    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Loss Configuration
    LABEL_SMOOTHING = 0.1
    BACKGROUND_WEIGHT = 0.5  # Prevent model collapse on background

    # Inference
    MEDIAN_FILTER_WINDOW = 5
    MIN_GESTURE_LENGTH = 5

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_loss_weights(cls):
        """Returns class weights tensor for CrossEntropyLoss."""
        weights = torch.ones(cls.NUM_CLASSES)
        weights[cls.BACKGROUND_LABEL] = cls.BACKGROUND_WEIGHT
        return weights.to(cls.DEVICE)
