import os

# Label Mapping for 20 Italian Gestures
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


class Config:
    # Reproducibility
    SEED = 42

    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working Directory for Idea 24
    WORK_DIR = "./working/idea_24"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORK_DIR, "submission")

    # Create directories
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Parameters
    # 20 gestures + 1 background class (ID 0)
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # Audio Physics-Based Parameters
    AUDIO_SAMPLE_RATE = 16000
    VIDEO_FPS = 20
    # Hop length = Samples per frame to align audio/video
    HOP_LENGTH = int(AUDIO_SAMPLE_RATE / VIDEO_FPS)  # 800
    # Large window size (4x Hop) for spectral continuity
    WIN_LENGTH = 4 * HOP_LENGTH  # 3200
    # N_FFT should be >= WIN_LENGTH, using next power of 2 for efficiency
    N_FFT = 4096
    N_MFCC = 13  # Compact MFCCs

    # Model Architecture
    HIDDEN_DIM = 256
    DROPOUT = 0.3

    # Training Hyperparameters
    BATCH_SIZE = 8  # Micro-batching
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05  # Aggressive regularization

    # Loss Weights
    BACKGROUND_WEIGHT = 0.5  # Prevent model collapse
    LABEL_SMOOTHING = 0.1  # Handle boundary ambiguity

    # Debugging
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 20

    @classmethod
    def get_class_weights(cls, device):
        """
        Returns the class weight tensor.
        Background (0) gets 0.5, others get 1.0.
        """
        import torch

        weights = torch.ones(cls.NUM_CLASSES, device=device)
        weights[cls.BACKGROUND_CLASS_ID] = cls.BACKGROUND_WEIGHT
        return weights
