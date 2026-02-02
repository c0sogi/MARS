import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the Decoupled-Anchor Gated-Injection Network (DAGI-Net) experiment.
    """

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on vCPU availability (12 vCPUs available)

    # ==========================================
    # Data Parameters
    # ==========================================
    # Classes: 0 = Background, 1-20 = Gestures
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # Video / Skeleton
    VIDEO_FPS = 20
    NUM_JOINTS = 20
    SKELETON_CHANNELS = 3  # (x, y, z)
    SKELETON_INPUT_DIM = NUM_JOINTS * SKELETON_CHANNELS  # 60

    # Audio (Physics-Based Alignment)
    SAMPLE_RATE = 16000
    # Hop length = SampleRate / VideoFPS = 16000 / 20 = 800
    # This ensures 1 audio frame per video frame.
    HOP_LENGTH = 800
    N_FFT = 2048  # Larger window for better frequency resolution and overlap
    N_MFCC = 13  # Compact MFCCs to prevent noise overfitting

    # Normalization (Computed from analysis or standard values)
    # These can be refined during the preprocessing stage
    SKELETON_MEAN = 0.0
    SKELETON_STD = 1.0

    # ==========================================
    # Model Architecture (DAGI-Net)
    # ==========================================
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2  # BiGRU layers
    DROPOUT = 0.3
    CNN_KERNEL_SIZE = 7  # Large receptive field for local stems

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 8  # Micro-batching strategy
    NUM_EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05  # Aggressive regularization
    PATIENCE = 10  # Early stopping patience

    # Loss Configuration
    LABEL_SMOOTHING = 0.1
    BACKGROUND_WEIGHT = 0.5  # Prevent model collapse on dominant background class

    # ==========================================
    # Label Mapping
    # ==========================================
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

    @staticmethod
    def get_class_weights():
        """
        Returns the class weight tensor for CrossEntropyLoss.
        Background class (0) gets 0.5, others get 1.0.
        """
        weights = torch.ones(Config.NUM_CLASSES)
        weights[Config.BACKGROUND_CLASS_ID] = Config.BACKGROUND_WEIGHT
        return weights.to(Config.DEVICE)


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
