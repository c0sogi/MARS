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

    # Working directory for caching processed data (Idea 42 specific)
    WORK_DIR = "./working/idea_42"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary writeable directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Data Engineering & Preprocessing
    # ==========================================
    # Sampling parameters
    WINDOW_SIZE = 64
    STRIDE = 32

    # Skeleton Configuration
    NUM_JOINTS = 20
    # Features per joint: Position(3) + Velocity(3) + Acceleration(3) = 9
    CHANNELS_PER_JOINT = 9
    SKELETON_INPUT_DIM = NUM_JOINTS * CHANNELS_PER_JOINT  # 180

    # Audio Configuration
    AUDIO_N_MFCC = 13
    AUDIO_INPUT_DIM = AUDIO_N_MFCC  # 13

    # Total Input Dimension for the Learnable Diagonal Scaling Layer
    # 180 (Skeleton) + 13 (Audio) = 193
    INPUT_DIM = SKELETON_INPUT_DIM + AUDIO_INPUT_DIM

    # ==========================================
    # Label Configuration
    # ==========================================
    # 20 Gestures + 1 Background Class (ID 0)
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0

    # Mapping from dataset strings to integer IDs (1-20)
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

    # Inverse mapping for decoding predictions
    ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

    # ==========================================
    # Model Architecture (LSM-CN)
    # ==========================================
    # Bi-GRU Encoder
    HIDDEN_DIM = 192  # 96 units per direction
    DROPOUT = 0.3

    # TCN Refinement
    TCN_KERNEL_SIZE = 3
    # Monotonic Dilation Schedule for Receptive Field = 63
    TCN_DILATIONS = [1, 2, 4, 8, 16]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Loss Function Weights
    # Weight for the background class (0) to handle imbalance
    BACKGROUND_WEIGHT = 0.2

    # Deep Supervision & Smoothness
    # Loss = CE + lambda * Smoothness
    SMOOTHING_LAMBDA = 0.15
    # Truncated MSE threshold for log-probabilities
    SMOOTHING_THRESHOLD = 1.0

    # ==========================================
    # Post-Processing / Inference
    # ==========================================
    # Minimum duration (in frames) for a detected gesture to be considered valid
    MIN_GESTURE_DURATION = 5

    @staticmethod
    def seed_everything(seed=42):
        """
        Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
