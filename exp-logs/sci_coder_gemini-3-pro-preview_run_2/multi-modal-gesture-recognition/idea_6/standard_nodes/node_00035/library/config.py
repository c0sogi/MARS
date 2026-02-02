import os
import torch
import numpy as np
import random


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/iteration
    WORKING_DIR = "./working/solution_optimization"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    # Gesture Vocabulary
    GESTURE_MAP = {
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

    # Classes: 0 = Background, 1-20 = Gestures
    NUM_CLASSES = 21

    # Skeleton Configuration
    # Indices for 12 Upper-Body Joints:
    # HipCenter(0), Spine(1), ShoulderCenter(2), Head(3)
    # ShoulderLeft(4), ElbowLeft(5), WristLeft(6), HandLeft(7)
    # ShoulderRight(8), ElbowRight(9), WristRight(10), HandRight(11)
    SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NUM_JOINTS = len(SELECTED_JOINTS)

    # Feature Dimensions
    # 3 coords (x,y,z) * 12 joints = 36
    # Velocity (3 coords) * 12 joints = 36
    # Audio MFCCs = 13 (standard)
    AUDIO_MFCC_N_MFCC = 13
    INPUT_DIM = (
        (NUM_JOINTS * 3) + (NUM_JOINTS * 3) + AUDIO_MFCC_N_MFCC
    )  # 36 + 36 + 13 = 85

    # =========================================================================
    # Model Architecture (IC-RCN)
    # =========================================================================
    HIDDEN_DIM = 128
    LSTM_LAYERS = 2
    TCN_LAYERS = 8  # Number of layers in each TCN stage
    TCN_KERNEL_SIZE = 3
    DROPOUT = 0.5  # High dropout as per Lesson 29

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 8  # Conservative batch size
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    PATIENCE = 10  # Early stopping patience

    # Loss Weights
    # Class weights: 0.1 for Background, 1.0 for Gestures
    CLASS_WEIGHTS = [0.1] + [1.0] * 20

    # TMSE (Truncated Mean Squared Error) Weight for smoothing
    # Conservative weight as per Lesson 29/31 analysis
    TMSE_WEIGHT = 3.0
    # TMSE_THRESHOLD removed as we switch to probability space (Cite Lesson 33)

    # Augmentation
    GAUSSIAN_NOISE_STD = 0.01

    # =========================================================================
    # Debug / Runtime Control
    # =========================================================================
    # Set to a small integer (e.g., 50) to debug with a subset of data
    # Set to None to use full dataset
    DEBUG_SUBSET_SIZE = None

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def ensure_dirs(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
