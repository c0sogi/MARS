import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    # ==========================================
    # Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "best_model.pth")
    CACHE_FILE_TRAIN = os.path.join(WORKING_DIR, "train_data_cache.npy")
    CACHE_FILE_VAL = os.path.join(WORKING_DIR, "val_data_cache.npy")
    CACHE_FILE_TEST = os.path.join(WORKING_DIR, "test_data_cache.npy")

    # ==========================================
    # Gesture Vocabulary
    # ==========================================
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
    # Inverse mapping for decoding predictions
    ID_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

    # Class 0 is reserved for "background" / "no gesture"
    NUM_CLASSES = 21

    # ==========================================
    # Data Processing Configuration
    # ==========================================
    # Skeleton: Upper body joints focus (Indices 0-11 based on dataset description order)
    # 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head,
    # 4:ShoulderLeft, 5:ElbowLeft, 6:WristLeft, 7:HandLeft,
    # 8:ShoulderRight, 9:ElbowRight, 10:WristRight, 11:HandRight
    SELECTED_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    NUM_JOINTS = len(SELECTED_JOINT_INDICES)

    # Audio: MFCC Extraction
    AUDIO_SR = 16000
    N_MFCC = 13
    HOP_LENGTH = 512  # Adjust to match video frame rate approx

    # Input Dimension Calculation
    # Skeleton: 12 joints * 3 coords (x,y,z) = 36
    # Velocity: 12 joints * 3 coords = 36
    # Audio: 13 MFCCs
    # Total = 36 + 36 + 13 = 85
    INPUT_DIM = 85

    # Sequence Handling
    MAX_SEQ_LEN = 200  # Pad/Truncate sequences to this length (approx 20s at 10fps)

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    PATIENCE = 7  # Early stopping patience

    # Architecture
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True

    # Class Weighting (Background vs Gestures)
    # Background (0) is very frequent, Gestures (1-20) are sparse.
    # We assign a lower weight to background.
    CLASS_WEIGHTS = [0.1] + [1.0] * 20

    # Debugging
    DEBUG_SUBSET_SIZE = (
        0  # Set to > 0 (e.g., 50) to train on a small subset for debugging
    )
