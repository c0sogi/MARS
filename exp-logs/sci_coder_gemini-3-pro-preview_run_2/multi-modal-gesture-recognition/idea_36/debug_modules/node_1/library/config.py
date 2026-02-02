import os
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working directory for the current experiment (Idea 36)
WORKING_DIR = "./working/idea_36"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

# Ensure working directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# DATA PARAMETERS
# =============================================================================
SEED = 42

# Gesture Vocabulary (1-based index)
# We will treat 0 as the 'background' class.
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

# Inverse mapping for submission
ID_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

# Total classes: 20 gestures + 1 background = 21
NUM_CLASSES = len(GESTURE_MAP) + 1

# Selected Upper-Body Joints (Indices based on Kinect Skeleton format)
# 1. HipCenter, 2. Spine, 3. ShoulderCenter, 4. Head,
# 5. ShoulderLeft, 6. ElbowLeft, 7. WristLeft, 8. HandLeft,
# 9. ShoulderRight, 10. ElbowRight, 11. WristRight, 12. HandRight
# Note: Kinect indices in the provided MAT files are 1-based or structure fields.
# We will assume a consistent ordering in the parser.
JOINTS_LIST = [
    "HipCenter",
    "Spine",
    "ShoulderCenter",
    "Head",
    "ShoulderLeft",
    "ElbowLeft",
    "WristLeft",
    "HandLeft",
    "ShoulderRight",
    "ElbowRight",
    "WristRight",
    "HandRight",
]

NUM_JOINTS = len(JOINTS_LIST)
# Features per joint: (x, y, z) position + (vx, vy, vz) velocity = 6
# Total Skeleton Features = 12 joints * 6 = 72
SKELETON_FEATURE_DIM = NUM_JOINTS * 6

# Audio Parameters
AUDIO_PARAMS = {
    "sample_rate": 16000,
    "n_mfcc": 13,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 40,
}
# MFCCs (13) + Delta (13) + Delta-Delta (13) = 39 features usually,
# but simple MFCC is 13. Let's assume we stick to basic MFCCs or defined extraction.
AUDIO_FEATURE_DIM = AUDIO_PARAMS["n_mfcc"]

# Total Input Dimension
INPUT_DIM = SKELETON_FEATURE_DIM + AUDIO_FEATURE_DIM

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Stage 1: LSTM Encoder
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
STEM_KERNEL_SIZE = 3

# Stage 2 & 3: MS-TCN
MSTCN_LAYERS = 10
MSTCN_CHANNELS = 256
MSTCN_KERNEL_SIZE = 3
DROPOUT = 0.3

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Loss Weights
# Background (class 0) gets 0.1 weight, Gestures (1-20) get 1.0
CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
CLASS_WEIGHTS[0] = 0.1

# Loss Component Weights
LAMBDA_CLS = 1.0
LAMBDA_BND = 1.0  # Boundary loss weight
LAMBDA_SMOOTH = 0.15  # Smoothness regularization

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_device():
    return torch.device(DEVICE)


def get_class_weights():
    return CLASS_WEIGHTS.to(get_device())
