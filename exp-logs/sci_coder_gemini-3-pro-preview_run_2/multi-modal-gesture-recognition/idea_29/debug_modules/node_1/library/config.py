import os
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_29"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission File Path
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Random Seed for Reproducibility
SEED = 42

# Gesture Vocabulary (Name to ID)
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

# ID to Name Mapping
ID_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

# Skeleton Configuration
# 12 Upper-Body Joints
SELECTED_JOINTS = [
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

# Indices assuming the order in the dataset documentation matches the list above for the first 12
SELECTED_JOINT_INDICES = list(range(12))

# Normalization
SCALE_FACTOR = 0.001  # Convert millimeters to meters

# Audio Configuration
AUDIO_SAMPLE_RATE = 16000
N_MFCC = 13
# Hop length will be dynamic based on video FPS to align features,
# or fixed if resampling video.

# =============================================================================
# MODEL ARCHITECTURE (CASG-CN)
# =============================================================================
NUM_CLASSES = 21  # 20 Gestures + 1 Background (Class 0)
INPUT_DIM = len(SELECTED_JOINTS) * 3 * 2 + N_MFCC  # (Joints * 3D * (Pos+Vel)) + Audio
HIDDEN_DIM = 256
NUM_STAGES = 3

# Stage 1: Encoder
LSTM_LAYERS = 2
BIDIRECTIONAL = True

# Stage 2 & 3: Refinement (TCN)
# Monotonically increasing dilation: 1, 2, 4, ..., 512
DILATIONS = [2**i for i in range(10)]  # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
KERNEL_SIZE = 3
DROPOUT = 0.3

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 8
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 4

# Loss Configuration
# Class Weights: 0.1 for Background (0), 1.0 for Gestures (1-20)
CLASS_WEIGHTS = [0.1] + [1.0] * 20

# Multi-Task Loss Weights
LOSS_WEIGHT_CLS = 1.0  # Classification Cross-Entropy
LOSS_WEIGHT_BND = 0.5  # Boundary Binary Cross-Entropy
LOSS_WEIGHT_SMOOTH = 0.15  # Truncated MSE for smoothing

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
