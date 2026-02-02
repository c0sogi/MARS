import os
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working directory for caching processed data (Idea 24)
WORKING_DIR = "./working/idea_24"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# DATA CONSTANTS
# =============================================================================
SEED = 42

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

# Inverse mapping for decoding
INV_GESTURE_MAP = {v: k for k, v in GESTURE_MAP.items()}

# Full Skeleton Joint Map (based on Kinect format provided in description)
FULL_JOINTS_MAP = {
    0: "HipCenter",
    1: "Spine",
    2: "ShoulderCenter",
    3: "Head",
    4: "ShoulderLeft",
    5: "ElbowLeft",
    6: "WristLeft",
    7: "HandLeft",
    8: "ShoulderRight",
    9: "ElbowRight",
    10: "WristRight",
    11: "HandRight",
    12: "HipLeft",
    13: "KneeLeft",
    14: "AnkleLeft",
    15: "FootLeft",
    16: "HipRight",
    17: "KneeRight",
    18: "AnkleRight",
    19: "FootRight",
}

# Selected Upper Body Joints (12 joints)
# We use these indices to extract data from the full skeleton frame
SELECTED_JOINTS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Skeleton Pairs for Bone Vector computation (Parent, Child)
# Indices refer to the subset of selected joints (0-11)
# 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head
# 4-7: Left Arm, 8-11: Right Arm
SKELETON_PAIRS = [
    (0, 1),  # HipCenter -> Spine
    (1, 2),  # Spine -> ShoulderCenter
    (2, 3),  # ShoulderCenter -> Head
    (2, 4),  # ShoulderCenter -> ShoulderLeft
    (4, 5),  # ShoulderLeft -> ElbowLeft
    (5, 6),  # ElbowLeft -> WristLeft
    (6, 7),  # WristLeft -> HandLeft
    (2, 8),  # ShoulderCenter -> ShoulderRight
    (8, 9),  # ShoulderRight -> ElbowRight
    (9, 10),  # ElbowRight -> WristRight
    (10, 11),  # WristRight -> HandRight
]

# Audio Configuration
N_MFCC = 13
AUDIO_SAMPLE_RATE = 16000  # Standard for the dataset
AUDIO_HOP_LENGTH = (
    512  # To align roughly with video FPS if needed, or handled via interpolation
)

# Normalization
SCALE_FACTOR = 0.001  # Convert millimeters to meters
HIP_CENTER_INDEX = 0  # Index in SELECTED_JOINTS_INDICES to use as origin

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Input Dimension Calculation:
# 12 Joints * 3 (XYZ) = 36 (Positions)
# 12 Joints * 3 (XYZ) = 36 (Velocities)
# 11 Bones * 3 (XYZ)  = 33 (Bone Vectors)
# Audio MFCCs         = 13
# Total               = 118
INPUT_DIM = 36 + 36 + 33 + N_MFCC

# Architecture
HIDDEN_DIM = 256
NUM_LAYERS_LSTM = 2
NUM_LAYERS_TCN = 10
KERNEL_SIZE_TCN = 3
DROPOUT = 0.3
NUM_CLASSES = 21  # 0 (Background) + 20 Gestures

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # Early stopping patience

# Loss Weights
# Class 0 (Background) gets 0.1, Classes 1-20 get 1.0
CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
CLASS_WEIGHTS[0] = 0.1

# Component Weights for Total Loss
LOSS_WEIGHT_CLS = 1.0
LOSS_WEIGHT_BND = 1.0  # Boundary regression
LOSS_WEIGHT_SMOOTH = 0.5  # T-MSE smoothing

# Soft Boundary Config
BOUNDARY_SIGMA = 1.5  # Sigma for Gaussian soft targets around transition frames

# =============================================================================
# INFERENCE & POST-PROCESSING
# =============================================================================
MEDIAN_FILTER_KERNEL = 7  # Kernel size for median filtering predictions
MIN_GESTURE_LENGTH = 5  # Minimum frames to consider a valid gesture
