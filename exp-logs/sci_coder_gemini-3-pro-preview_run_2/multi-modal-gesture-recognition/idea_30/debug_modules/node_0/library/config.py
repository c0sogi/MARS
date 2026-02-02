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

# Working directory for caching and checkpoints
WORKING_DIR = "./working/idea_30"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# DATA SPECIFICATIONS
# =============================================================================
RANDOM_SEED = 42

# Gesture Vocabulary (Name -> ID)
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

# Inverse mapping (ID -> Name)
ID_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

# Full Skeleton Joint Map (Kinect v2)
JOINTS_MAP = {
    "HipCenter": 0,
    "Spine": 1,
    "ShoulderCenter": 2,
    "Head": 3,
    "ShoulderLeft": 4,
    "ElbowLeft": 5,
    "WristLeft": 6,
    "HandLeft": 7,
    "ShoulderRight": 8,
    "ElbowRight": 9,
    "WristRight": 10,
    "HandRight": 11,
    "HipLeft": 12,
    "KneeLeft": 13,
    "AnkleLeft": 14,
    "FootLeft": 15,
    "HipRight": 16,
    "KneeRight": 17,
    "AnkleRight": 18,
    "FootRight": 19,
}

# Feature Selection: 12 Upper-Body Joints
SELECTED_JOINTS = [
    0,  # HipCenter
    1,  # Spine
    2,  # ShoulderCenter
    3,  # Head
    4,  # ShoulderLeft
    5,  # ElbowLeft
    6,  # WristLeft
    7,  # HandLeft
    8,  # ShoulderRight
    9,  # ElbowRight
    10,  # WristRight
    11,  # HandRight
]

# Audio Features
NUM_MFCC = 13

# Input Dimension Calculation
# 12 Joints * (3 Position + 3 Velocity) + 13 MFCCs
INPUT_DIM = (len(SELECTED_JOINTS) * 6) + NUM_MFCC

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Architecture: Masked Channel-Attentive Gated-Cascaded Network (MCAG-CN)
HIDDEN_DIM = 256
NUM_CLASSES = 21  # 20 gestures + 1 background (index 0)
NUM_STAGES = 3
DROPOUT = 0.3

# Dilation schedule for Refinement Stages (Monotonically Increasing)
# Powers of 2 up to 512: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
DILATIONS = [2**i for i in range(10)]

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
BATCH_SIZE = 8
NUM_EPOCHS = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Early Stopping
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-4

# Loss Weights
# Class weights: Background (0) vs Gestures (1-20)
# Strict ratio of 0.1 : 1.0
CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
CLASS_WEIGHTS[0] = 0.1

# Multi-task Loss Components
LOSS_WEIGHTS = {
    "cls": 1.0,  # Classification Cross-Entropy
    "bnd": 1.0,  # Boundary Binary Cross-Entropy
    "smooth": 0.5,  # Truncated MSE for smoothing (Probability Space)
}

# Smoothing Hyperparameters
TMSE_THRESHOLD = 0.15  # Threshold for Truncated MSE

# =============================================================================
# INFERENCE & POST-PROCESSING
# =============================================================================
MEDIAN_FILTER_KERNEL = 7  # Kernel size for label-space smoothing
