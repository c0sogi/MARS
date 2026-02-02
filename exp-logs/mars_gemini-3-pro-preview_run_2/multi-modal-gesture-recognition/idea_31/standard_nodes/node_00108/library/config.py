import os
import torch

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for this idea (Idea 31)
WORKING_DIR = "./working/idea_31"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata file paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
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

# Reverse mapping for decoding
ID_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

# Class definitions
# 20 gestures + 1 background class (index 0)
NUM_CLASSES = 21
BACKGROUND_CLASS_ID = 0

# Feature Selection
# Indices for Upper-Body Joints based on dataset description order:
# 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head,
# 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft,
# 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
UPPER_BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
NUM_JOINTS = len(UPPER_BODY_JOINTS)
JOINT_DIM = 3  # (x, y, z)

# Audio Features
NUM_MFCC = 13

# Input Dimension Calculation
# Structure: [Joint Position (12*3)] + [Joint Velocity (12*3)] + [Audio MFCC (13)]
INPUT_DIM = (NUM_JOINTS * JOINT_DIM) + (NUM_JOINTS * JOINT_DIM) + NUM_MFCC

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
HIDDEN_DIM = 256
NUM_STAGES = 3
KERNEL_SIZE = 3

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Loss Weights
# Weighted Cross Entropy: Background gets lower weight to handle imbalance
CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
CLASS_WEIGHTS[0] = 0.1  # Background weight
# Note: Tensor must be moved to device during training

# Multi-task Loss Components
BOUNDARY_LOSS_WEIGHT = 0.5
TMSE_LOSS_WEIGHT = 0.15  # Truncated MSE for probability smoothing

# Early Stopping
PATIENCE = 10

# =============================================================================
# DATA PROCESSING CONSTANTS
# =============================================================================
# Unit conversion: Millimeters to Meters
SCALE_FACTOR = 0.001

# Augmentation Parameters
NOISE_STD = 0.01  # Standard deviation for Gaussian noise
TEMPORAL_FILTER_WIDTH = 5  # For low-pass filtering noise

# Inference
MEDIAN_FILTER_KERNEL = 7  # For post-processing smoothing
