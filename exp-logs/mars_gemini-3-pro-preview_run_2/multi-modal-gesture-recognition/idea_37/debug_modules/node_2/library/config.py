import os
import torch

# =============================================================================
# Paths & Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_37"
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# Global Constants
# =============================================================================
SEED = 42
NUM_WORKERS = 4

# =============================================================================
# Data Configuration
# =============================================================================
# Mapping from gesture names to IDs (1-20)
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

# Skeleton Joint Indices
# We strictly select the 12 Upper-Body joints based on the dataset description.
# Indices correspond to: HipCenter(0), Spine(1), ShoulderCenter(2), Head(3),
# ShoulderLeft(4), ElbowLeft(5), WristLeft(6), HandLeft(7),
# ShoulderRight(8), ElbowRight(9), WristRight(10), HandRight(11).
UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Feature Dimensions
NUM_JOINTS = len(UPPER_BODY_INDICES)
# 3 Position (x,y,z) + 3 Velocity (vx,vy,vz)
CHANNELS_PER_JOINT = 6
N_MFCC = 13

# Total Input Dimension: (12 * 6) + 13 = 85
INPUT_DIM = (NUM_JOINTS * CHANNELS_PER_JOINT) + N_MFCC

# Normalization Constants
SKELETON_SCALE = 0.001  # Convert mm to meters

# =============================================================================
# Model & Training Hyperparameters
# =============================================================================
HYPERPARAMS = {
    # Training
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 16,  # Optimized for A100 memory with sequence data
    "num_epochs": 100,
    "patience": 15,  # Early stopping patience
    # Architecture
    "hidden_dim": 256,
    "lstm_layers": 2,
    "tcn_layers": 10,
    "tcn_kernel_size": 3,
    "dropout": 0.3,
    "num_classes": 21,  # 0 (Background) + 20 Gestures
    # Loss Weights
    "w_cls": 1.0,  # Classification weight
    "w_bnd": 1.0,  # Boundary regression weight
    "w_smooth": 0.5,  # Truncated MSE smoothing weight
    # Smoothing Parameters
    # Threshold for Truncated MSE (squared difference limit)
    "tmse_threshold": 0.15,
}

# Class Weights for CrossEntropy
# 0.1 for Background (Index 0), 1.0 for Gestures (Indices 1-20)
CLASS_WEIGHTS = [0.1] + [1.0] * 20
