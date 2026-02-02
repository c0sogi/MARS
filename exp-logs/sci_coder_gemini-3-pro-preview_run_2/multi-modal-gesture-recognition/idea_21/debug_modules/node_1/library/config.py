import os
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_21")
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure critical directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Map gesture names to IDs (1-20)
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

# Class configuration
# 0: Background/Null
# 1-20: Gesture Categories
NUM_CLASSES = 21

# Joint Selection: 12 Upper-Body Joints
# Indices based on the dataset description order:
# 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head,
# 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft,
# 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Normalization
SCALE_FACTOR = 0.001  # Convert millimeters to meters

# Audio Features
AUDIO_MFCC_N_MFCC = 13

# Debugging / Development
# Set to an integer (e.g., 50) to limit dataset size for rapid testing. Set to None for full run.
DEBUG_SUBSET_SIZE = None

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
TCN_KERNEL_SIZE = 3
NUM_TCN_LAYERS = 10
DROPOUT = 0.2

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8

# Loss Weights
# Background (0) gets 0.1, Gestures (1-20) get 1.0
CLASS_WEIGHTS = [0.1] + [1.0] * 20
BOUNDARY_LOSS_WEIGHT = 1.0
SMOOTHING_LOSS_WEIGHT = 0.15  # Weight for Truncated MSE on probabilities

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
