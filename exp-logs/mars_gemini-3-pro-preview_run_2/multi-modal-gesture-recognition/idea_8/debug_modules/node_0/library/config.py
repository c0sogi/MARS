import os

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"

# Sub-directories for artifacts
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42

# =============================================================================
# DATA & FEATURE CONFIGURATION
# =============================================================================
# Gesture Vocabulary
# 0 is reserved for 'background' / 'silence' / 'no gesture'
# 1-20 are the actual gesture classes
NUM_CLASSES = 21  # 20 gestures + 1 background

GESTURE_MAP = {
    "background": 0,
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

# Skeleton Feature Selection
# We use only the 12 Upper-Body Joints to reduce noise from lower body
# Indices based on Kinect format provided in description
UPPER_BODY_JOINTS = [
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

NUM_SELECTED_JOINTS = len(UPPER_BODY_JOINTS)
# Features per joint: (x, y, z) position + (dx, dy, dz) velocity
FEATURES_PER_JOINT = 6
TOTAL_SKELETON_FEATURES = NUM_SELECTED_JOINTS * FEATURES_PER_JOINT

# Audio Features
N_MFCC = 13
AUDIO_FEATURES = N_MFCC

# Total Input Dimension for the Model
INPUT_DIM = TOTAL_SKELETON_FEATURES + AUDIO_FEATURES

# =============================================================================
# MODEL HYPERPARAMETERS (DSR-CRCN)
# =============================================================================
# Encoder (Bi-LSTM)
HIDDEN_DIM = 256
LSTM_LAYERS = 2
BIDIRECTIONAL = True

# Refinement Stages (MS-TCN)
NUM_STAGES = 2  # Number of refinement stages (Stage 1 + Stage 2)
NUM_LAYERS = 10  # Layers per stage
NUM_F_MAPS = 64  # Feature maps in TCN layers
KERNEL_SIZE = 3  # Kernel size for dilated convolutions
DROPOUT = 0.3  # Dropout rate

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
BATCH_SIZE = 4  # Small batch size due to variable length sequences
LEARNING_RATE = 1e-4  # Lower learning rate for stability
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Loss Function Weights
# Strict weighting to handle class imbalance (Background dominates)
CLASS_WEIGHTS = [0.1] + [1.0] * 20  # Index 0 (BG): 0.1, Indices 1-20: 1.0

# Deep Supervision Weights
LAMBDA_GEN = 1.0  # Weight for Generation Stage (Encoder) Loss
LAMBDA_REF1 = 1.0  # Weight for Refinement Stage 1 Loss
LAMBDA_REF2 = 1.0  # Weight for Refinement Stage 2 Loss

# T-MSE (Truncated Mean Squared Error) Weight for smoothing
TMSE_WEIGHT = 0.15

# Augmentation
GAUSSIAN_NOISE_STD = 0.01  # Std dev for noise injection during training

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================
# Median filter kernel size for post-processing smoothing
MEDIAN_WINDOW = 5
