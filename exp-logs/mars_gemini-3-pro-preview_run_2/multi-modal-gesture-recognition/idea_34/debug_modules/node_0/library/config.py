import os

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

# Working directory for the specific idea
WORKING_DIR = os.path.join(BASE_DIR, "working", "idea_34")
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

# Ensure working directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata file paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# DATA DEFINITIONS
# =============================================================================
SEED = 42

# Gesture Vocabulary Mapping
# 0 is reserved for background/null gesture
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
ID_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

# Skeleton Joint Indices (0-based)
# Based on Kinect format:
# 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head,
# 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft,
# 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
UPPER_BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Feature Dimensions
NUM_JOINTS = len(UPPER_BODY_JOINTS)
COORDS_PER_JOINT = 3  # X, Y, Z
AUDIO_FEATURES = 13  # MFCCs (example default, adjustable in feature extraction)
# Input dim = (Joints * 3 pos) + (Joints * 3 vel) + Audio
INPUT_DIM = (NUM_JOINTS * 3) + (NUM_JOINTS * 3) + AUDIO_FEATURES

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
HYPERPARAMS = {
    # Training
    "seed": SEED,
    "batch_size": 8,
    "num_epochs": 80,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "clip_grad_norm": 5.0,
    # Loss Weights
    # Class weights: 0.1 for background (index 0), 1.0 for gestures (indices 1-20)
    "class_weights": [0.1] + [1.0] * 20,
    "loss_weights": {
        "cls": 1.0,  # Classification loss weight
        "bnd": 0.5,  # Boundary regression loss weight
        "smooth": 0.15,  # T-MSE smoothing loss weight
    },
    # Model Architecture: RLSG-CN
    "model": {
        "input_dim": INPUT_DIM,
        "num_classes": 21,  # 20 gestures + 1 background
        # Stage 1: Encoder (Bi-LSTM)
        "lstm_hidden_dim": 256,
        "lstm_layers": 2,
        "lstm_dropout": 0.3,
        # Stage 2 & 3: Refinement (Gated TCN)
        "tcn_num_layers": 10,
        "tcn_channels": 256,
        "tcn_kernel_size": 3,
        "tcn_dropout": 0.3,
    },
    # Data Processing
    "max_frames": 300,  # Truncate/Pad sequences to this length (if batching requires fixed size)
    "target_fps": 20,  # Resample data to this FPS if needed
}
