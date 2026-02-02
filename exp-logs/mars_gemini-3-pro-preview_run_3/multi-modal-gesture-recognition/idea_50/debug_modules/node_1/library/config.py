import os

# ==========================================
# 1. Directory Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_50"

# Sub-directories for artifacts
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
LOG_DIR = os.path.join(WORKING_DIR, "logs")

# Ensure working directories exist
for d in [CACHE_DIR, CHECKPOINT_DIR, SUBMISSION_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# 2. Data Processing & Augmentation
# ==========================================
# Sequence Windowing
WINDOW_SIZE = 64
STRIDE = 32

# Kinematic Features
USE_BONE_VECTORS = True  # Explicit structural feature
USE_VELOCITY = True  # Temporal feature
USE_ACCELERATION = True  # Dynamic feature

# Augmentation & Robustness
NOISE_SIGMA = 0.01  # Gaussian noise injection for sensor jitter robustness
RANDOM_ROTATION = True  # Random Y-axis rotation
RANDOM_SCALE = True  # Random scaling

# Audio Features
AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MFCC = 13

# Decoding
MIN_DURATION = 5  # Minimum frames to keep a gesture segment

# ==========================================
# 3. Model Architecture (SKD-GN)
# ==========================================
# Stage 1: Structural-Kinematic Decoupled-Gated Encoder
GRU_HIDDEN_SIZE = 96  # Units per direction (Total = 192)
GRU_NUM_LAYERS = 2
GRU_DROPOUT = 0.4
BIDIRECTIONAL = True

# Stage 2 & 3: RF-Aligned Monotonic Refinement (TCN)
TCN_CHANNELS = 64  # Hidden channels for TCN
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8, 16]  # Receptive field ~63 frames
TCN_DROPOUT = 0.2

# ==========================================
# 4. Training Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Class Configuration
NUM_GESTURES = 20
NUM_CLASSES = NUM_GESTURES + 1  # +1 for Background (Index 0)
BACKGROUND_CLASS_ID = 0

# Weighted Cross-Entropy
CLASS_WEIGHTS = [1.0] * NUM_CLASSES
CLASS_WEIGHTS[BACKGROUND_CLASS_ID] = 0.2  # Suppress background dominance

# Log-Space Smoothing Loss (Calibration)
SMOOTHING_LAMBDA = 0.15
SMOOTHING_THRESHOLD = 1.0

# Debugging
DEBUG = False
MAX_SAMPLES = None  # Set to int (e.g., 50) to limit dataset for quick debugging

# ==========================================
# 5. Skeleton Topology (Kinect v1)
# ==========================================
# 20 Joints
JOINT_NAMES = [
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
    "HipLeft",
    "KneeLeft",
    "AnkleLeft",
    "FootLeft",
    "HipRight",
    "KneeRight",
    "AnkleRight",
    "FootRight",
]

# Parent-Child Pairs for Bone Vector Calculation
# Format: (Child Index, Parent Index)
SKELETON_PAIRS = [
    (1, 0),  # Spine -> HipCenter
    (2, 1),  # ShoulderCenter -> Spine
    (3, 2),  # Head -> ShoulderCenter
    (4, 2),  # ShoulderLeft -> ShoulderCenter
    (5, 4),  # ElbowLeft -> ShoulderLeft
    (6, 5),  # WristLeft -> ElbowLeft
    (7, 6),  # HandLeft -> WristLeft
    (8, 2),  # ShoulderRight -> ShoulderCenter
    (9, 8),  # ElbowRight -> ShoulderRight
    (10, 9),  # WristRight -> ElbowRight
    (11, 10),  # HandRight -> WristRight
    (12, 0),  # HipLeft -> HipCenter
    (13, 12),  # KneeLeft -> HipLeft
    (14, 13),  # AnkleLeft -> KneeLeft
    (15, 14),  # FootLeft -> AnkleLeft
    (16, 0),  # HipRight -> HipCenter
    (17, 16),  # KneeRight -> HipRight
    (18, 17),  # AnkleRight -> KneeRight
    (19, 18),  # FootRight -> AnkleRight
]

# ==========================================
# 6. Label Mapping
# ==========================================
# Maps string labels to IDs 1-20. 0 is reserved for background.
LABEL_MAP = {
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

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


# ==========================================
# 7. Helper Functions
# ==========================================
def get_skeleton_input_dim():
    """Calculates the total input dimension for the skeleton stream."""
    features_per_joint = 3  # Position (X, Y, Z)
    if USE_BONE_VECTORS:
        features_per_joint += 3
    if USE_VELOCITY:
        features_per_joint += 3
    if USE_ACCELERATION:
        features_per_joint += 3

    return features_per_joint * len(JOINT_NAMES)
