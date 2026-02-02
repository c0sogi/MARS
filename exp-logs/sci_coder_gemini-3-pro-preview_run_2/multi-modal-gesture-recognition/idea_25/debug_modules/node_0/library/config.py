import os

# -----------------------------------------------------------------------------
# Paths & Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_25"
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary writeable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 42

# -----------------------------------------------------------------------------
# Data Definitions & Preprocessing
# -----------------------------------------------------------------------------
# Gesture Vocabulary Mapping
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

# Inverse Mapping (ID -> Name)
ID_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

# Skeleton Configuration
# Indices based on the dataset description (0-based)
# 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head
# 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft
# 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
UPPER_BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
HIP_CENTER_INDEX = 0

# Edges for Bone Vector computation (Pairs of indices from the original skeleton)
# These define the connectivity for the 12 upper body joints.
SKELETON_EDGES = [
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

# Feature Dimensions
NUM_SELECTED_JOINTS = len(UPPER_BODY_JOINTS)  # 12
NUM_BONES = len(SKELETON_EDGES)  # 11
AUDIO_MFCC_DIM = 13

# Total Input Dimension Calculation:
# (12 joints * 3 pos) + (12 joints * 3 vel) + (11 bones * 3 vec) + 13 audio
# 36 + 36 + 33 + 13 = 118
INPUT_DIM = (
    (NUM_SELECTED_JOINTS * 3)
    + (NUM_SELECTED_JOINTS * 3)
    + (NUM_BONES * 3)
    + AUDIO_MFCC_DIM
)

# Preprocessing Constants
SCALE_FACTOR = 0.001  # Convert millimeters to meters
MAX_SEQ_LEN = 3000  # Maximum sequence length for padding/batching

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
# Stage 1: Geometric Recurrent Encoder
HIDDEN_DIM = 256
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.0  # Dropout between LSTM layers (if > 1)

# Stage 2 & 3: Soft-Gated TCN
TCN_STAGES = 3
TCN_LAYERS = 10
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.2
TCN_CHANNELS = 256
# Monotonically increasing dilation: 1, 2, 4, ..., 512
DILATIONS = [2**i for i in range(TCN_LAYERS)]

# Output Configuration
# 20 Gestures + 1 Background (Index 0)
NUM_CLASSES = 21

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Loss Function Configuration
# Class Weights: 0.1 for Background (index 0), 1.0 for all Gestures (indices 1-20)
CLASS_WEIGHTS = [0.1] + [1.0] * 20

# Multi-task Loss Components
LOSS_LAMBDA_CLS = 1.0  # Weight for Classification Cross-Entropy
LOSS_LAMBDA_BND = 1.0  # Weight for Boundary Regression MSE
LOSS_LAMBDA_SMOOTH = 0.15  # Weight for Probability Smoothing (T-MSE)

# Soft Boundary Generation
GAUSSIAN_SIGMA = 2.0  # Standard deviation for Gaussian kernels at transition frames

# -----------------------------------------------------------------------------
# Debugging & Caching
# -----------------------------------------------------------------------------
# Set to an integer (e.g., 50) to train on a small subset for debugging, or None for full training
DEBUG_SUBSET_SIZE = None

# Caching Configuration
CACHE_DATA = True
CACHE_FILE_PREFIX = "data_cache"
