import os

# ==========================================
# 1. File System & Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_39")
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# 2. Data Processing Hyperparameters
# ==========================================
# Sliding Window Configuration
WINDOW_SIZE = 64
STRIDE = 32  # Moderate stride to avoid overfitting

# Input Dimensions
NUM_JOINTS = 20
CHANNELS_PER_JOINT = 9  # 3 (Pos) + 3 (Vel) + 3 (Acc)
NUM_MFCC = 13
# Total Input Features: (20 joints * 9 channels) + 13 MFCCs = 193
INPUT_DIM = (NUM_JOINTS * CHANNELS_PER_JOINT) + NUM_MFCC

# Class Configuration
# 20 Gestures + 1 Background Class (Index 0)
NUM_CLASSES = 21
BACKGROUND_CLASS_ID = 0

# ==========================================
# 3. Model Architecture Hyperparameters
# ==========================================
# Stage 1: Adaptive-Scale High-Capacity Encoder
HIDDEN_DIM = 256  # Total hidden units for Bi-GRU (128 per direction)
DROPOUT_ENCODER = 0.5  # High dropout for regularization

# Stage 2 & 3: TCN Refinement
DROPOUT_TCN = 0.3
TCN_KERNEL_SIZE = 3
# Monotonically Increasing Dilation Schedule
TCN_DILATIONS = [1, 2, 4, 8, 16]
TCN_CHANNELS = 64  # Internal channel dimension for TCN blocks

# ==========================================
# 4. Training Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # Adam default
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Loss Function Weights
BACKGROUND_WEIGHT = 0.2  # Downweight background class in CrossEntropy
SMOOTHING_LOSS_WEIGHT = 0.15  # Weight for the log-space smoothing loss
SMOOTHING_THRESHOLD = 1.0  # Truncation threshold for smoothing loss

# ==========================================
# 5. Inference & Post-Processing
# ==========================================
INFERENCE_STRIDE = 32  # 50% overlap for temporal ensembling
MIN_DURATION = 5  # Minimum frame duration for a valid gesture prediction

# ==========================================
# 6. Mappings & Constants
# ==========================================
# Label Map: Name -> ID (1-20)
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

# Reverse Map: ID -> Name
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Ordered list of skeleton joints as per dataset spec
JOINTS_LIST = [
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
