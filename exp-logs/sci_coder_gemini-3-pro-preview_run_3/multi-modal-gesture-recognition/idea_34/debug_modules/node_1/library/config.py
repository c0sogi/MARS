import os

# ==========================================
# 1. Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_34")
SUBMISSION_DIR = "./submission"

# Ensure necessary writable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# 2. Data Processing Configuration
# ==========================================
# Temporal Windowing
WINDOW_SIZE = 64
STRIDE_TRAIN = 32  # Moderate stride for training
STRIDE_TEST = 32  # 50% overlap for inference

# Classes
# 0 = Background, 1-20 = Gesture Categories
NUM_CLASSES = 21
BACKGROUND_CLASS_ID = 0

# Skeleton Input Configuration
# 20 Joints * 3 Coords (X,Y,Z) * 3 Derivatives (Pos, Vel, Acc)
NUM_JOINTS = 20
CHANNELS_PER_JOINT = 9
INPUT_DIM_SKELETON = NUM_JOINTS * CHANNELS_PER_JOINT  # 180

# Audio Input Configuration
# Standard MFCC count
N_MFCC = 13
INPUT_DIM_AUDIO = N_MFCC

# Gesture Vocabulary (Name -> ID)
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
# Inverse mapping for decoding
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# ==========================================
# 3. Model Hyperparameters
# ==========================================
# Stage 1: Projected-Gated Encoder
PROJECTION_DIM = 128  # Target dim for modality-specific projections
HIDDEN_DIM = 256  # Bi-GRU Hidden Size (128 per direction * 2)
DROPOUT_ENCODER = 0.3

# Stage 2 & 3: TCN Refinement
DROPOUT_TCN = 0.2
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8, 16]  # Receptive field ~63 frames

# ==========================================
# 4. Training Configuration
# ==========================================
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Optimization & Regularization
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_VAL = 1.0

# Loss Weights
# Background class gets 0.2 weight, others 1.0
CLASS_WEIGHTS = [0.2] + [1.0] * 20
SMOOTHING_LOSS_WEIGHT = 0.5
TRUNCATION_THRESHOLD = 1.0

# ==========================================
# 5. Post-Processing & Inference
# ==========================================
MIN_GESTURE_DURATION = 5  # Minimum frames for a valid gesture
