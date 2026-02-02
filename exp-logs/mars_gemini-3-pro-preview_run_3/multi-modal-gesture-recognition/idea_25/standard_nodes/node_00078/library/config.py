import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_25"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure mutable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Data Processing Parameters
# ==========================================
SEED = 42
WINDOW_SIZE = 64
STRIDE = 32
NUM_CLASSES = 21  # 20 gestures + 1 background (index 0)

# Audio Parameters
AUDIO_SAMPLE_RATE = 16000
N_MFCC = 13

# Skeleton Parameters
# 20 joints * 3 coords (x,y,z)
NUM_JOINTS = 20
JOINTS_DIM = 3

# ==========================================
# Model Architecture Parameters
# ==========================================
# Stage 1: Bi-GRU Encoder
HIDDEN_SIZE = 128  # Hidden units per direction (Total = 256)

# Stage 2 & 3: Sawtooth TCN
# Repeated Sawtooth Dilation Schedule as per Lesson 00057
SAWTOOTH_DILATIONS = [1, 2, 4, 8, 1, 2, 4, 8]
KERNEL_SIZE = 3
DROPOUT = 0.2
TCN_HIDDEN_CHANNELS = 64

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # Adam default
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Loss Function Weights
# Weight for the background class (Class 0) to handle imbalance (Lesson 00010)
BACKGROUND_CLASS_WEIGHT = 0.2
# Truncation threshold for Log-Space Smoothing Loss (Lesson 00055)
SMOOTHING_THRESHOLD = 1.0
# Weight for the smoothing loss component
SMOOTHING_LOSS_WEIGHT = 0.5

# ==========================================
# Post-Processing
# ==========================================
# Minimum duration in frames to consider a prediction valid (Lesson 00071)
MIN_GESTURE_DURATION = 5

# ==========================================
# Label Mapping
# ==========================================
# Mapping from gesture name to ID (1-20). 0 is reserved for background.
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

# Reverse mapping for decoding
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
