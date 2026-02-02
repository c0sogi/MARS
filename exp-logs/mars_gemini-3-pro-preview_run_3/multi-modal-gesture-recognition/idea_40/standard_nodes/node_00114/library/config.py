import os

# ==========================================
# File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Directories
CACHE_DIR = os.path.join(WORKING_DIR, "idea_40")
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Data Pipeline Hyperparameters
# ==========================================
# Sliding Window Configuration
WINDOW_SIZE = 64
STRIDE = 32

# Audio Feature Extraction
AUDIO_SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# Skeleton Data
# We use raw positions (mm), velocity, and acceleration
# 20 joints * 3 coords * 3 derivatives (pos, vel, acc) = 180 features approx
# Exact dimension depends on implementation of feature extractor
SKELETON_JOINTS = 20

# Labels
NUM_CLASSES = 21  # 0 = Background, 1-20 = Gestures
BACKGROUND_CLASS_ID = 0

# Mapping (for reference and decoding)
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
# Reverse map for decoding
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
# Stage 1: Encoder
HIDDEN_DIM = 128  # Bi-GRU total hidden dimension (64 per direction)

# Stage 2 & 3: TCN Refinement
# Monotonically increasing dilation schedule
DILATION_SCHEDULE = [1, 2, 4, 8, 16]
KERNEL_SIZE = 3
TCN_DROPOUT = 0.2

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30

# Loss Function Weights
BACKGROUND_WEIGHT = 0.2  # Weight for class 0 in CrossEntropy
SMOOTHING_LOSS_WEIGHT = 0.15  # Weight for the Log-Space Smoothing Loss
TRUNCATION_THRESHOLD = 1.0  # Threshold for Truncated MSE

# Debugging / Development
# Set to an integer (e.g., 100) to limit dataset size for rapid debugging
# Set to None for full training
DEBUG_DATA_LIMIT = None

# ==========================================
# Inference & Post-Processing
# ==========================================
# Minimum duration (in frames) to keep a predicted gesture
MIN_GESTURE_DURATION = 5
