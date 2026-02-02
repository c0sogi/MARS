import os
import torch

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_36"

# Create working directory if it doesn't exist
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Files
SUBMISSION_PATH = "./submission/submission.csv"
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================================
# Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Data Processing Hyperparameters
# ==========================================
# Sliding Window Strategy
WINDOW_SIZE = 64
STRIDE = 32

# Audio Processing (MFCC)
AUDIO_SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# Skeleton Processing
# 20 Joints
JOINTS_COUNT = 20
# Feature streams: Position, Velocity, Acceleration
# Total raw features before embedding/fusion: 20 * 3 * 3 = 180
USE_LOG_MODULUS = True  # Flag to enable sign(x) * log(1 + |x|)

# Labels
NUM_CLASSES = 21  # 20 Gestures + 1 Background
BACKGROUND_CLASS_ID = 0
GESTURE_CLASS_IDS = list(range(1, 21))

# Label Map
LABEL_MAP = {
    0: "background",
    1: "vattene",
    2: "vieniqui",
    3: "perfetto",
    4: "furbo",
    5: "cheduepalle",
    6: "chevuoi",
    7: "daccordo",
    8: "seipazzo",
    9: "combinato",
    10: "freganiente",
    11: "ok",
    12: "cosatifarei",
    13: "basta",
    14: "prendere",
    15: "noncenepiu",
    16: "fame",
    17: "tantotempo",
    18: "buonissimo",
    19: "messidaccordo",
    20: "sonostufo",
}

# ==========================================
# Model Architecture
# ==========================================
# Stage 1: Encoder
HIDDEN_DIM = 256  # Total hidden dim (128 per direction for Bi-GRU)
ENCODER_DROPOUT = 0.3

# Stage 2 & 3: Refinement (TCN)
DILATIONS = [1, 2, 4, 8, 16]
KERNEL_SIZE = 3
REFINEMENT_DROPOUT = 0.2

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Debugging / Development
DEBUG = False
DEBUG_SUBSET_SIZE = 50  # Number of samples to use if DEBUG is True

# ==========================================
# Loss Function Configuration
# ==========================================
# Class Weights: Background (0) is 0.2, others 1.0
CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
CLASS_WEIGHTS[BACKGROUND_CLASS_ID] = 0.2

# Deep Supervision Weights
LOSS_WEIGHTS = {
    "stage1": 1.0,  # Encoder output
    "stage2": 1.0,  # Refinement 1 output
    "stage3": 1.0,  # Refinement 2 output
}

# Log-Space Smoothing Loss
SMOOTHING_LOSS_WEIGHT = 0.15
SMOOTHING_THRESHOLD = 1.0

# ==========================================
# Post-Processing
# ==========================================
MIN_GESTURE_FRAMES = 5  # Remove gestures shorter than this
