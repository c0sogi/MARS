import os
import torch

# ==========================================
# 1. Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working directory for this specific experiment (Idea 14)
# We ensure these directories exist immediately upon import
WORKING_DIR = "./working/idea_14"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata CSV Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# 2. Reproducibility
# ==========================================
SEED = 42

# ==========================================
# 3. Data Processing Constants
# ==========================================
# Video / Skeleton
FPS = 20
SKELETON_JOINTS = 20
SKELETON_CHANNELS = 3  # (x, y, z)
SKELETON_INPUT_SIZE = SKELETON_JOINTS * SKELETON_CHANNELS  # 60 flattened features

# Audio (Physics-Based Alignment)
SAMPLE_RATE = 16000
N_FFT = 2048
# Hop length calculated to match video frame rate: 16000 Hz / 20 FPS = 800 samples/frame
HOP_LENGTH = 800
N_MFCC = 20
AUDIO_INPUT_SIZE = N_MFCC

# Labels
# 20 Gestures + 1 Background
NUM_CLASSES = 21
BACKGROUND_LABEL = 0

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
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}
ID_TO_NAME[0] = "background"

# ==========================================
# 4. Model Hyperparameters
# ==========================================
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
USE_CONTEXT_GATING = True
USE_GLOBAL_ANCHOR = True

# ==========================================
# 5. Training Hyperparameters
# ==========================================
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Loss Configuration
# Higher weight for background to prevent insertion errors
BACKGROUND_WEIGHT = 0.7
LABEL_SMOOTHING = 0.1

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 6. Inference / Post-Processing
# ==========================================
MEDIAN_FILTER_KERNEL = 5
MIN_GESTURE_LENGTH = 5
