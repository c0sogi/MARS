import os
import random
import numpy as np
import torch

# ==========================================
# File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_2")
SUBMISSION_DIR = "./submission"

# Ensure necessary writable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Data Processing Configuration
# ==========================================
RANDOM_SEED = 42
VIDEO_FPS = 20
AUDIO_SAMPLE_RATE = 16000

# Audio Feature Extraction (MFCC)
# We align audio windows to video frames.
# Hop length = samples_per_sec / frames_per_sec = 16000 / 20 = 800
MFCC_N_MFCC = 13
MFCC_HOP_LENGTH = int(AUDIO_SAMPLE_RATE / VIDEO_FPS)
MFCC_N_FFT = 2048  # Standard window size

# Skeleton Normalization
# Joint index to use as the center (root) for relative coordinates.
# 'HipCenter' is usually a stable root. Assuming index 0 based on common Kinect formats,
# but the code logic should handle name-to-index mapping if possible.
# Here we define the concept; implementation will handle the specific index.
SKELETON_ROOT_JOINT_NAME = "HipCenter"

# ==========================================
# Model Hyperparameters
# ==========================================
# 20 Gesture Classes + 1 Background Class (Index 0)
NUM_CLASSES = 21
INPUT_DIM_SKELETON = 60  # 20 joints * 3 coordinates (x,y,z)
INPUT_DIM_AUDIO = MFCC_N_MFCC
# Total input dimension = Skeleton + Audio
INPUT_DIM = INPUT_DIM_SKELETON + INPUT_DIM_AUDIO

HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True  # Changed to True for better offline recognition

# ==========================================
# Training Configuration
# ==========================================
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_VAL = 1.0

# Class Weighting
# Background class (0) is dominant. We can down-weight it.
# Specific weights can be calculated dynamically or set heuristically.
# 0.1 for background, 1.0 for gestures is a starting point.
LOSS_WEIGHTS = torch.ones(NUM_CLASSES)
LOSS_WEIGHTS[0] = 0.1

# ==========================================
# Label Map
# ==========================================
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

# Inverse map for decoding
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}


# ==========================================
# Utilities
# ==========================================
def seed_everything(seed=RANDOM_SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
