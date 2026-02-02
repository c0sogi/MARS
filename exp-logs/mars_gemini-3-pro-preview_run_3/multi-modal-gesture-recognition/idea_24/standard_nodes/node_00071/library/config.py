import os
import torch

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_24"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Data Hyperparameters
# ==========================================
# Sliding window parameters
WINDOW_SIZE = 64
STRIDE = 32

# Class definitions
# 20 gestures + 1 background class (index 0)
NUM_CLASSES = 21

# Mapping for reference (Index 0 is background)
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

# Feature dimensions
# Audio MFCCs (usually 12-13) + Skeleton features
# Skeleton: 20 joints * 3 coords * 3 (pos, vel, acc) = 180
# Audio: 13 (approx)
# Total input dim will be calculated dynamically or set here if fixed.
# Assuming standard MFCC=13 + 180 = 193.
# We'll set a default here, but the model should ideally infer or be passed this.
INPUT_DIM = 193

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
# Stage 1: Encoder
HIDDEN_DIM = 128  # Per direction for Bi-GRU (Total 256)
GRU_LAYERS = 2
DROPOUT = 0.3

# Stage 2 & 3: Refinement TCN
# Standard Powers of 2 Schedule
DILATIONS = [1, 2, 4, 8, 16]
KERNEL_SIZE = 3
TCN_CHANNELS = 64

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Loss Function Weights
# Weight for background class (index 0) is 0.2, others 1.0
CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
CLASS_WEIGHTS[0] = 0.2

# Smoothing Loss
SMOOTHING_THRESHOLD = 1.0  # Truncation threshold for log-space smoothing
SMOOTHING_WEIGHT = 0.15

# ==========================================
# Debugging / Development
# ==========================================
# Set to True to use a small subset of data for rapid testing
DEBUG = False
SUBSET_SIZE = 20  # Number of samples to use if DEBUG is True
