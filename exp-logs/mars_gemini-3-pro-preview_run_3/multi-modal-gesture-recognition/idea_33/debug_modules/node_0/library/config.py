import os
import random
import numpy as np
import torch

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working directory for the specific idea
WORKING_DIR = "./working/idea_33"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Processing Hyperparameters
# ==========================================
SEED = 42
WINDOW_SIZE = 64
STRIDE = 32  # For training generation
NUM_CLASSES = 21  # 20 gestures + 1 background
NUM_JOINTS = 20  # Kinect v1 Skeleton
AUDIO_N_MFCC = 13
AUDIO_SAMPLE_RATE = 16000

# Gesture Vocabulary
# 0 is reserved for background/null
GESTURE_MAP = {
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
# Model Architecture Hyperparameters
# ==========================================
# Stage 1: Encoder
HIDDEN_SIZE = 128  # Per direction for Bi-GRU (Total 256)
DROPOUT_ENCODER = 0.3

# Stage 2 & 3: TCN Refinement
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8, 16]
TCN_CHANNELS = 64
DROPOUT_TCN = 0.2

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Loss Weights
BACKGROUND_CLASS_WEIGHT = 0.2
SMOOTHING_LOSS_WEIGHT = 0.15
TRUNCATION_THRESHOLD = 1.0  # For Log-Space Smoothing Loss

# ==========================================
# Inference & Post-Processing
# ==========================================
INFERENCE_STRIDE = 32  # 50% overlap of WINDOW_SIZE (64)
MIN_GESTURE_DURATION = 5  # Frames


# ==========================================
# Utilities
# ==========================================
def seed_everything(seed=SEED):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
