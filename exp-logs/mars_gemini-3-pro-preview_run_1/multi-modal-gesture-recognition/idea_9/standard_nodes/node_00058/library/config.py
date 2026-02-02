import os
import torch

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_9"

# Sub-directories for artifacts
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Data Configuration
# ==========================================
SEED = 42
VIDEO_FPS = 20
AUDIO_SR = 16000
# Physics-Based Hop Length: SampleRate / VideoFPS = 16000 / 20 = 800
HOP_LENGTH = int(AUDIO_SR / VIDEO_FPS)
N_MFCC = 13

# Label Configuration
# 20 Gesture Classes + 1 Background Class (Index 0)
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

# Inverse mapping for submission generation
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

# ==========================================
# Model Architecture Configuration
# ==========================================
# Stream-specific parameters (Skeleton & Audio)
HIDDEN_DIM_STREAM = 128  # BiGRU hidden size for independent streams
KERNEL_SIZE = 7  # Conv1d kernel size

# Fusion Backbone parameters
HIDDEN_DIM_BACKBONE = 256  # Residual BiGRU hidden size
DROPOUT_RATE = 0.3

# ==========================================
# Training Configuration
# ==========================================
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
BACKGROUND_WEIGHT = 0.5

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_class_weights(device=DEVICE):
    """
    Returns the class weight tensor.
    Background class (0) gets weight 0.5, others get 1.0.
    """
    weights = torch.ones(NUM_CLASSES, device=device)
    weights[BACKGROUND_LABEL] = BACKGROUND_WEIGHT
    return weights
