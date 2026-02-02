import os
import torch

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORK_DIR = "./working/idea_16"
CACHE_DIR = os.path.join(WORK_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Data Configuration
# ==========================================
FRAME_RATE = 20
AUDIO_SAMPLE_RATE = 16000
# Physics-based alignment: Hop length matches video frame rate (16000 / 20 = 800)
AUDIO_HOP_LENGTH = 800
AUDIO_N_FFT = 2048
AUDIO_N_MELS = 64

# Skeleton Data
SKELETON_JOINTS = 20
SKELETON_CHANNELS = 3  # X, Y, Z

# Label Mapping (0 is reserved for Background/Padding)
LABEL_MAP = {
    "background": 0,
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
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)  # 21 classes total

# ==========================================
# Model Architecture
# ==========================================
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
CNN_KERNEL_SIZE = 7  # For the temporal input stem
PYRAMID_LEVELS = [1, 2, 4]  # Levels for Temporal Pyramid Anchoring
BIDIRECTIONAL = True

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05  # Aggressive regularization
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping patience

# Loss Configuration
LABEL_SMOOTHING = 0.1
BACKGROUND_WEIGHT = 0.7  # Weight for class 0 to prevent insertion errors

# ==========================================
# Augmentation
# ==========================================
# Random Uniform Temporal Scaling factor range
TEMPORAL_SCALE_MIN = 0.8
TEMPORAL_SCALE_MAX = 1.2

# ==========================================
# Debugging / Development
# ==========================================
# Set to an integer (e.g., 50) to train on a small subset for debugging
DEBUG_SUBSET_SIZE = None
DEBUG_EPOCHS = 5

# ==========================================
# Hardware
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
