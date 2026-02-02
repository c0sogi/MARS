import os
import torch

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_17"
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

# Ensure writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Label Configuration
# ==========================================
# Mapping from Gesture Name to ID (1-20)
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

# Reverse Mapping
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

# Model Configuration
# We use ID 0 for Background/Null class.
# IDs 1-20 are the actual gestures.
BACKGROUND_CLASS_ID = 0
NUM_GESTURE_CLASSES = 20
# Total output classes = Background (1) + Gestures (20)
MODEL_OUTPUT_CLASSES = NUM_GESTURE_CLASSES + 1

# ==========================================
# Signal Processing & Data
# ==========================================
SEED = 42

# Audio Parameters
AUDIO_SAMPLE_RATE = 16000
VIDEO_FPS = 20
# Physics-Based Alignment:
# We want one audio feature vector per video frame.
# Hop Length = Sample Rate / FPS = 16000 / 20 = 800 samples.
AUDIO_HOP_LENGTH = 800
AUDIO_N_FFT = 2048  # Large window for sufficient overlap
AUDIO_N_MELS = 64  # Dimension of audio features

# Skeleton Parameters
SKELETON_JOINTS = 20
SKELETON_CHANNELS = 3  # X, Y, Z
# Input dimension for skeleton stream: 20 joints * 3 coords = 60
SKELETON_INPUT_DIM = SKELETON_JOINTS * SKELETON_CHANNELS

# ==========================================
# Model Hyperparameters
# ==========================================
# Architecture
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
KERNEL_SIZE = 7  # For Temporal Conv1d in stems

# Optimization
BATCH_SIZE = 8  # Micro-batching strategy
LEARNING_RATE = 1e-3  # Initial learning rate
WEIGHT_DECAY = 0.05  # Aggressive regularization
NUM_EPOCHS = 60  # Total training epochs
PATIENCE = 15  # Early stopping patience

# Loss Configuration
LABEL_SMOOTHING = 0.1
BACKGROUND_WEIGHT = 0.7  # Weight for class 0 to prevent insertion errors

# ==========================================
# Runtime & Debugging
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Number of subprocesses for data loading

# Debugging: Set to an integer (e.g., 50) to train on a small subset.
# Set to None for full training.
DEBUG_SUBSET_SIZE = None
