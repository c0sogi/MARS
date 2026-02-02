import os
import torch

# ==========================================
# File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_32"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Model Checkpoint
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# Submission File
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Data Processing Hyperparameters
# ==========================================
# Windowing
WINDOW_SIZE = 64
STRIDE = 32  # Moderate stride to avoid overfitting

# Classes
# 20 gestures + 1 background class (index 0)
NUM_CLASSES = 21
BACKGROUND_CLASS_ID = 0

# Input Features
# 20 joints * 3 coords (pos) + 20*3 (vel) + 20*3 (acc) = 180
# Plus Audio MFCCs (usually 13 or similar, handled dynamically by loader)
# We assume the loader handles the exact input dimension, but we define the skeleton part here.
SKELETON_JOINTS = 20
SKELETON_CHANNELS = 9  # x,y,z pos + x,y,z vel + x,y,z acc

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
# Stage 1: Encoder
HIDDEN_DIM = 128  # Bi-GRU: 64 per direction * 2 = 128
ENCODER_LAYERS = 2  # Number of GRU layers
DROPOUT = 0.3  # Moderate dropout for regularization

# Stage 2 & 3: TCN Refinement
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8, 16]  # Monotonic schedule
TCN_CHANNELS = 64  # Internal channel size for TCN layers

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 60
NUM_WORKERS = 4

# Loss Weights
BACKGROUND_WEIGHT = 0.2  # Weight for class 0 in CrossEntropy
SMOOTHING_LOSS_WEIGHT = 0.15  # Weight for the log-space smoothing loss
SMOOTHING_THRESHOLD = 1.0  # Truncation threshold for smoothing loss

# Deep Supervision Weights (sum to 1.0 or used as is)
LOSS_WEIGHT_STAGE1 = 1.0
LOSS_WEIGHT_STAGE2 = 1.0
LOSS_WEIGHT_STAGE3 = 1.0

# ==========================================
# Post-Processing
# ==========================================
MIN_GESTURE_DURATION = 5  # Minimum frames to keep a gesture prediction
INFERENCE_OVERLAP = 0.5  # 50% overlap for sliding window inference

# ==========================================
# Device Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
