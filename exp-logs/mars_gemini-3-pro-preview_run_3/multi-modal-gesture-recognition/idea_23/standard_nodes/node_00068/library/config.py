import os
import torch

# ==========================================
# Global Configuration for WES-KN (Idea 23)
# ==========================================

# ------------------------------------------
# Reproducibility
# ------------------------------------------
SEED = 42

# ------------------------------------------
# File System Paths
# ------------------------------------------
# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Idea-Specific Paths
IDEA_NAME = "idea_23"
IDEA_DIR = os.path.join(WORKING_DIR, IDEA_NAME)
CACHE_DIR = os.path.join(IDEA_DIR, "cache")
MODEL_SAVE_PATH = os.path.join(IDEA_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Create necessary directories
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ------------------------------------------
# Data Processing Hyperparameters
# ------------------------------------------
# Temporal Windowing
WINDOW_SIZE = 64
STRIDE = 32

# Feature Dimensions
# Skeleton: 20 joints * 3 coords (x,y,z) = 60
# Features: Position (60) + Velocity (60) + Acceleration (60) = 180
SKELETON_INPUT_DIM = 180
# Audio: 13 MFCC coefficients
AUDIO_INPUT_DIM = 13
# Total Input Dimension for Early Fusion
INPUT_DIM = SKELETON_INPUT_DIM + AUDIO_INPUT_DIM  # 193

# Classes
NUM_CLASSES = 21  # 20 Gestures + 1 Background
BACKGROUND_CLASS_ID = 0

# Audio Extraction Settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MFCC = 13
AUDIO_N_FFT = 2048
AUDIO_HOP_LENGTH = 512

# ------------------------------------------
# Model Architecture Hyperparameters
# ------------------------------------------
# Stage 1: Wide-Capacity Kinematic Encoder (Bi-GRU)
HIDDEN_DIM = 128  # Dimension per direction (Total internal = 256)
ENCODER_LAYERS = 1

# Stage 2 & 3: Sawtooth Refinement (TCN)
# Repeated Sawtooth Schedule: [1, 2, 4, 8, 1, 2, 4, 8]
SAWTOOTH_DILATIONS = [1, 2, 4, 8, 1, 2, 4, 8]
KERNEL_SIZE = 3
DROPOUT = 0.3

# ------------------------------------------
# Training Hyperparameters
# ------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3  # Standard Adam LR
WEIGHT_DECAY = 1e-4  # L2 Regularization
PATIENCE = 10  # Early Stopping Patience

# ------------------------------------------
# Loss Function Configuration
# ------------------------------------------
# Weighted Cross Entropy
# Background class (0) gets weight 0.2, others 1.0
CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
CLASS_WEIGHTS[BACKGROUND_CLASS_ID] = 0.2

# Log-Space Smoothing Loss (Truncated MSE)
SMOOTHING_LOSS_WEIGHT = 0.15
SMOOTHING_THRESHOLD = 1.0

# ------------------------------------------
# Hardware & Execution
# ------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
