import os
import torch

# ==========================================
# Reproducibility & Hardware
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Number of subprocesses for data loading

# ==========================================
# File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
IDEA_DIR = os.path.join(WORKING_DIR, "idea_20")
CACHE_DIR = IDEA_DIR
SUBMISSION_DIR = "./submission"

# Ensure necessary writeable directories exist
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Paths
MODEL_SAVE_PATH = os.path.join(IDEA_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Processing Hyperparameters
# ==========================================
# Sliding Window Strategy
WINDOW_SIZE = 64
STRIDE = 32

# Class Definitions
# IDs 1-20 are gestures, ID 0 is background
NUM_CLASSES = 21
BACKGROUND_CLASS_ID = 0

# Input Feature Dimensions
SKELETON_JOINTS = 20
# 3 (Position) + 3 (Velocity) + 3 (Acceleration)
SKELETON_CHANNELS = 9
AUDIO_MFCC_DIM = 13

# Total Input Dimension: (20 joints * 9 channels) + 13 audio features = 193
INPUT_DIM = (SKELETON_JOINTS * SKELETON_CHANNELS) + AUDIO_MFCC_DIM

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
# Lightweight Gated-Kinematic Refinement Network (LG-KRN)
HIDDEN_DIM = 64  # Hidden dimension for GRU and Convolutional channels
KERNEL_SIZE = 3  # Kernel size for Temporal Convolutions
DROPOUT = 0.2  # Dropout probability
DILATIONS = [1, 2, 4, 8, 16]  # Dilation schedule for the TCN stages

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10

# Cascaded Loss Configuration
# Weights for the 3 stages: [Stage1(GRU), Stage2(TCN), Stage3(TCN)]
LOSS_STAGE_WEIGHTS = [1.0, 1.0, 1.0]

# Class Imbalance Handling
# Apply lower weight to the background class (0.2) vs 1.0 for gestures
BG_CLASS_WEIGHT = 0.2

# Smoothing Loss (Truncated MSE on Log-Probs)
SMOOTHING_LOSS_WEIGHT = 0.15
TRUNCATION_THRESHOLD = 1.0

# ==========================================
# Debugging & Development
# ==========================================
# Set to an integer (e.g., 100) to train on a small subset of data for quick debugging.
# Set to None for full training.
DEBUG_SUBSET_SIZE = None
