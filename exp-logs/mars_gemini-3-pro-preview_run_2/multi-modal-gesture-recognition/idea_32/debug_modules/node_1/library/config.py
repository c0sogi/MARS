import os
import torch

# -----------------------------------------------------------------------------
# Global Paths & Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working directory for caching processed data (Parquet/NPY)
WORKING_DIR = "./working/idea_32"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 42

# -----------------------------------------------------------------------------
# Gesture Vocabulary & Labels
# -----------------------------------------------------------------------------
GESTURE_MAP = {
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

# Inverse mapping for decoding predictions
INV_GESTURE_MAP = {v: k for k, v in GESTURE_MAP.items()}

# Total classes: 20 gestures + 1 background (index 0)
NUM_CLASSES = 21

# -----------------------------------------------------------------------------
# Data Processing Configuration
# -----------------------------------------------------------------------------
# Skeleton Joints Selection (Upper Body)
# Indices based on the provided dataset description order:
# 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head
# 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft
# 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
SELECTED_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Feature Extraction
SKELETON_SCALE = 0.001  # Convert Millimeters to Meters
AUDIO_SAMPLE_RATE = 16000
N_MFCC = 13

# Input Dimension Calculation:
# 12 Joints * 3 Coords (Pos) + 12 Joints * 3 Coords (Vel) + 13 MFCCs
INPUT_DIM = (len(SELECTED_JOINTS) * 3) * 2 + N_MFCC

# -----------------------------------------------------------------------------
# Model Architecture Hyperparameters (BMG-CN)
# -----------------------------------------------------------------------------
MODEL_CONFIG = {
    "input_dim": INPUT_DIM,
    "hidden_dim": 256,
    "lstm_layers": 2,  # Stage 1 Backbone
    "num_stages": 3,  # 1 Encoder + 2 Refinement Stages
    "refinement_layers": 10,  # Layers per refinement stage
    "dilation_channels": 256,
    "kernel_size": 3,
    "dropout": 0.3,
    "num_classes": NUM_CLASSES,
}

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
TRAIN_CONFIG = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 8,
    "num_epochs": 50,
    "patience": 10,  # Early Stopping Patience
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
    "gradient_clip": 1.0,
    # Loss Weights
    # Class weights: 0.1 for background (0), 1.0 for gestures (1-20)
    "class_weights": [0.1] + [1.0] * 20,
    # Multi-Task Loss Components
    "lambda_cls": 1.0,  # Classification Loss
    "lambda_bnd": 1.0,  # Boundary Supervision Loss
    "lambda_smooth": 0.15,  # Truncated MSE for Smoothness
    # Debugging: Set to an integer (e.g., 100) to train on a small subset
    "debug_subset_size": None,
}

# -----------------------------------------------------------------------------
# Inference / Post-Processing Configuration
# -----------------------------------------------------------------------------
INFERENCE_CONFIG = {
    "median_window": 7,  # Window size for median filtering predictions
    "pad_mode": "edge",  # Padding mode for filtering to protect boundaries
}
