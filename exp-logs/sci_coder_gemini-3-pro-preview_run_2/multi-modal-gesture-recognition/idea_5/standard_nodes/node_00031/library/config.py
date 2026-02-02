import os
import torch

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Gesture Vocabulary
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

# Skeleton Processing
# Indices based on the dataset description order:
# 0:HipCenter, 1:Spine, 2:ShoulderCenter, 3:Head,
# 4:ShoulderLeft, 5:ElbowLeft, 6:WristLeft, 7:HandLeft,
# 8:ShoulderRight, 9:ElbowRight, 10:WristRight, 11:HandRight
UPPER_BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
REF_JOINT_INDEX = 0  # HipCenter for normalization

DATA_PARAMS = {
    "selected_joints": UPPER_BODY_JOINTS,
    "ref_joint": REF_JOINT_INDEX,
    "use_velocity": True,  # Concatenate first derivative of position
    "normalize_skeleton": True,  # Relative to Ref Joint
}

# Audio Processing
AUDIO_PARAMS = {
    "sample_rate": 16000,
    "n_mfcc": 13,
    "n_fft": 2048,
    "hop_length": 512,  # Will be adjusted dynamically or resampled to match video FPS
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Calculation of Input Dimension:
# Skeleton: 12 joints * 3 coordinates * 2 (pos + vel) = 72 features
# Audio: 13 MFCCs = 13 features
# Total Input Dim = 85
INPUT_DIM = len(UPPER_BODY_JOINTS) * 3 * 2 + AUDIO_PARAMS["n_mfcc"]
NUM_CLASSES = 21  # 20 gestures + 1 background (index 0)

MODEL_PARAMS = {
    "input_dim": INPUT_DIM,
    "num_classes": NUM_CLASSES,
    # Stage 1: Bi-LSTM Encoder
    "lstm_hidden_dim": 256,
    "lstm_layers": 2,
    "lstm_dropout": 0.5,
    # Stage 2: MS-TCN Refinement
    "tcn_num_stages": 2,
    "tcn_num_layers": 10,  # Layers per stage
    "tcn_num_f_maps": 64,  # Feature maps
    "tcn_kernel_size": 3,
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Class Weights: 0.1 for Background (0), 1.0 for Gestures (1-20)
# This addresses background collapse while maintaining sensitivity to gestures.
CLASS_WEIGHTS = [0.1] + [1.0] * 20

TRAIN_PARAMS = {
    "num_epochs": 40,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "patience": 40,  # Disable early stopping effectively, rely on Cosine Scheduler
    "grad_clip": 5.0,  # Gradient clipping value
    "class_weights": CLASS_WEIGHTS,
    "noise_std": 0.01,  # Gaussian noise std dev for augmentation
    "tmse_weight": 5.0,  # Increased weight for T-MSE smoothing loss
    "tmse_threshold": 0.1,  # Threshold for Truncated MSE (Cite Lesson 00029)
}

# =============================================================================
# POST-PROCESSING CONFIGURATION
# =============================================================================
POST_PROCESS_PARAMS = {
    "median_window": 7,  # Kernel size for median filter smoothing
    "pad_mode": "nearest",  # Nearest-neighbor padding (replicate boundary)
}
