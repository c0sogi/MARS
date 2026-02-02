import os

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory specific to this idea for caching processed data
WORKING_DIR = "./working/idea_26"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# DATA CONSTANTS
# =============================================================================
SEED = 42

# Gesture Vocabulary: 20 Italian gestures
# ID 0 is reserved for the 'background' / 'no gesture' class
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

# Reverse mapping for decoding predictions
INV_GESTURE_MAP = {v: k for k, v in GESTURE_MAP.items()}

# Total classes = 20 gestures + 1 background
NUM_CLASSES = 21

# Indices for the 12 Upper-Body Joints based on the Kinect skeleton format
# 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head
# 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft
# 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
UPPER_BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Scale factor to convert millimeters to meters for numerical stability
SKELETON_SCALE = 0.001

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
HYPERPARAMS = {
    # Data Processing
    "max_seq_length": 0,  # 0 implies no truncation (dynamic batching) or handle in collate
    "audio_n_mfcc": 13,  # Number of MFCC features to extract
    # Training Loop
    "batch_size": 8,
    "num_epochs": 80,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "gradient_clip_val": 1.0,
    # Architecture: HG-GCRCN
    "hidden_dim": 256,
    "lstm_layers": 2,  # Stage 1: Bi-LSTM layers
    "tcn_layers": 10,  # Stage 2 & 3: MS-TCN layers per stage
    "tcn_kernel_size": 3,
    "dropout": 0.3,
    # Monotonically increasing dilations for TCN: 1, 2, 4, ..., 512
    "tcn_dilations": [2**i for i in range(10)],
    # Loss Weights for Multi-Task Learning
    # L_total = L_cls + L_bnd + L_fg + L_smooth
    "weight_cls": 1.0,
    "weight_bnd": 0.5,  # Boundary supervision
    "weight_fg": 0.5,  # Foreground supervision
    "weight_smooth": 0.15,  # T-MSE probability smoothing
    # Class Weights: 0.1 for Background (index 0), 1.0 for Gestures (indices 1-20)
    # This addresses the background dominance issue.
    "class_weights": [0.1] + [1.0] * 20,
    # T-MSE Threshold (tau) for smoothing loss
    "tmse_threshold": 4.0,
    # Inference
    "median_filter_kernel": 15,  # Size of median filter for post-processing
}
