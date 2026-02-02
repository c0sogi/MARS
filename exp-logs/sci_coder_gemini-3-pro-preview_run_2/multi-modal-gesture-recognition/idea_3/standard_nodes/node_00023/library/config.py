import os

# -----------------------------------------------------------------------------
# PATHS & DIRECTORIES
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata file paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache file paths for deterministic loading
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.npz")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.npz")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.npz")

# Model checkpoint and submission paths
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# DATA CONFIGURATION
# -----------------------------------------------------------------------------
SEED = 42

# Gesture Vocabulary Mapping
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
ID_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

# Class definitions
NUM_GESTURES = 20
NUM_CLASSES = NUM_GESTURES + 1  # 0 is background, 1-20 are gestures

# Feature Extraction Dimensions
# Skeleton: 20 joints * 3 coordinates (x, y, z)
NUM_JOINTS = 20
COORDS_PER_JOINT = 3
SKELETON_DIM = NUM_JOINTS * COORDS_PER_JOINT  # 60

# Velocity: First derivative of skeleton coordinates
VELOCITY_DIM = SKELETON_DIM  # 60

# Audio: MFCC features
AUDIO_DIM = 13

# Total Input Dimension for the Model
INPUT_SIZE = SKELETON_DIM + VELOCITY_DIM + AUDIO_DIM  # 133

# Audio Processing
AUDIO_SAMPLE_RATE = 16000
# Approximate video FPS is ~10 based on analysis.
# Hop length for audio to align roughly with video frames: 16000 / 10 = 1600
AUDIO_HOP_LENGTH = 1600

# -----------------------------------------------------------------------------
# MODEL HYPERPARAMETERS
# -----------------------------------------------------------------------------
# Stage 1: Bi-LSTM Encoder
LSTM_HIDDEN_DIM = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2

# Stage 2: TCN Refinement
# Channels for each dilated layer
TCN_NUM_CHANNELS = [64, 64, 64]
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.2

# -----------------------------------------------------------------------------
# TRAINING HYPERPARAMETERS
# -----------------------------------------------------------------------------
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping patience
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

# Class Weights for Loss Function
# Aggressively down-weight background (class 0) to 0.1, keep gestures at 1.0
CLASS_WEIGHTS = [0.1] + [1.0] * NUM_GESTURES

# Augmentation
NOISE_STD = 0.01  # Standard deviation for Gaussian noise injection

# Post-processing
MEDIAN_FILTER_KERNEL = 5  # Kernel size for median filtering predictions
