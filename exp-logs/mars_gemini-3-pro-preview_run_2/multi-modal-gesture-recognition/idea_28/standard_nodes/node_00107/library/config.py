import os

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_28"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# DATA SPECIFICATIONS
# =============================================================================
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

# Inverse mapping for decoding predictions
ID_TO_GESTURE = {v: k for k, v in GESTURE_MAP.items()}

# Class Configuration
# 20 Gestures + 1 Background class (Index 0)
NUM_CLASSES = 21
BACKGROUND_CLASS_ID = 0

# Skeleton Joints Selection (Upper Body)
# Indices correspond to the Kinect format:
# 0: HipCenter, 1: Spine, 2: ShoulderCenter, 3: Head
# 4: ShoulderLeft, 5: ElbowLeft, 6: WristLeft, 7: HandLeft
# 8: ShoulderRight, 9: ElbowRight, 10: WristRight, 11: HandRight
SKELETON_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
NUM_JOINTS = len(SKELETON_JOINTS)
JOINT_DIM = 3  # (x, y, z)

# Normalization
SKELETON_SCALE_FACTOR = 0.001  # Convert millimeters to meters

# Audio Features
AUDIO_SAMPLE_RATE = 16000
NUM_MFCC = 13

# =============================================================================
# MODEL HYPERPARAMETERS (SSG-CRCN)
# =============================================================================
NUM_STAGES = 3
HIDDEN_DIM = 256
NUM_LAYERS = 10  # Number of dilated layers per stage (Stage 2 & 3)
KERNEL_SIZE = 3
DROPOUT = 0.5

# Dilation Schedule: 1, 2, 4, ..., 512
DILATIONS = [2**i for i in range(NUM_LAYERS)]

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 100  # Upper limit, controlled by Early Stopping
PATIENCE = 10  # Early stopping patience

# Optimizer
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Loss Weights
# Strict ratio of 0.1 (Background) : 1.0 (Gesture)
LOSS_WEIGHT_CLS_BG = 0.1
LOSS_WEIGHT_CLS_FG = 1.0

# Smoothing Loss
TMSE_WEIGHT = 0.15  # Truncated MSE weight for probability smoothing

# =============================================================================
# INFERENCE HYPERPARAMETERS
# =============================================================================
MEDIAN_FILTER_KERNEL = 7  # Kernel size for post-processing median filter
