import os
import torch

# ==========================================
# Global Random Seed
# ==========================================
SEED = 42

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_7"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache & Output Paths
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
OUTPUT_DIR = os.path.join(WORKING_DIR, "outputs")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Configuration
# ==========================================
# Windowing
WINDOW_SIZE = 64
STRIDE = 32  # Overlap stride for training samples

# Classes
# 20 Gestures + 1 Background class (Index 0)
NUM_CLASSES = 21
BACKGROUND_CLASS_ID = 0

# Feature Dimensions
NUM_JOINTS = 20
# Per joint: Position (3) + Velocity (3) + Acceleration (3) = 9
CHANNELS_PER_JOINT = 9
SKELETON_FEATURE_DIM = NUM_JOINTS * CHANNELS_PER_JOINT  # 180

# Audio
AUDIO_MFCC_DIM = 13
AUDIO_SAMPLE_RATE = 16000  # Standard for this dataset

# Total Input Dimension (Early Fusion)
INPUT_DIM = SKELETON_FEATURE_DIM + AUDIO_MFCC_DIM  # 193

# ==========================================
# Model Architecture Configuration
# ==========================================
# Stage 1: Bi-GRU Encoder
GRU_HIDDEN_DIM = 128
GRU_LAYERS = 2
GRU_DROPOUT = 0.3

# Stage 2 & 3: Gated Dilated TCN (Refinement)
# Dilation factors: 1, 2, 4, 8, 16
TCN_NUM_CHANNELS = [64, 64, 64, 64, 64]
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.2

# ==========================================
# Training Configuration
# ==========================================
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # Early stopping patience
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
GRAD_CLIP = 1.0

# Loss Configuration
BACKGROUND_WEIGHT = 0.2
# Construct weight tensor: [0.2, 1.0, 1.0, ..., 1.0]
CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
CLASS_WEIGHTS[BACKGROUND_CLASS_ID] = BACKGROUND_WEIGHT

# Smoothing Loss Weight (MSE on log-probs)
SMOOTHING_LAMBDA = 0.15

# Debugging / Development
# Set to an integer (e.g., 100) to train on a small subset, or None for full training
DEBUG_SUBSET_SIZE = None

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Augmentation Configuration
# ==========================================
# Applied on-the-fly to raw skeleton positions before deriving kinematics
AUGMENT_ROTATION_RANGE = 15.0  # Degrees (Y-axis rotation)
AUGMENT_SCALE_RANGE = 0.1  # +/- 10% scaling

# ==========================================
# Label Mapping (for reference/decoding)
# ==========================================
# Maps internal ID (1-20) to Gesture Name. 0 is Background.
ID_TO_NAME = {
    1: "vattene",
    2: "vieniqui",
    3: "perfetto",
    4: "furbo",
    5: "cheduepalle",
    6: "chevuoi",
    7: "daccordo",
    8: "seipazzo",
    9: "combinato",
    10: "freganiente",
    11: "ok",
    12: "cosatifarei",
    13: "basta",
    14: "prendere",
    15: "noncenepiu",
    16: "fame",
    17: "tantotempo",
    18: "buonissimo",
    19: "messidaccordo",
    20: "sonostufo",
}
