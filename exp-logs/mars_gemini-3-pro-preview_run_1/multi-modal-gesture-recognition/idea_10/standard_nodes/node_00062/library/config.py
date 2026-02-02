import os

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_10"

# Derived directories
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output File Paths
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# Create necessary directories
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Global Constants & Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Data Configuration
# ==========================================
# 20 Gesture Classes + 1 Background Class (Index 0)
NUM_CLASSES = 21
BACKGROUND_CLASS_ID = 0

# Skeleton Data Specs
SKELETON_JOINTS = 20
SKELETON_CHANNELS = 3  # X, Y, Z
SKELETON_INPUT_DIM = SKELETON_JOINTS * SKELETON_CHANNELS  # 60

# Audio Data Specs
AUDIO_N_MFCC = 13
AUDIO_INPUT_DIM = AUDIO_N_MFCC

# Label Mapping (Name -> ID)
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

# Reverse Mapping (ID -> Name)
ID_TO_NAME = {v: k for k, v in LABEL_MAP.items()}

# ==========================================
# Training Hyperparameters
# ==========================================
# Micro-batching strategy
BATCH_SIZE = 8

# Optimization
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05  # Aggressive regularization
GRADIENT_CLIP_VAL = 1.0

# Loss Function Configuration
LABEL_SMOOTHING = 0.1
BACKGROUND_WEIGHT = 0.5  # Weight for class 0 to balance precision/recall

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT_RATE = 0.3
KERNEL_SIZE_STEM = 7  # For 1D Convolution in stems
