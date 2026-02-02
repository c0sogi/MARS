import os

# ==========================================
# 1. File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Specific working directory for this idea
IDEA_NAME = "idea_45"
CACHE_DIR = os.path.join(WORKING_DIR, IDEA_NAME)
CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# 2. Data Processing & Augmentation
# ==========================================
# Reproducibility
SEED = 42

# Windowing Strategy (Lesson 00017)
WINDOW_SIZE = 64
STRIDE = 32

# Physical Alignment (Lesson 00082, 00102)
# Scale factor to convert millimeters to meters
SKELETON_SCALE = 0.001

# Audio Features
AUDIO_SR = 16000
N_MFCC = 13
HOP_LENGTH = 512  # Approx 32ms at 16kHz, creates reasonable frame alignment

# Label Configuration
# 20 Gestures + 1 Background class (Index 0)
NUM_CLASSES = 21
BACKGROUND_CLASS_ID = 0

# Mapping from dataset ID (1-20) to Model Index (1-20), 0 reserved for BG
# The dataset labels are 1-based. We will use 0 for background.
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

# ==========================================
# 3. Model Architecture (SH-PAM-CN)
# ==========================================
# Encoder (Lesson 00066, 00110)
# Moderate capacity to prevent bottleneck/overfitting
HIDDEN_SIZE = 96
NUM_GRU_LAYERS = 1
DROPOUT_RATE = 0.4

# Split-Horizon Dilation Schedule (Lesson 00057)
# Restricted schedule to ensure RF < Window/2
DILATION_SCHEDULE = [1, 2, 4, 8]
KERNEL_SIZE = 3

# ==========================================
# 4. Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Loss Function Configuration (Lesson 00010, 00111)
BG_WEIGHT = 0.2
SMOOTHING_LAMBDA = 0.15
SMOOTHING_THRESHOLD = 1.0

# Early Stopping
PATIENCE = 10

# ==========================================
# 5. Inference & Post-Processing
# ==========================================
# Minimum duration to consider a valid gesture (Lesson 00071)
MIN_GESTURE_LENGTH = 5

# Debugging / Development
# Set to a small number (e.g., 100) to run on a subset, or None for full data
DEBUG_SAMPLE_SIZE = None
